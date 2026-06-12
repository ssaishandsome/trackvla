#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenTrackVLA 的训练 / 评估 / 推理主入口。

这个文件把最小可运行闭环放在了同一个脚本里：

1. 读取 JSON / JSONL 格式的规划监督数据。
2. 解析当前帧 fine token 和历史帧 coarse token。
3. 将视觉 token 投影到 LLM 隐空间。
4. 插入时间视角指示 token（TVI）。
5. 拼接文本 token、历史视觉 token、当前视觉 token 和可学习 ACT token。
6. 用 ACT 位置的最终隐藏状态预测固定 horizon 的未来轨迹/动作。

这里的实现是偏保守的：
- 训练仍然基于独立样本，而不是跨 batch 递归状态传播；
- 不做序列级 BPTT；
- 历史主要由缓存好的帧 token 表示，而不是在线时序展开。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import os, json, math, argparse, time, csv
from pathlib import Path
from contextlib import nullcontext
from PIL import Image, ImageDraw

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModel
from cache_gridpool import VisionFeatureCacher, VisionCacheConfig, grid_pool_tokens, adapt_siglip_grid


# ----------------------- utils -----------------------

# 避免 dataloader worker 里反复出现 tokenizer fork 警告
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_tokens_file(path: str) -> torch.Tensor:
    try:
        obj = torch.load(path, map_location='cpu')
    except Exception:
        # PyTorch 2.6 默认 weights_only=True。
        # 对于我们自己生成、可信的 cache，这里允许回退到完整反序列化。
        try:
            obj = torch.load(path, map_location='cpu', weights_only=False)
        except Exception as e:
            raise e
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, dict):
        for k in ("V", "Vfine", "Vcoarse", "tokens", "feat", "features"):
            if k in obj and isinstance(obj[k], torch.Tensor):
                t = obj[k]
                if t.dim() == 3 and t.size(0) == 1:
                    t = t[0]
                return t.float()
    raise ValueError(f"Unrecognized token file: {path}")


def integrate_actions_to_waypoints(actions: np.ndarray, n_waypoints: int, dt: float = 0.2) -> np.ndarray:
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim == 1: a = a[None, :]
    T, D = a.shape
    vx = a[:, 0].astype(np.float32)
    vy = a[:, 1].astype(np.float32) if D > 1 else np.zeros_like(vx)
    wz = a[:, 2].astype(np.float32) if D > 2 else np.zeros_like(vx)

    x = np.zeros(T, dtype=np.float32)
    y = np.zeros(T, dtype=np.float32)
    th = np.zeros(T, dtype=np.float32)

    for t in range(1, T):
        th[t] = th[t-1] + wz[t-1] * dt
        c, s = np.cos(th[t-1]), np.sin(th[t-1])
        x[t] = x[t-1] + (c * vx[t-1] - s * vy[t-1]) * dt
        y[t] = y[t-1] + (s * vx[t-1] + c * vy[t-1]) * dt

    traj = np.stack([x, y, th], axis=-1)
    if n_waypoints <= 1: return traj[-1:]
    idx = np.linspace(0, T-1, n_waypoints).round().astype(int)
    return traj[idx]


# 兼容 JSON / JSONL / 目录的统一读取入口
def _read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with open(file_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            examples.append(json.loads(s))
    return examples


def load_examples_from_path(train_path: str) -> List[Dict[str, Any]]:
    """从 JSON 列表、JSONL 文件或目录中读取样本。

    - 如果是目录：递归加载其下所有 `.jsonl`
    - 如果是文件：支持 `.json`（顶层是 list）和 `.jsonl`（逐行一个样本）
    """
    p = Path(train_path)
    if p.is_dir():
        jsonl_files = sorted(p.rglob('*.jsonl'))
        if len(jsonl_files) == 0:
            raise FileNotFoundError(f"No .jsonl files found under directory: {train_path}")
        all_items: List[Dict[str, Any]] = []
        for fp in jsonl_files:
            all_items.extend(_read_jsonl_file(str(fp)))
        return all_items
    if p.is_file():
        if p.suffix.lower() == '.jsonl':
            return _read_jsonl_file(str(p))
        if p.suffix.lower() == '.json':
            with open(p, 'r') as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"JSON file must contain a list at top-level: {train_path}")
            return data
        raise ValueError(f"Unsupported file type: {train_path}")
    raise FileNotFoundError(f"Path does not exist: {train_path}")

def _cleanup_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """统一 checkpoint 的参数名前缀，兼容单卡和 DDP。

    - 如果 key 以 `module.` 开头，说明来自 DDP/DataParallel，去掉该前缀
    - 否则保持不变
    """
    if not state_dict:
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

# ----------------------- Sanity checks -----------------------

def _dataset_sanity_report(ds: 'JsonTrackingDataset', cfg: 'TrainConfig', max_items: int = 512):
    try:
        import numpy as _np
        n = min(max_items, len(ds))
        xs, ys, thetas = [], [], []
        mask_cov = []
        yaw_hist_present = 0
        yaw_curr_present = 0
        img_ok = 0
        for i in range(n):
            ex = ds.get_example(i)
            # 这里只检查监督信号本身，避免为了 sanity check 触发图像 token 编码
            if 'waypoints' in ex:
                wp = _np.asarray(ex['waypoints'], dtype=_np.float32)
            elif 'actions' in ex:
                dt = float(ex.get('dt', ds.cfg.default_dt))
                wp = integrate_actions_to_waypoints(_np.asarray(ex['actions'], dtype=_np.float32), cfg.n_waypoints, dt)
            else:
                continue
            xs.append(wp[:, 0])
            if wp.shape[1] >= 2:
                ys.append(wp[:, 1])
            if wp.shape[1] >= 3:
                thetas.append(wp[:, 2])
            # 统计有效 waypoint 掩码覆盖率
            if 'valid_mask' in ex and isinstance(ex['valid_mask'], list):
                mv = _np.asarray(ex['valid_mask'], dtype=bool)
                mask_cov.append(float(mv.mean()))
            elif 'valid_idx' in ex and isinstance(ex['valid_idx'], list):
                mv = _np.zeros(cfg.n_waypoints, dtype=bool)
                idx = _np.asarray(ex['valid_idx'], dtype=int)
                mv[_np.clip(idx, 0, cfg.n_waypoints-1)] = True
                mask_cov.append(float(mv.mean()))
            # 统计是否存在 yaw 相关字段
            if 'yaw_hist' in ex:
                yaw_hist_present += 1
            if 'yaw_curr' in ex:
                yaw_curr_present += 1
            # 快速检查当前图像路径是否存在
            cur_rel = Path(ex.get('current', ''))
            if str(cur_rel):
                cur_abs = cur_rel if cur_rel.is_absolute() else (ds.base_root / cur_rel)
                if cur_abs.exists():
                    img_ok += 1
        if xs:
            x = _np.concatenate(xs)
            x_mu, x_sd = float(_np.mean(x)), float(_np.std(x))
        else:
            x_mu = x_sd = float('nan')
        if ys:
            y = _np.concatenate(ys)
            y_mu, y_sd = float(_np.mean(y)), float(_np.std(y))
        else:
            y_mu = y_sd = float('nan')
        th_sd = float(_np.std(_np.concatenate(thetas))) if thetas else float('nan')
        cov_mu = float(_np.mean(mask_cov)) if mask_cov else float('nan')
        print(f"[SANITY] GT x(mean={x_mu:.3f}, std={x_sd:.3f}) y(mean={y_mu:.3f}, std={y_sd:.3f}) theta(std={th_sd:.3f}) | mask_cov_mean={cov_mu:.3f}")
        print(f"[SANITY] yaw_hist_present={yaw_hist_present}/{n} yaw_curr_present={yaw_curr_present}/{n} | current_img_exists={img_ok}/{n}")
        # 给一些简单但实用的数据告警
        if _np.isfinite(y_sd) and y_sd < 0.05:
            print("[SANITY][warn] GT 横向变化很小，模型可能倾向于学习近似直线。")
        if _np.isfinite(cov_mu) and cov_mu < 0.2:
            print("[SANITY][warn] 有效 waypoint 很稀疏，训练监督可能偏弱。")
    except Exception as _e:
        print(f"[SANITY] skipped due to error: {_e}")


# ----------------------- TVI + projector + planner -----------------------

class TVIEmbedder(nn.Module):
    """时间/视角指示器（TVI）。

    它会在每一帧视觉 token 前插入辅助 token，告诉 LLM：
    - 这是第几帧
    - 它属于 history 还是 current
    - 是否还要额外编码朝向角

    - make_time_token(t, kind_id, view_id)
    - make_angle_token(theta, kind_id, view_id) -> 用 [sinθ, cosθ] 投影得到
    kind_id: 0 = coarse/history, 1 = fine/current.
    """
    def __init__(self, d_model: int, max_time: int = 4096, max_views: int = 1):
        super().__init__()
        self.time_emb   = nn.Embedding(max_time, d_model)
        self.view_emb   = nn.Embedding(max_views, d_model)
        self.kind_emb   = nn.Embedding(2, d_model)
        self.angle_proj = nn.Linear(2, d_model)

    def make_time_token(self, t_scalar: int, kind_id: int, view_id: int = 0,
                        device: Optional[torch.device] = None) -> torch.Tensor:
        tok = self.time_emb.weight[t_scalar] + self.view_emb.weight[view_id] + self.kind_emb.weight[kind_id]
        return tok.to(device) if device is not None else tok

    def make_angle_token(self, theta: float, kind_id: int, view_id: int = 0,
                         device: Optional[torch.device] = None) -> torch.Tensor:
        # project [sinθ, cosθ] into d_model
        theta = (theta + math.pi) % (2*math.pi) - math.pi
        sincos = torch.tensor([math.sin(theta), math.cos(theta)],
                              dtype=self.angle_proj.weight.dtype,
                              device=device)
        ang = F.linear(sincos, self.angle_proj.weight, self.angle_proj.bias)
        tok = ang + self.view_emb.weight[view_id] + self.kind_emb.weight[kind_id]
        return tok


class CrossModalityProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim), nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x): return self.net(x)


class PlannerHead3L(nn.Module):
    """Three-layer MLP A_θ mapping E_A^T → normalized waypoints â ∈ [-1,1]."""
    def __init__(self, d_model: int, n_waypoints: int, action_dims: int, use_tanh: bool = True):
        super().__init__()
        hid = d_model * 2
        out_dim = n_waypoints * action_dims
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hid), nn.GELU(),
            nn.Linear(hid, hid), nn.GELU(),
            nn.Linear(hid, out_dim)
        )
        self.nw = n_waypoints
        self.ad = action_dims
        self.use_tanh = use_tanh
    def forward(self, act_h: torch.Tensor) -> torch.Tensor:
        y = self.mlp(act_h)
        if self.use_tanh:
            y = torch.tanh(y)                 # bound to [-1,1]
        return y.view(-1, self.nw, self.ad)   # (B, M, D_action)


# ----------------------- Model -----------------------

@dataclass
class ModelConfig:
    llm_name: str = "Qwen/Qwen3-0.6B"
    freeze_llm: bool = False
    n_waypoints: int = 8
    max_time: int = 4096
    beta_nav: float = 10.0
    use_angle_tvi: bool = False     # single-cam default: off
    # Action/target configuration
    use_tanh_actions: bool = True   # allow removing tanh cap via flag
    alpha_xy: Optional[float] = 2.0  # Optional scalar to scale XY only; yaw stays unscaled


class OpenTrackVLA(nn.Module):
    def __init__(self, cfg: ModelConfig, vision_feat_dim: int):
        super().__init__()
        self.cfg = cfg
        # LLM 是这里的跨模态主干。
        # 视觉 token 会先投影到它的隐藏空间，再与文本 token 和 ACT token 拼接。
        self.llm = AutoModel.from_pretrained(cfg.llm_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None)
        self.llm.requires_grad_(not cfg.freeze_llm)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
        self.D = self.llm.config.hidden_size
        # 将拼接后的视觉特征（例如 DINO + SigLIP）投影到 LLM hidden size。
        self.proj = CrossModalityProjector(vision_feat_dim, self.D)
        # Always keep projector trainable regardless of LLM freeze
        self.proj.requires_grad_(True)
        # TVI 显式提供时序标签，帮助 LLM 区分帧顺序。
        self.tvi = TVIEmbedder(self.D, max_time=cfg.max_time)
        # 在序列末尾拼接一个可学习 ACT token。
        # 最终就是拿这个位置的 hidden state 解码未来轨迹。
        self.act_token = nn.Parameter(torch.zeros(1, 1, self.D))
        nn.init.normal_(self.act_token, std=0.02)
        # 当前默认预测 3 维动作/轨迹量，例如 [x, y, yaw]
        action_dims = 3
        self.action_dims = action_dims
        self.planner = PlannerHead3L(self.D, cfg.n_waypoints, action_dims, use_tanh=cfg.use_tanh_actions)
        # Always keep planner trainable regardless of LLM freeze
        self.planner.requires_grad_(True)
        # 若未启用角度 TVI，则冻结对应参数，避免 DDP 误以为这些分支也应产生梯度
        if not cfg.use_angle_tvi:
            for p in self.tvi.angle_proj.parameters():
                p.requires_grad = False
        if cfg.alpha_xy is not None:
            vec = [1.0] * action_dims
            if action_dims >= 2:
                vec[0] = cfg.alpha_xy
                vec[1] = cfg.alpha_xy
            alpha = torch.tensor(vec, dtype=torch.float32).view(1, 1, -1)
        else:
            vec = [1.0] * action_dims
            alpha = torch.tensor(vec, dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("alpha_task", alpha)

    def _embed_text(self, instructions: List[str], device):
        # instructions: 长度为 B 的字符串列表
        # 返回：
        #   emb:  [B, L_txt, D]
        #   mask: [B, L_txt]
        tok = self.tokenizer(instructions, return_tensors='pt', padding=True, truncation=True, max_length=128)
        tok = {k: v.to(device) for k, v in tok.items()}
        emb = self.llm.get_input_embeddings()(tok['input_ids'])
        return emb, tok['attention_mask']

    def _interleave_tvi(self, tokens: torch.Tensor, t_idx: torch.Tensor, kind_id: int,
                        yaw_per_frame: Optional[torch.Tensor] = None, use_angle: bool = False) -> torch.Tensor:
        """在每一帧 token block 前插入 TVI 时间 token（以及可选角度 token）。

        tokens: (B, N, D_llm)
        t_idx: (B, N)，每个 token 属于哪一帧
        yaw_per_frame: (B, F) 或 None
        返回: (B, N + (1 or 2)*F, D_llm)
        """
        B, N, D = tokens.shape
        out_list = []
        for b in range(B):
            tb = t_idx[b]
            xb = tokens[b]
            items = []
            i = 0
            fcount = 0
            while i < N:
                tcur = int(tb[i].item())
                j = i + 1
                while j < N and int(tb[j].item()) == tcur:
                    j += 1
                time_tok = self.tvi.make_time_token(tcur, kind_id, device=xb.device).unsqueeze(0)
                items.append(time_tok)
                if use_angle:
                    theta = 0.0
                    if yaw_per_frame is not None and fcount < yaw_per_frame.size(1):
                        theta = float(yaw_per_frame[b, fcount].item())
                    angle_tok = self.tvi.make_angle_token(theta, kind_id, device=xb.device).unsqueeze(0)
                    items.append(angle_tok)
                items.append(xb[i:j])
                i = j
                fcount += 1
            out_list.append(torch.cat(items, dim=0))
        return torch.stack(out_list, dim=0)

    def forward(self,
                coarse_tokens, coarse_tidx,
                fine_tokens, fine_tidx,
                instructions,
                yaw_hist: Optional[torch.Tensor] = None,
                yaw_curr: Optional[torch.Tensor] = None):
        """预测固定 horizon 的未来轨迹/动作序列。

        Args:
            coarse_tokens: [B, H*Tc, C_v]
                历史帧 token。在本代码里 Tc 通常是每帧池化出的 4 个 token。
            coarse_tidx: [B, H*Tc]
                每个 coarse token 对应的时间编号。同一帧的 token 共享同一个 t。
            fine_tokens: [B, Tf, C_v]
                当前帧的 fine token。在本代码里 Tf 通常为 64。
            fine_tidx: [B, Tf]
                当前帧 fine token 的时间编号，通常统一取 H。
            instructions: list[str] of length B
            yaw_hist: [B, H] or None
            yaw_curr: [B, 1] or None

        Returns:
            tau_pred: [B, M, 3]
                经过 alpha 缩放后的绝对任务空间预测，shape 为 [B, M, 3]。
        """
        device = next(self.parameters()).device
        B = coarse_tokens.size(0)
        # 先投影到 LLM 隐空间
        vis_c = self.proj(coarse_tokens.to(device))   # (B, Nc, D)
        vis_f = self.proj(fine_tokens.to(device))     # (B, Nf, D)
        # 按帧插入 TVI token
        vis_c = self._interleave_tvi(
            vis_c, coarse_tidx.to(device), kind_id=0,
            yaw_per_frame=yaw_hist, use_angle=self.cfg.use_angle_tvi
        )
        vis_f = self._interleave_tvi(
            vis_f, fine_tidx.to(device), kind_id=1,
            yaw_per_frame=yaw_curr, use_angle=self.cfg.use_angle_tvi
        )
        txt_emb, txt_mask = self._embed_text(instructions, device)  # (B, Ltxt, D), (B, Ltxt)
        extra = []
        act = self.act_token.expand(B, 1, -1)
        # 最终序列形式：
        #   [文本 token] + [可选额外 token] + [带 TVI 的历史视觉 token]
        #   + [带 TVI 的当前视觉 token] + [ACT]
        #
        # 这里只从最后的 ACT hidden state 解码，
        # 这样训练目标始终是“根据当前观测一次性预测未来计划”。
        pieces = [txt_emb] + ([extra[0]] if extra else []) + [vis_c, vis_f, act]
        seq = torch.cat(pieces, dim=1).to(self.llm.dtype)
        extra_len = (extra[0].size(1) if extra else 0)
        attn = torch.cat([
            txt_mask,
            torch.ones(B, extra_len + vis_c.size(1) + vis_f.size(1) + 1, dtype=torch.long, device=device)  # +1 for ACT
        ], dim=1)
        out = self.llm(inputs_embeds=seq, attention_mask=attn, output_hidden_states=True, use_cache=False)
        h_act = out.last_hidden_state[:, -1, :]        # E_A^T (ACT is last)
        # 转成 float32，和 planner 的 LayerNorm / Linear 参数 dtype 对齐
        h_act = h_act.float()
        a_hat = self.planner(h_act)                # normalized [-1,1]
        tau_pred = a_hat * self.alpha_task             # absolute units
        return tau_pred


# ----------------------- Dataset -----------------------

@dataclass
class DataConfig:
    train_json: str
    n_waypoints: int = 8
    history: int = 31
    default_dt: float = 0.1
    cache_root: Optional[str] = None


class JsonTrackingDataset(Dataset):
    def __init__(self, cfg: DataConfig):
        super().__init__()
        self.cfg = cfg
        p = Path(cfg.train_json)
        original_candidate = p if p.is_dir() else p.parent
        # 自动推断数据根目录，用来把 JSON 里的相对路径补成绝对路径
        # 常见布局是 <root>/frames/...，所以这里会一直向上找到 frames 所在层
        candidate = original_candidate
        max_up = 4
        while max_up >= 0 and not (candidate / 'frames').exists():
            if candidate.parent == candidate:
                break
            candidate = candidate.parent
            max_up -= 1
        # 如果一直没找到全局 frames 目录，就退回到用户给定的数据根本身。
        # 这对形如 <root>/<scene>/<episode>/frames/000000.jpg 的 EVTBenchmark 结构更合理。
        if not (candidate / 'frames').exists():
            candidate = original_candidate
        self.base_root = candidate
        # token 缓存根目录，默认放在 <base_root>/vision_cache
        self.cache_root = Path(cfg.cache_root) if cfg.cache_root is not None else (self.base_root / "vision_cache")
        # 当缓存不存在时，才会懒加载在线编码器
        self._online_encoder: Optional[VisionFeatureCacher] = None
        # 数据既支持一次性加载 JSON，也支持对 JSONL 建立惰性索引
        self._lazy = False
        self._index: Optional[List[Tuple[str, int]]] = None  # list of (file_path, byte_offset) per example
        self.examples: Optional[List[Dict[str, Any]]] = None
        if p.is_file() and p.suffix.lower() == '.json':
            data = load_examples_from_path(cfg.train_json)
            assert isinstance(data, list) and len(data) > 0, "JSON file must contain a non-empty list"
            self.examples = data
        else:
            # 为 .jsonl 文件（或目录下所有 .jsonl）建立字节级索引，避免一次性全读入内存
            files: List[Path] = []
            if p.is_file() and p.suffix.lower() == '.jsonl':
                files = [p]
            elif p.is_dir():
                files = sorted(p.rglob('*.jsonl'))
            if len(files) == 0:
                raise FileNotFoundError(f"No .jsonl files found under: {cfg.train_json}")
            self._lazy = True
            self._index = []
            for fp in files:
                try:
                    with open(fp, 'rb') as f:
                        pos = 0
                        while True:
                            line = f.readline()
                            if not line:
                                break
                            if line.strip():
                                self._index.append((str(fp), pos))
                            pos += len(line)
                except Exception as _e:
                    raise _e
            if len(self._index) == 0:
                raise RuntimeError(f"No examples indexed from .jsonl sources under: {cfg.train_json}")
        # 历史长度固定为 H；每个样本都会被裁剪/补齐到这个长度
        # 这样 batch 内所有样本的历史维度严格一致
        H_target = int(self.cfg.history)
        self.coarse_frames = H_target

    def __len__(self):
        if self.examples is not None:
            return len(self.examples)
        if self._index is not None:
            return len(self._index)
        return 0

    def _load_tokens(self, path: str) -> torch.Tensor:
        return load_tokens_file(path)

    def _get_online_encoder(self) -> VisionFeatureCacher:
        if self._online_encoder is None:
            # Use CPU when running with multiple workers to avoid GPU contention
            from torch.utils.data import get_worker_info
            worker_info = get_worker_info()
            use_cuda = torch.cuda.is_available() and (worker_info is None)
            cfg = VisionCacheConfig(image_size=384, batch_size=8, device=('cuda' if use_cuda else 'cpu'))
            self._online_encoder = VisionFeatureCacher(cfg)
            self._online_encoder.eval()
        return self._online_encoder

    @torch.inference_mode()
    def _encode_image_tokens(self, img_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        # 当缓存 token 不存在时，在线编码图像作为回退方案
        # 返回：
        #   Vcoarse: [4,  C_total]
        #   Vfine:   [64, C_total]
        enc = self._get_online_encoder()
        pil = Image.open(str(img_path)).convert('RGB')
        tok_dino, Hp, Wp = enc._encode_dino([pil])               # (1, P, C_dino)
        tok_sigl = enc._encode_siglip([pil], out_hw=(Hp, Wp))    # (1, P, C_sigl)
        Vt_cat = torch.cat([tok_dino, tok_sigl], dim=-1)         # (1, P, C_total)
        Vfine = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=64)  # (1, 64, C_total)
        Vcoarse = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=4) # (1, 4,  C_total)
        return Vcoarse[0].cpu().float(), Vfine[0].cpu().float()

    def _read_indexed_example(self, idx: int) -> Dict[str, Any]:
        assert self._lazy and self._index is not None
        fp, off = self._index[idx]
        with open(fp, 'rb') as f:
            f.seek(off)
            line = f.readline()
        return json.loads(line.decode('utf-8'))

    def _is_evtbenchmark_raw_example(self, ex: Dict[str, Any]) -> bool:
        """判断当前样本是否来自 EVTBenchmark 的原始采集格式。"""
        return (
            isinstance(ex, dict)
            and 'current' not in ex
            and 'frame_path' in ex
            and 'teacher_action' in ex
        )

    def _collect_evtbenchmark_future_actions(
        self,
        idx: int,
        current_ex: Dict[str, Any],
        horizon: int,
    ) -> List[List[float]]:
        """从同一 episode 的后续样本中动态拼接未来动作序列。

        设计目标：
        - 不修改现有训练主干和 loss 接口；
        - 在 dataset 层把 EVTBenchmark 原始 step-wise 记录整理成固定 horizon 监督；
        - 若 episode 尾部不足 `horizon` 步，则用零动作补齐，保证维度稳定。

        返回：
            actions: 长度为 horizon 的 [[forward, lateral, yaw], ...]
        """
        actions: List[List[float]] = []
        target_horizon = max(1, int(horizon))

        def _normalize_action(raw_action: Any) -> List[float]:
            if not isinstance(raw_action, list):
                return [0.0, 0.0, 0.0]
            vals = [float(v) for v in raw_action[:3]]
            if len(vals) < 3:
                vals = vals + [0.0] * (3 - len(vals))
            return vals

        episode_id = str(current_ex.get('episode_id', ''))
        current_file = None
        if self._lazy and self._index is not None:
            current_file = self._index[idx][0]

        # 直接使用当前 step 的 teacher_action 作为未来序列的第一个动作。
        actions.append(_normalize_action(current_ex.get('teacher_action', [0.0, 0.0, 0.0])))

        if self._lazy and self._index is not None and current_file is not None:
            next_idx = idx + 1
            while len(actions) < target_horizon and next_idx < len(self._index):
                next_file, _ = self._index[next_idx]
                # 超出当前 jsonl 文件，说明已经切到别的 episode shard，停止。
                if next_file != current_file:
                    break
                next_ex = self._read_indexed_example(next_idx)
                if str(next_ex.get('episode_id', '')) != episode_id:
                    break
                if not self._is_evtbenchmark_raw_example(next_ex):
                    break
                actions.append(_normalize_action(next_ex.get('teacher_action', [0.0, 0.0, 0.0])))
                if bool(next_ex.get('episode_over_after_step', False)):
                    break
                next_idx += 1

        # episode 尾部若不足 horizon，用零动作补齐。
        while len(actions) < target_horizon:
            actions.append([0.0, 0.0, 0.0])
        return actions[:target_horizon]

    def _normalize_evtbenchmark_example(self, idx: int, ex: Dict[str, Any]) -> Dict[str, Any]:
        """将 EVTBenchmark 原始样本整理成当前训练代码可直接使用的统一格式。

        原始结构示例：
            {
                "frame_path": "17DRP5sb8fy/4/frames/000000.jpg",
                "instruction": "...",
                "step": 0,
                "tau_gt": [[...], [...], ...],
                ...
            }

        目标结构：
            {
                "current": "...",
                "images": ["历史帧1", "历史帧2", ...],
                "instruction": "...",
                "actions": [[...], [...], ...],
            }
        """
        frame_rel = ex.get('frame_path')
        tau_gt = ex.get('tau_gt')
        if not isinstance(frame_rel, str) or not isinstance(tau_gt, list):
            return ex

        current_rel = Path(frame_rel).as_posix()
        step = int(ex.get('step', 0))
        current_path = Path(current_rel)
        frame_dir = current_path.parent
        suffix = current_path.suffix or ".jpg"

        # 按 step 直接回构历史帧路径，保持与当前采集结构一致：
        # <scene>/<episode>/frames/000000.jpg
        hist_start = max(0, step - self.coarse_frames)
        images = [
            (frame_dir / f"{hist_idx:06d}{suffix}").as_posix()
            for hist_idx in range(hist_start, step)
        ]

        normalized = dict(ex)
        normalized['current'] = current_rel
        normalized['images'] = images
        # 对 EVTBenchmark 原始数据，不再直接使用当前行自带的 tau_gt（通常只有 4 步）。
        # 而是优先基于同一 episode 的连续 teacher_action 动态拼出 cfg.n_waypoints 步 future supervision。
        normalized['actions'] = self._collect_evtbenchmark_future_actions(idx, ex, self.cfg.n_waypoints)
        normalized['tau_gt_raw'] = tau_gt

        # 这个字段在当前训练代码里并不是必须的，但保留下来便于后续做模式区分。
        if 'episode_mode' in ex:
            normalized['mode'] = ex['episode_mode']
        return normalized

    def get_example(self, idx: int) -> Dict[str, Any]:
        if self._lazy:
            ex = self._read_indexed_example(idx)
        else:
            assert self.examples is not None
            ex = self.examples[idx]
        # 兼容 EVTBenchmark 原始采集格式：若样本里没有 current/images，
        # 但有 frame_path/tau_gt，则在线规范化成当前训练接口。
        if self._is_evtbenchmark_raw_example(ex):
            ex = self._normalize_evtbenchmark_example(idx, ex)
        return ex

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.get_example(idx)
        H = self.coarse_frames

        # 先处理当前帧 fine token。
        # 当前帧始终提供更细粒度的视觉信息，直接参与最终预测。
        curr_path = Path(ex['current'])
        abs_curr_img = curr_path if curr_path.is_absolute() else (self.base_root / curr_path)
        try:
            rel_curr = abs_curr_img.relative_to(self.base_root)
        except ValueError:
            rel_curr = abs_curr_img
        curr_token_dir = self.cache_root / rel_curr.parent
        curr_token_name = rel_curr.stem + "_vfine.pt"
        curr_tok_path = curr_token_dir / curr_token_name
        try:    
            fine_tokens = self._load_tokens(str(curr_tok_path))  # (64, C)
        except Exception as e:            
            # Fallback: encode online and save for reuse
            curr_token_dir.mkdir(parents=True, exist_ok=True)
            vc, vf = self._encode_image_tokens(abs_curr_img)
            print (curr_tok_path)
            try:
                torch.save(vf.half(), str(curr_tok_path))
            except Exception as e:
                print (str(e))
                pass
            fine_tokens = vf
        fine_tidx = torch.full((fine_tokens.size(0),), fill_value=H, dtype=torch.long)

        # 处理历史 coarse token，并在左侧做 padding
        # 这里保持非递归训练接口：
        # - 历史仅来自样本自带的 images 列表
        # - 缺失的前缀帧只做 padding，不做递归预测补全
        imgs_src = ex.get('images', [])
        imgs_trim = imgs_src[-H:]
        missing = H - len(imgs_trim)
        coarse_list, coarse_tidx = [], []
        first_tok: Optional[torch.Tensor] = None
        current_vc: Optional[torch.Tensor] = None
        for t in range(H):
            if t < missing:
                # 占位，等拿到第一个真实 token 后再回填
                tok = None
            else:
                img_p = imgs_trim[t - missing]
                # 将图像路径映射到 cache_root 下对应的 coarse token 路径
                rp = Path(img_p)
                abs_img = rp if rp.is_absolute() else (self.base_root / rp)
                try:
                    rel_img = abs_img.relative_to(self.base_root)
                except ValueError:
                    rel_img = abs_img
                token_dir = self.cache_root / rel_img.parent
                token_name = rel_img.stem + "_vcoarse.pt"
                tok_path = token_dir / token_name
                try:
                    tok = self._load_tokens(str(tok_path))
                except Exception as e:
                    # 缓存不存在时，在线编码并尽量写回缓存，方便后续复用
                    token_dir.mkdir(parents=True, exist_ok=True)
                    vc, vf = self._encode_image_tokens(abs_img)
                    try:
                        torch.save(vc.half(), str(tok_path))
                    except Exception:
                        pass
                    tok = vc
                if first_tok is None:
                    first_tok = tok
            # 左侧 padding 的优先级：
            # 1. 用最早可用的历史 token 做 edge padding
            # 2. 若历史完全为空，则退化为当前帧 coarse token
            # 3. 再不行就补零，至少保证 shape 正确
            if tok is None:
                if first_tok is not None:
                    tok = first_tok
                else:
                    # 尝试拿当前帧 coarse token；如果仍失败，就补零，避免整个 epoch 直接中断
                    try:
                        if current_vc is None:
                            # 优先尝试加载当前帧 coarse cache
                            cur_coarse_name = rel_curr.stem + "_vcoarse.pt"
                            cur_coarse_path = curr_token_dir / cur_coarse_name
                            try:
                                current_vc = self._load_tokens(str(cur_coarse_path))
                            except Exception:
                                # 若无 coarse cache，则直接从当前图像在线编码
                                vc_tmp, _ = self._encode_image_tokens(abs_curr_img)
                                current_vc = vc_tmp
                        tok = current_vc
                    except Exception:
                        tok = torch.zeros(4, fine_tokens.size(1), dtype=torch.float32)
            coarse_list.append(tok)
            coarse_tidx.append(torch.full((tok.size(0),), fill_value=t, dtype=torch.long))
        coarse_tokens = torch.cat(coarse_list, dim=0)      # (H*4, C)
        coarse_tidx   = torch.cat(coarse_tidx, dim=0)      # (H*4,)

        # yaw 是可选信息；只有启用 angle TVI 时才会真正参与 forward
        yaw_hist = torch.tensor(ex.get('yaw_hist', [0.0]*H), dtype=torch.float32)            # (H,)
        yaw_curr = torch.tensor(ex.get('yaw_curr', 0.0), dtype=torch.float32).view(1)        # (1,)

        # 监督目标既可以直接给 waypoints，也可以由 actions 积分得到
        if 'waypoints' in ex:
            wp = torch.tensor(ex['waypoints'], dtype=torch.float32)
        else:
            assert 'actions' in ex, "JSON needs either 'waypoints' or 'actions'"
            dt = float(ex.get('dt', self.cfg.default_dt))
            traj = integrate_actions_to_waypoints(np.asarray(ex['actions'], dtype=np.float32), self.cfg.n_waypoints, dt)
            wp = torch.from_numpy(traj)

        # 可选的有效 waypoint 掩码
        if 'valid_mask' in ex:
            vm = torch.tensor(ex['valid_mask'], dtype=torch.bool)
        elif 'valid_idx' in ex:
            vm = torch.zeros(self.cfg.n_waypoints, dtype=torch.bool)
            vm[torch.tensor(ex['valid_idx'], dtype=torch.long)] = True
        else:
            vm = torch.ones(self.cfg.n_waypoints, dtype=torch.bool)

        item: Dict[str, Any] = {
            'coarse_tokens': coarse_tokens,
            'coarse_tidx':   coarse_tidx,
            'fine_tokens':   fine_tokens,
            'fine_tidx':     fine_tidx,
            'yaw_hist':      yaw_hist,     # (H,)
            'yaw_curr':      yaw_curr,     # (1,)
            'waypoints':     wp,           # (M, D_action)
            'valid_mask':    vm,           # (M,)
            'instruction':   ex.get('instruction', 'follow the person'),
            'current_path':  str(abs_curr_img),
        }
        return item


# ----------------------- Loss & Train -----------------------

def mse_masked(pred: torch.Tensor, target: torch.Tensor, mask_waypoints: torch.Tensor) -> torch.Tensor:
    """Mean squared error over selected waypoints (absolute units)."""
    assert pred.shape == target.shape
    B, M, D = pred.shape
    mask = mask_waypoints.view(B, M, 1).expand(B, M, D)
    se = (pred - target).pow(2)
    se = se[mask]
    return se.mean() if se.numel() > 0 else pred.new_tensor(0.0)


def _compute_total_grad_norm(parameters, norm_type: float = 2.0) -> float:
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return 0.0
    device = parameters[0].grad.device
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
        return float(total_norm.item())
    total = torch.zeros([], device=device)
    for p in parameters:
        param_norm = p.grad.detach().data.norm(norm_type)
        total += param_norm.pow(norm_type)
    total = total.pow(1.0 / norm_type)
    return float(total.item())


@dataclass
class TrainConfig:
    train_json: str
    out_dir: str = './ckpts_qwen4'
    n_waypoints: int = 8
    history: int = 31
    llm_name: str = "Qwen/Qwen3-0.6B"
    epochs: int = 1
    batch_size: int = 12
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    mixed_precision: bool = True
    vision_feat_dim: int = 1536
    seed: int = 0
    num_workers: int = 4
    # model
    use_angle_tvi: bool = False
    beta_nav: float = 10.0
    cache_root: Optional[str] = None
    distributed: bool = False
    dist_backend: str = 'nccl'
    alpha_xy: Optional[float] = 2.0
    # logging
    log_every: int = 10
    csv_logging: bool = True
    # trajectory saving
    save_trajectories: bool = False
    traj_subdir: str = 'trajectories'
    # evaluation
    val_json: Optional[str] = None
    eval_every: int = 0
    eval_batches: int = 8
    final_wp_threshold: float = 0.2
    # single-episode evaluation
    episode_json: Optional[str] = None
    episode_eval_every: int = 0
    episode_threshold: float = 0.2
    episode_max_frames: int = 256
    # modeling options
    no_tanh_actions: bool = True
    # checkpoint retention
    max_ckpts: int = 2
    # resume
    resume: bool = False
    resume_ckpt: Optional[str] = None
    # inference
    infer_json: Optional[str] = None
    infer_ckpt: Optional[str] = None
    infer_out: str = './infer_out'
    infer_batches: int = 0
    infer_vis: bool = False
    infer_save_npz: bool = True


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 各 tensor 字段在 __getitem__ 阶段已经被整理成固定 shape，
    # 因此这里可以直接 stack；只有 instruction / current_path 保持 Python 列表形式。
    instr = [b['instruction'] for b in batch]
    return {
        'coarse_tokens': torch.stack([b['coarse_tokens'] for b in batch], dim=0),
        'coarse_tidx':   torch.stack([b['coarse_tidx']   for b in batch], dim=0),
        'fine_tokens':   torch.stack([b['fine_tokens']   for b in batch], dim=0),
        'fine_tidx':     torch.stack([b['fine_tidx']     for b in batch], dim=0),
        'yaw_hist':      torch.stack([b['yaw_hist']      for b in batch], dim=0),   # (B,H)
        'yaw_curr':      torch.stack([b['yaw_curr']      for b in batch], dim=0),   # (B,1)
        'waypoints':     torch.stack([b['waypoints']     for b in batch], dim=0),
        'valid_mask':    torch.stack([b['valid_mask']    for b in batch], dim=0),
        'instruction':   instr,
        'current_path':  [b['current_path'] for b in batch]
    }


def train(cfg: TrainConfig):
    """主训练循环。

    数据流：
        JSON/JSONL -> dataset item
        -> 缓存/在线视觉 token
        -> LLM 条件轨迹预测
        -> masked waypoint loss
        -> 日志 / 可视化 / checkpoint / eval
    """
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    # 分布式训练初始化
    is_cuda = torch.cuda.is_available()
    use_ddp = bool(cfg.distributed) and is_cuda
    if use_ddp:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)

        dist.init_process_group(
            backend=cfg.dist_backend,
            init_method='env://',
            device_id=device,  # or device_id=local_rank
        )

        rank = dist.get_rank()
        world_size = dist.get_world_size()

    else:
        device = torch.device('cuda' if is_cuda else 'cpu')
        local_rank = 0
        rank = 0
        world_size = 1

    ds = JsonTrackingDataset(DataConfig(train_json=cfg.train_json, n_waypoints=cfg.n_waypoints, history=cfg.history, cache_root=cfg.cache_root))
    if rank == 0:
        _dataset_sanity_report(ds, cfg)
    # 若未显式给出 alpha_xy，则从数据分布中自动估计一个 XY 缩放系数
    if cfg.alpha_xy is None:
        try:
            import numpy as _np
            vals = []
            sample_n = min(4000, len(ds))
            for i in range(sample_n):
                ex = ds.get_example(i)
                arr = None
                if 'waypoints' in ex:
                    arr = _np.asarray(ex['waypoints'], dtype=_np.float32)
                elif 'actions' in ex:
                    dt = float(ex.get('dt', ds.cfg.default_dt))
                    arr = integrate_actions_to_waypoints(_np.asarray(ex['actions'], dtype=_np.float32), cfg.n_waypoints, dt)
                if arr is None:
                    continue
                if arr.ndim == 1:
                    arr = arr[None, :]
                if arr.shape[1] >= 2:
                    r = _np.linalg.norm(arr[:, :2], axis=-1)
                    vals.append(r)
            if vals:
                allr = _np.concatenate(vals)
                alpha_est = float(_np.percentile(allr, 95))
                alpha_est = max(alpha_est, 1e-3)
                cfg.alpha_xy = alpha_est
                if rank == 0:
                    print(f"[auto_alpha_xy] alpha_xy set to {alpha_est:.3f} from dataset percentiles ({cfg.train_json})")
        except Exception as _e:
            if rank == 0:
                print(f"[auto_alpha_xy] skipped due to error: {_e}")
    # 自动探测视觉 token 维度，避免配置和真实 cache 维度不一致
    if rank == 0:
        try:
            sample_item = ds[0]
            detected_dim = None
            if 'fine_tokens' in sample_item:
                detected_dim = sample_item['fine_tokens'].shape[-1]
            elif 'coarse_tokens' in sample_item:
                detected_dim = sample_item['coarse_tokens'].shape[-1]
            if detected_dim is not None and detected_dim != cfg.vision_feat_dim:
                print(f"[AUTO_DIM] Detected vision_feat_dim={detected_dim} from dataset (config had {cfg.vision_feat_dim}), updating...")
                cfg.vision_feat_dim = detected_dim
        except Exception as e:
            print(f"[AUTO_DIM] Failed to auto-detect vision_feat_dim: {e}, using config value {cfg.vision_feat_dim}")
    
    # DDP 下把自动探测出的维度同步到所有 rank
    if use_ddp:
        vision_feat_dim_tensor = torch.tensor([cfg.vision_feat_dim], dtype=torch.int32, device=device)
        dist.broadcast(vision_feat_dim_tensor, src=0)
        cfg.vision_feat_dim = int(vision_feat_dim_tensor.item())

    sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=(sampler is None), num_workers=cfg.num_workers,
                    pin_memory=True, collate_fn=collate_batch, sampler=sampler)

    model = OpenTrackVLA(
        ModelConfig(
            llm_name=cfg.llm_name,
            n_waypoints=cfg.n_waypoints,
            beta_nav=cfg.beta_nav,
            use_angle_tvi=cfg.use_angle_tvi,
            use_tanh_actions=(not cfg.no_tanh_actions),
            alpha_xy=cfg.alpha_xy,
        ),
        vision_feat_dim=cfg.vision_feat_dim,
    ).to(device)
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # 统计可训练参数与冻结参数，便于确认当前训练设置是否符合预期
    if rank == 0:
        try:
            from collections import defaultdict
            model_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            total_params = sum(p.numel() for p in model_inspect.parameters())
            trainable_params = sum(p.numel() for p in model_inspect.parameters() if p.requires_grad)
            pct = (trainable_params / max(1, total_params)) * 100.0
            print(f"[PARAMS] total={total_params:,} trainable={trainable_params:,} ({pct:.2f}%)")
            group_counts = defaultdict(lambda: [0, 0])  # [total, trainable]
            for name, p in model_inspect.named_parameters():
                head = name.split('.')[0]
                n = p.numel()
                group_counts[head][0] += n
                if p.requires_grad:
                    group_counts[head][1] += n
            summary = ' '.join([f"{k}:{v[1]}/{v[0]}" for k, v in group_counts.items()])
            print(f"[PARAMS groups] {summary}")
            tn = [n for n, p in model_inspect.named_parameters() if p.requires_grad][:16]
            print(f"[TRAINABLE names (first 16)] {tn}")
        except Exception as _e:
            print(f"[PARAMS] logging skipped due to error: {_e}")

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=cfg.lr, weight_decay=cfg.weight_decay)
    # 混合精度相关设置
    amp_enabled = cfg.mixed_precision and is_cuda
    amp_dtype = torch.bfloat16  # switch to torch.float16 if you want fp16
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_enabled and amp_dtype == torch.float16))

    # 可选：从 checkpoint 恢复训练
    start_epoch = 0
    # 记录恢复时所在 epoch 内已经完成到哪个 batch。
    # 约定：
    # - checkpoint 里的 batch_idx 表示“最近一次已经完成训练并成功保存时的 batch 下标”
    # - resume 时需要从下一个 batch 继续，因此会 skip 掉 [0, batch_idx] 这些 batch
    start_batch_idx = -1
    step = 0
    if cfg.resume:
        try:
            import glob as _glob
            ckpt_path = cfg.resume_ckpt
            if ckpt_path is None:
                pts = sorted(_glob.glob(os.path.join(cfg.out_dir, 'model_epoch*.pt')), key=lambda p: os.path.getmtime(p))
                ckpt_path = pts[-1] if pts else None
            if ckpt_path and os.path.exists(ckpt_path):
                obj = torch.load(ckpt_path, map_location=device)
                msd = obj.get('model_state', None)
                if msd:
                    msd = _cleanup_state_dict_keys(msd)
                    model_to_load = model.module if isinstance(
                        model, torch.nn.parallel.DistributedDataParallel
                    ) else model
                    model_to_load.load_state_dict(msd, strict=False)
                osd = obj.get('optim_state', None)
                if osd:
                    optim.load_state_dict(osd)
                ssd = obj.get('scaler_state', None)
                if ssd and scaler.is_enabled():
                    scaler.load_state_dict(ssd)
                start_epoch = int(obj.get('epoch', 0))
                start_batch_idx = int(obj.get('batch_idx', -1))
                step = int(obj.get('step', 0))
                if rank == 0:
                    print(f"[RESUME] Loaded {ckpt_path} | epoch={start_epoch} batch_idx={start_batch_idx} step={step}")
        except Exception as _e:
            if rank == 0:
                print(f"[RESUME] Skipped due to error: {_e}")

    if rank == 0:
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
        text_log_path = os.path.join(cfg.out_dir, 'train.log')
    else:
        text_log_path = None

    # 如果是 resume，step 会接着之前的编号继续累加
    ema_loss: Optional[float] = None
    ema_nav: Optional[float] = None
    last_log_time = time.time()
    epoch_start_time = last_log_time
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_samples_seen = 0
        if use_ddp and sampler is not None:
            sampler.set_epoch(epoch)
        num_batches = len(dl)
        # 只在 resume 命中的第一个 epoch 里跳过已完成 batch；
        # 后续 epoch 正常从 0 开始。
        epoch_resume_skip = 0
        if cfg.resume and epoch == start_epoch and start_batch_idx >= 0:
            epoch_resume_skip = min(start_batch_idx + 1, num_batches)
        progress = None
        if rank == 0:
            progress = tqdm(
                total=num_batches,
                desc=f"train {epoch}",
                leave=True,
                dynamic_ncols=True,
                initial=epoch_resume_skip,
            )
        for batch_idx, batch in enumerate(dl):
            if epoch_resume_skip and batch_idx < epoch_resume_skip:
                continue
            # 一个 batch 的核心 shape：
            #   coarse_tokens: [B, H*4, C]
            #   fine_tokens:   [B, 64, C]
            #   waypoints:     [B, M, 3]
            coarse_tokens = batch['coarse_tokens'].to(device)
            coarse_tidx   = batch['coarse_tidx'].to(device)
            fine_tokens   = batch['fine_tokens'].to(device)
            fine_tidx     = batch['fine_tidx'].to(device)
            yaw_hist      = batch['yaw_hist'].to(device)   # (B,H)
            yaw_curr      = batch['yaw_curr'].to(device)   # (B,1)
            gt_wp         = batch['waypoints'].to(device)
            valid_mask    = batch['valid_mask'].to(device)
            instr         = batch['instruction']

            optim.zero_grad(set_to_none=True)
            amp_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled) if amp_enabled else nullcontext()
            with amp_ctx:
                tau_pred = model(
                    coarse_tokens, coarse_tidx,
                    fine_tokens, fine_tidx,
                    instr,
                    yaw_hist=yaw_hist if cfg.use_angle_tvi else None,
                    yaw_curr=yaw_curr if cfg.use_angle_tvi else None
                )
                # 做法 A：在归一化空间算 loss
                # 只对 XY 除以 alpha，yaw 保持原尺度
                # 这样能减弱 XY 大数值对 yaw 的压制，同时不改变推理时的输出接口
                model_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                alpha_vec = getattr(model_inspect, 'alpha_task', None)
                if alpha_vec is None:
                    # Fallback: no scaling information; compute loss in absolute space
                    pred_norm = tau_pred
                    gt_norm = gt_wp
                else:
                    # Normalize only XY dims (0,1); leave others (e.g., yaw) unscaled
                    pred_norm = tau_pred
                    gt_norm = gt_wp
                    if pred_norm.size(-1) >= 2 and alpha_vec.size(-1) >= 2:
                        ax = alpha_vec[..., 0:2].clamp_min(1e-6)
                        pred_norm = pred_norm.clone()
                        gt_norm = gt_norm.clone()
                        pred_norm[..., 0:2] = pred_norm[..., 0:2] / ax
                        gt_norm[..., 0:2] = gt_norm[..., 0:2] / ax
                L_nav = mse_masked(pred_norm, gt_norm, valid_mask)
                L_QA = tau_pred.new_tensor(0.0)
                loss = cfg.beta_nav * L_nav + L_QA
            # 可选：按 step 保存预测轨迹，方便后处理和可视化检查
            if rank == 0 and cfg.save_trajectories:
                try:
                    traj_root = os.path.join(cfg.out_dir, cfg.traj_subdir)
                    os.makedirs(traj_root, exist_ok=True)
                    with torch.no_grad():
                        pred_np = tau_pred.detach().float().cpu().numpy()
                        gt_np = gt_wp.detach().float().cpu().numpy()
                        vm_np = valid_mask.detach().cpu().numpy()
                    Bcur = pred_np.shape[0]
                    for bi in range(Bcur):
                        fpath = os.path.join(traj_root, f"ep{epoch:02d}_st{step+1:06d}_b{bi:03d}.npz")
                        # 文件名使用 step+1，这样和打印日志里的 step 编号对齐
                        np.savez_compressed(
                            fpath,
                            pred=pred_np[bi],
                            gt=gt_np[bi],
                            valid_mask=vm_np[bi],
                            instruction=instr[bi],
                            epoch=epoch,
                            step=step+1
                        )
                except Exception:
                    pass
            scaler.scale(loss).backward()
            grad_norm_before = 0.0
            if cfg.grad_clip is not None:
                scaler.unscale_(optim)
                grad_norm_before = _compute_total_grad_norm([p for p in model.parameters() if getattr(p, 'grad', None) is not None])
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim); scaler.update()

            step += 1
            epoch_samples_seen += coarse_tokens.size(0)
            if rank == 0 and (step % cfg.log_every == 0):
                now = time.time()
                dt = now - last_log_time
                last_log_time = now
                B = coarse_tokens.size(0)
                current_fps = B / max(dt, 1e-6)
                avg_fps = epoch_samples_seen / max(now - epoch_start_time, 1e-6)
                lr = optim.param_groups[0]['lr']
                with torch.no_grad():
                    tp = tau_pred.detach().float()
                    gwp = gt_wp.detach().float()
                    vm = valid_mask.detach().float()
                    # 日志里尽量打印绝对任务空间下的预测量，便于和 GT 直接对比
                    tp_abs = tp
                    try:
                        model_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                        alpha_vec = getattr(model_inspect, 'alpha_task', None)
                        if alpha_vec is not None and alpha_vec.size(-1) >= tp.size(-1):
                            av = alpha_vec.to(tp.device, tp.dtype)
                            tp_abs = tp * av
                    except Exception:
                        pass
                    pred_mean = tp_abs.mean().item()
                    pred_std = tp_abs.std().item()
                    pred_absmax = tp_abs.abs().max().item()
                    gt_mean = gwp.mean().item()
                    gt_std = gwp.std().item()
                    mask_cov = vm.mean().item()
                    mse_total = (tp_abs - gwp).pow(2).mean().item()
                    ad = tp_abs.size(-1)
                    per_dim_mse = []
                    for d in range(min(4, ad)):
                        per_dim_mse.append(float((tp_abs[..., d] - gwp[..., d]).pow(2).mean().item()))
                loss_val = float(loss.detach().item())
                nav_val = float(L_nav.detach().item())
                ema_loss = loss_val if ema_loss is None else (0.98 * ema_loss + 0.02 * loss_val)
                ema_nav = nav_val if ema_nav is None else (0.98 * ema_nav + 0.02 * nav_val)

                mem_alloc_mb = mem_peak_mb = 0.0
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    mem_alloc_mb = torch.cuda.memory_allocated(device) / (1024**2)
                    mem_peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

                metric_parts = [
                    f"[train: {epoch}, {batch_idx + 1:03d} / {num_batches:03d}]",
                    f"FPS: {current_fps:.1f} ({avg_fps:.1f})",
                    f"LR: {lr:.5e}",
                    f"Loss/total: {loss_val:.5f}",
                    f"Loss/total_ema: {ema_loss:.5f}",
                    f"Loss/nav: {nav_val:.5f}",
                    f"Loss/nav_ema: {ema_nav:.5f}",
                    f"Mask/cov: {mask_cov:.5f}",
                    f"Grad/preclip: {grad_norm_before:.5f}",
                    f"Pred/mean: {pred_mean:.5f}",
                    f"Pred/std: {pred_std:.5f}",
                    f"Pred/absmax: {pred_absmax:.5f}",
                    f"GT/mean: {gt_mean:.5f}",
                    f"GT/std: {gt_std:.5f}",
                    f"MSE/total: {mse_total:.5f}",
                    f"MSE/x: {per_dim_mse[0]:.5f}" if len(per_dim_mse) > 0 else "MSE/x: nan",
                    f"MSE/y: {per_dim_mse[1]:.5f}" if len(per_dim_mse) > 1 else "MSE/y: nan",
                    f"MSE/yaw: {per_dim_mse[2]:.5f}" if len(per_dim_mse) > 2 else "MSE/yaw: nan",
                    f"Time/step: {dt:.3f}s",
                    f"Mem/alloc_mb: {mem_alloc_mb:.1f}",
                    f"Mem/peak_mb: {mem_peak_mb:.1f}",
                    f"Step/global: {step}",
                ]
                log_line = "  ,  ".join(metric_parts)
                print(log_line, flush=True)
                if text_log_path is not None:
                    try:
                        with open(text_log_path, 'a', encoding='utf-8') as f:
                            f.write(log_line + "\n")
                    except Exception:
                        pass
                if progress is not None:
                    progress.set_postfix_str(
                        "FPS: "
                        f"{current_fps:.1f} ({avg_fps:.1f})"
                        " | "
                        f"Loss/total: {loss_val:.5f}"
                        " | "
                        f"Loss/nav: {nav_val:.5f}"
                        " | "
                        f"MSE/total: {mse_total:.5f}"
                    )

                # 调试预览：需要时可以打开下面这段，打印 GT / Pred waypoint
                """
                try:
                    import numpy as _np
                    model_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                    alpha_vec = getattr(model_inspect, 'alpha_task', None)
                    pred_abs_b0 = tp[0].detach().cpu().numpy()
                    gt_abs_b0 = gwp[0].detach().cpu().numpy()
                    print("[WAYPOINTS abs][b0] pred=", _np.array2string(pred_abs_b0, precision=3, floatmode='fixed'))
                    print("[WAYPOINTS abs][b0]   gt=", _np.array2string(gt_abs_b0, precision=3, floatmode='fixed'))
                    if alpha_vec is not None and pred_abs_b0.shape[1] >= 2 and alpha_vec.size(-1) >= 2:
                        ax = alpha_vec[0, 0, 0:2].clamp_min(1e-6).detach().float().cpu()
                        pred_n_xy = (tp[0, :, 0:2].detach().cpu() / ax).numpy()
                        gt_n_xy = (gwp[0, :, 0:2].detach().cpu() / ax).numpy()
                        print(f"[WAYPOINTS norm][b0] alpha_xy={ax.numpy().tolist()} pred_xy=", _np.array2string(pred_n_xy, precision=3, floatmode='fixed'))
                        print("[WAYPOINTS norm][b0]   gt_xy=", _np.array2string(gt_n_xy, precision=3, floatmode='fixed'))
                except Exception:
                    pass
                """
                if cfg.csv_logging:
                    csv_path = os.path.join(cfg.out_dir, 'train_log.csv')
                    header = [
                        'epoch','step','lr','loss','loss_ema','L_nav','L_nav_ema','mask_cov',
                        'grad_norm_preclip','step_time','throughput_it_per_s','pred_mean','pred_std','pred_absmax',
                        'gt_mean','gt_std','mse_total','mem_alloc_mb','mem_peak_mb'
                    ] + [f'mse_dim_{i}' for i in range(len(per_dim_mse))]
                    write_header = not os.path.exists(csv_path)
                    try:
                        with open(csv_path, 'a', newline='') as f:
                            w = csv.writer(f)
                            if write_header:
                                w.writerow(header)
                            row = [
                                epoch, step, lr, loss_val, ema_loss, nav_val, ema_nav, mask_cov,
                                grad_norm_before, dt, (B/dt), pred_mean, pred_std, pred_absmax,
                                gt_mean, gt_std, mse_total, mem_alloc_mb, mem_peak_mb
                            ] + per_dim_mse
                            w.writerow(row)
                    except Exception:
                        pass

                # 在当前图像上可视化 GT 与 Pred 轨迹
                if rank == 0 and step % 100 == 0:
                    try:
                        vis_dir = os.path.join(cfg.out_dir, 'vis')
                        os.makedirs(vis_dir, exist_ok=True)
                        with torch.no_grad():
                            # Ensure predictions are in absolute units for visualization
                            pred_draw = tau_pred.detach().float()
                            pred_np = pred_draw.cpu().numpy()
                            gt_np = gwp.detach().float().cpu().numpy()
                            cur_paths = batch.get('current_path', [])
                        Bcur = pred_np.shape[0]
                        for bi in range(min(Bcur, 4)):
                            cur_path = cur_paths[bi] if isinstance(cur_paths, list) and bi < len(cur_paths) else None
                            if cur_path is None or (not os.path.exists(cur_path)):
                                continue
                            pil_img = Image.open(cur_path).convert('RGB')
                            draw = ImageDraw.Draw(pil_img)
                            w, h = pil_img.size
                            base_x = w // 2
                            base_y = int(h * 0.86)
                            def to_pxxy(traj):
                                pts = []
                                for i in range(min(traj.shape[0], 64)):
                                    x, y = float(traj[i, 0]), float(traj[i, 1])
                                    # 机器人坐标里 y 向左为正；映射到屏幕时需要做对应转换
                                    px = base_x - int(y * 120)
                                    py = base_y - int(x * 120)
                                    pts.append((px, py))
                                return pts
                            pts_pred = to_pxxy(pred_np[bi])
                            pts_gt   = to_pxxy(gt_np[bi])
                            # outline
                            for seq, color in ((pts_gt, (0,0,0)), (pts_pred, (0,0,0))):
                                for i2 in range(1, len(seq)):
                                    draw.line([seq[i2-1], seq[i2]], fill=color, width=10)
                            # body
                            for i2 in range(1, len(pts_gt)):
                                draw.line([pts_gt[i2-1], pts_gt[i2]], fill=(255, 200, 0), width=6)
                            for i2 in range(1, len(pts_pred)):
                                draw.line([pts_pred[i2-1], pts_pred[i2]], fill=(0, 255, 200), width=6)
                            # start points
                            if pts_gt:
                                r0 = 6
                                sx, sy = pts_gt[0]
                                draw.ellipse([sx-r0, sy-r0, sx+r0, sy+r0], fill=(255,255,255))
                            if pts_pred:
                                r0 = 6
                                sx, sy = pts_pred[0]
                                draw.ellipse([sx-r0, sy-r0, sx+r0, sy+r0], fill=(0,255,0))
                            try:
                                from inspect import isclass
                                model_cfg = (model.module.cfg if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.cfg)
                            except Exception:
                                pass
                            out_path = os.path.join(vis_dir, f"ep{epoch:02d}_st{step:06d}_b{bi:03d}.jpg")
                            pil_img.save(out_path)
                            print(f"[VIS] saved {out_path}")
                    except Exception:
                        pass

            if step % 100 == 0 and rank == 0:
                ckpt = os.path.join(cfg.out_dir, f"model_epoch{epoch:02d}_step{step:06d}.pt")
                # 始终保存底层模型，而不是 DDP 外壳，避免 state_dict 带 module. 前缀
                model_to_save = model.module if isinstance(
                    model, torch.nn.parallel.DistributedDataParallel
                ) else model

                torch.save(
                {
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'model_state': model_to_save.state_dict(),
                    'optim_state': optim.state_dict(),
                    'scaler_state': (scaler.state_dict() if scaler.is_enabled() else None),
                    'config': cfg.__dict__,
                    'step': step,
                },
                ckpt,
                )
                try:
                    from glob import glob
                    pts = sorted(glob(os.path.join(cfg.out_dir, 'model_epoch*.pt')), key=lambda p: os.path.getmtime(p), reverse=True)
                    if cfg.max_ckpts is not None and cfg.max_ckpts > 0 and len(pts) > cfg.max_ckpts:
                        for old in pts[cfg.max_ckpts:]:
                            try:
                                os.remove(old)
                            except Exception:
                                pass
                except Exception:
                    pass
                if torch.cuda.is_available():
                    try:
                        torch.cuda.reset_peak_memory_stats(device)
                    except Exception:
                        pass

            # 周期性验证（仅 rank 0）
            # 这里故意复用训练阶段的单样本预测接口，而不是切换成真正递归 rollout
            if (cfg.eval_every and (step % cfg.eval_every == 0) and rank == 0 and cfg.val_json):
                try:
                    model_eval = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                    model_eval.eval()
                    with torch.inference_mode():
                        vds = JsonTrackingDataset(DataConfig(train_json=cfg.val_json, n_waypoints=cfg.n_waypoints, history=cfg.history, cache_root=cfg.cache_root))
                        vdl = DataLoader(vds, batch_size=cfg.batch_size, shuffle=False, num_workers=min(2, cfg.num_workers), pin_memory=True, collate_fn=collate_batch)
                        total_mse = 0.0
                        total_count = 0
                        final_errors: List[float] = []
                        hits = 0
                        max_batches = max(1, cfg.eval_batches)
                        bdone = 0
                        for vbatch in vdl:
                            coarse_tokens = vbatch['coarse_tokens'].to(device)
                            coarse_tidx   = vbatch['coarse_tidx'].to(device)
                            fine_tokens   = vbatch['fine_tokens'].to(device)
                            fine_tidx     = vbatch['fine_tidx'].to(device)
                            yaw_hist      = vbatch['yaw_hist'].to(device)
                            yaw_curr      = vbatch['yaw_curr'].to(device)
                            gt_wp         = vbatch['waypoints'].to(device)
                            valid_mask    = vbatch['valid_mask'].to(device)
                            instr         = vbatch['instruction']

                            pred = model_eval(
                                coarse_tokens, coarse_tidx,
                                fine_tokens, fine_tidx,
                                instr,
                                yaw_hist=yaw_hist if cfg.use_angle_tvi else None,
                                yaw_curr=yaw_curr if cfg.use_angle_tvi else None
                            )
                            # 和训练一致：在归一化空间计算 masked MSE
                            model_inspect = model_eval
                            alpha_vec = getattr(model_inspect, 'alpha_task', None)
                            if alpha_vec is not None and pred.size(-1) >= 2 and alpha_vec.size(-1) >= 2:
                                ax = alpha_vec[..., 0:2].clamp_min(1e-6)
                                pred_n = pred.clone()
                                gt_n = gt_wp.clone()
                                pred_n[..., 0:2] = pred_n[..., 0:2] / ax
                                gt_n[..., 0:2] = gt_n[..., 0:2] / ax
                                mse = mse_masked(pred_n, gt_n, valid_mask).item()
                            else:
                                mse = mse_masked(pred, gt_wp, valid_mask).item()
                            total_mse += mse * pred.size(0)
                            total_count += pred.size(0)
                            # 统计最后一个 waypoint 的 EPE 和命中率
                            pred_xy = pred[:, -1, :2].float()
                            gt_xy = gt_wp[:, -1, :2].float()
                            epe = torch.linalg.norm(pred_xy - gt_xy, dim=-1)  # (B,)
                            final_errors.extend(epe.cpu().tolist())
                            hits += (epe <= cfg.final_wp_threshold).sum().item()

                            bdone += 1
                            if bdone >= max_batches:
                                break
                    mean_mse = total_mse / max(1, total_count)
                    if len(final_errors) > 0:
                        import numpy as _np
                        epe_mean = float(_np.mean(final_errors))
                        epe_median = float(_np.median(final_errors))
                        hit_rate = float(hits / len(final_errors))
                    else:
                        epe_mean = epe_median = hit_rate = float('nan')
                    print(f"[VAL] step {step} | masked_MSE={mean_mse:.5f} | final_EPE_mean={epe_mean:.4f} | final_EPE_median={epe_median:.4f} | hit@{cfg.final_wp_threshold}={hit_rate:.3f}", flush=True)
                except Exception as _e:
                    print(f"[VAL] evaluation skipped due to error: {_e}")
                finally:
                    model.train()

            if progress is not None:
                progress.update(1)

            # 单 episode 评估（仅 rank 0）
            # 与 val_json 的 batch 验证不同，这里逐帧走一个 episode，并统计跟随成功率
            if (cfg.episode_eval_every and (step % cfg.episode_eval_every == 0) and rank == 0 and cfg.episode_json):
                try:
                    model_eval = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                    model_eval.eval()
                    with torch.inference_mode():
                        ds_tmp = JsonTrackingDataset(DataConfig(train_json=cfg.episode_json, n_waypoints=cfg.n_waypoints, history=cfg.history, cache_root=cfg.cache_root))
                        if len(ds_tmp) == 0:
                            raise RuntimeError('episode_json produced no examples')
                        max_frames = min(cfg.episode_max_frames, len(ds_tmp))
                        epe_list: List[float] = []
                        hits = 0
                        for i in range(max_frames):
                            item = ds_tmp[i]
                            coarse_tokens = item['coarse_tokens'].unsqueeze(0).to(device)
                            coarse_tidx   = item['coarse_tidx'].unsqueeze(0).to(device)
                            fine_tokens   = item['fine_tokens'].unsqueeze(0).to(device)
                            fine_tidx     = item['fine_tidx'].unsqueeze(0).to(device)
                            yaw_hist      = item['yaw_hist'].unsqueeze(0).to(device)
                            yaw_curr      = item['yaw_curr'].unsqueeze(0).to(device)
                            gt_wp         = item['waypoints'].unsqueeze(0).to(device)
                            instr         = [item['instruction']]

                            pred = model_eval(
                                coarse_tokens, coarse_tidx,
                                fine_tokens, fine_tidx,
                                instr,
                                yaw_hist=yaw_hist if cfg.use_angle_tvi else None,
                                yaw_curr=yaw_curr if cfg.use_angle_tvi else None
                            )
                            pred_xy = pred[:, -1, :2].float()
                            gt_xy = gt_wp[:, -1, :2].float()
                            epe = torch.linalg.norm(pred_xy - gt_xy, dim=-1)  # (1,)
                            e = float(epe.item())
                            epe_list.append(e)
                            if e <= cfg.episode_threshold:
                                hits += 1
                        if len(epe_list) > 0:
                            import numpy as _np
                            epe_mean = float(_np.mean(epe_list))
                            epe_median = float(_np.median(epe_list))
                            follow_rate = float(hits / len(epe_list))
                        else:
                            epe_mean = epe_median = follow_rate = float('nan')
                    print(f"[EPISODE] step {step} | frames={len(epe_list)} | EPE_mean={epe_mean:.4f} | EPE_median={epe_median:.4f} | follow@{cfg.episode_threshold}={follow_rate:.3f}", flush=True)
                except Exception as _e:
                    print(f"[EPISODE] evaluation skipped due to error: {_e}")
                finally:
                    model.train()
        if progress is not None:
            progress.close()

    # 若配置了 infer_json，则训练结束后可直接接着做一次离线推理
    if rank == 0 and cfg.infer_json:
        try:
            _run_inference(cfg)
        except Exception as _e:
            print(f"[INFER] failed: {_e}")

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        print(f"[TRAIN] Finished all epochs. last_step={step}")


# ----------------------- Inference -----------------------

@torch.inference_mode()
def _run_inference(cfg: TrainConfig):
    """在 JSON / JSONL / 目录数据上做离线推理。

    这里尽量复用训练时的数据处理逻辑：
    - 使用同一个 dataset 类
    - 使用同一套 token 加载 / 回退机制
    - 模型配置优先从 checkpoint 恢复
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 自动解析要加载的 checkpoint
    ckpt_path = cfg.infer_ckpt
    try:
        if ckpt_path is None:
            from glob import glob
            pts = sorted(glob(os.path.join(cfg.out_dir, 'model_epoch*.pt')), key=lambda p: os.path.getmtime(p))
            ckpt_path = pts[-1] if pts else None
    except Exception:
        ckpt_path = None
    if ckpt_path is None or (not os.path.exists(ckpt_path)):
        raise FileNotFoundError(f"No checkpoint found for inference (looked at --infer_ckpt and {cfg.out_dir})")

    # 读取 checkpoint 内保存的训练配置
    obj = torch.load(ckpt_path, map_location=device)
    ck = obj if isinstance(obj, dict) else {}
    ck_cfg = ck.get('config', {})

    # 优先使用 checkpoint 中记录的配置重建模型，
    # 避免当前 CLI 默认值和训练时设置不一致。
    n_waypoints = int(ck_cfg.get('n_waypoints', cfg.n_waypoints))
    use_angle_tvi = bool(ck_cfg.get('use_angle_tvi', cfg.use_angle_tvi))
    no_tanh_actions = bool(ck_cfg.get('no_tanh_actions', cfg.no_tanh_actions))
    vision_feat_dim = int(ck_cfg.get('vision_feat_dim', cfg.vision_feat_dim))
    alpha_xy        = ck_cfg.get('alpha_xy', getattr(cfg, 'alpha_xy', None))
    llm_name        = str(ck_cfg.get('llm_name', getattr(cfg, 'llm_name', "Qwen/Qwen3-8B")))
    print (llm_name)

    model = OpenTrackVLA(
        ModelConfig(
            llm_name=llm_name,
            n_waypoints=n_waypoints,
            beta_nav=float(ck_cfg.get('beta_nav', cfg.beta_nav)),
            use_angle_tvi=use_angle_tvi,
            use_tanh_actions=(not no_tanh_actions),
            alpha_xy=alpha_xy,
        ),
        vision_feat_dim=vision_feat_dim,
    ).to(device).eval()
    msd = ck.get('model_state', None)
    if msd:
        model.load_state_dict(msd, strict=False)

    # 推理数据
    if cfg.infer_json is None:
        raise ValueError('--infer_json is required for inference')
    vds = JsonTrackingDataset(DataConfig(train_json=cfg.infer_json, n_waypoints=n_waypoints, history=cfg.history, cache_root=cfg.cache_root))
    vdl = DataLoader(vds, batch_size=cfg.batch_size, shuffle=False, num_workers=min(2, cfg.num_workers), pin_memory=True, collate_fn=collate_batch)

    # 输出目录
    os.makedirs(cfg.infer_out, exist_ok=True)
    vis_dir = os.path.join(cfg.infer_out, 'vis')
    npz_dir = os.path.join(cfg.infer_out, 'npz')
    if cfg.infer_vis:
        os.makedirs(vis_dir, exist_ok=True)
    if cfg.infer_save_npz:
        os.makedirs(npz_dir, exist_ok=True)

    batches_limit = max(0, int(cfg.infer_batches))
    bdone = 0
    for bidx, batch in enumerate(vdl):
        coarse_tokens = batch['coarse_tokens'].to(device)
        coarse_tidx   = batch['coarse_tidx'].to(device)
        fine_tokens   = batch['fine_tokens'].to(device)
        fine_tidx     = batch['fine_tidx'].to(device)
        yaw_hist      = batch['yaw_hist'].to(device)
        yaw_curr      = batch['yaw_curr'].to(device)
        instr         = batch['instruction']

        pred = model(
            coarse_tokens, coarse_tidx,
            fine_tokens, fine_tidx,
            instr,
            yaw_hist=yaw_hist if use_angle_tvi else None,
            yaw_curr=yaw_curr if use_angle_tvi else None
        )  # 这里输出已经是绝对任务空间量，alpha 已在模型内部作用

        # 保存 NPZ；每个样本单独存，方便后处理检查
        if cfg.infer_save_npz:
            try:
                with torch.no_grad():
                    pred_np = pred.detach().float().cpu().numpy()
                Bcur = pred_np.shape[0]
                for bi in range(Bcur):
                    fpath = os.path.join(npz_dir, f"b{bidx:06d}_i{bi:03d}.npz")
                    np.savez_compressed(
                        fpath,
                        pred=pred_np[bi],
                        instruction=instr[bi],
                        current_path=(batch.get('current_path', [''])[bi] if isinstance(batch.get('current_path', []), list) else '')
                    )
            except Exception:
                pass

        # 在绝对任务空间下可视化，和训练阶段可视化保持一致
        if cfg.infer_vis:
            try:
                with torch.no_grad():
                    # 再次确保可视化使用的是绝对量
                    pred_draw = pred.detach().float()
                    try:
                        model_inspect = model
                        alpha_vec = getattr(model_inspect, 'alpha_task', None)
                        if alpha_vec is not None and pred_draw.size(-1) >= 2 and alpha_vec.size(-1) >= 2:
                            max_xy = pred_draw[..., 0:2].abs().max().item()
                            if max_xy <= 1.5:
                                ax = alpha_vec[..., 0:2].clamp_min(1e-6).to(pred_draw.device, pred_draw.dtype)
                                pred_draw = pred_draw.clone()
                                pred_draw[..., 0:2] = pred_draw[..., 0:2] * ax
                    except Exception:
                        pass
                    pred_np = pred_draw.cpu().numpy()
                cur_paths = batch.get('current_path', [])
                Bcur = pred_np.shape[0]
                for bi in range(min(Bcur, 4)):
                    cur_path = cur_paths[bi] if isinstance(cur_paths, list) and bi < len(cur_paths) else None
                    if cur_path is None or (not os.path.exists(cur_path)):
                        continue
                    pil_img = Image.open(cur_path).convert('RGB')
                    draw = ImageDraw.Draw(pil_img)
                    w, h = pil_img.size
                    base_x = w // 2
                    base_y = int(h * 0.86)
                    def to_pxxy(traj):
                        pts = []
                        for i in range(min(traj.shape[0], 64)):
                            x, y = float(traj[i, 0]), float(traj[i, 1])
                            px = base_x - int(y * 120)
                            py = base_y - int(x * 120)
                            pts.append((px, py))
                        return pts
                    pts_pred = to_pxxy(pred_np[bi])
                    for i2 in range(1, len(pts_pred)):
                        draw.line([pts_pred[i2-1], pts_pred[i2]], fill=(0, 255, 200), width=6)
                    if pts_pred:
                        r0 = 6
                        sx, sy = pts_pred[0]
                        draw.ellipse([sx-r0, sy-r0, sx+r0, sy+r0], fill=(0,255,0))
                    out_path = os.path.join(vis_dir, f"b{bidx:06d}_i{bi:03d}.jpg")
                    pil_img.save(out_path)
            except Exception:
                pass

        bdone += 1
        if batches_limit and bdone >= batches_limit:
            break

    print(f"[INFER] Done. Outputs under {cfg.infer_out}")


# ----------------------- CLI -----------------------

def parse_args() -> TrainConfig:
    # 一个 CLI 入口同时覆盖：
    # - 训练
    # - 周期性验证
    # - 单 episode 评估
    # - 纯离线推理
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_json', type=str, required=True)
    ap.add_argument('--out_dir', type=str, default='./ckpts')
    ap.add_argument('--n_waypoints', type=int, default=8)
    ap.add_argument('--history', type=int, default=31)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--mixed_precision', action='store_true')
    ap.add_argument('--vision_feat_dim', type=int, default=1536)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--cache_root', type=str, default=None, help='视觉 token 缓存根目录；默认使用 <train_json> 所在数据根目录下的 vision_cache')
    ap.add_argument('--distributed', action='store_true')
    ap.add_argument('--dist_backend', type=str, default='nccl')
    ap.add_argument('--use_angle_tvi', action='store_true')  # 默认关闭
    ap.add_argument('--alpha_xy', type=float, default=2.0, help='仅对 XY 目标做缩放的系数；yaw 不缩放')
    ap.add_argument('--beta_nav',      type=float, default=10.0)
    # 日志 / 保存
    ap.add_argument('--log_every', type=int, default=10)
    ap.add_argument('--csv_logging', action='store_true')
    ap.add_argument('--save_trajectories', action='store_true')
    ap.add_argument('--traj_subdir', type=str, default='trajectories')
    # 验证
    ap.add_argument('--eval_every', type=int, default=0, help='每隔多少个 step 做一次验证；0 表示关闭')
    ap.add_argument('--eval_batches', type=int, default=8, help='每次验证最多跑多少个 batch')
    ap.add_argument('--final_wp_threshold', type=float, default=0.2, help='最后一个 waypoint 的命中阈值（按 XY 距离）')
    # 模型选项
    ap.add_argument('--no_tanh_actions', action=argparse.BooleanOptionalAction, default=True, help='是否去掉动作头输出端的 tanh 限幅；去掉后输出可无界')
    # checkpoint 保留策略
    ap.add_argument('--max_ckpts', type=int, default=3, help='最多保留多少个 checkpoint')
    # 断点恢复
    ap.add_argument('--resume', action='store_true', help='从 out_dir 下最新的 checkpoint（或 --resume_ckpt 指定的 checkpoint）恢复训练')
    ap.add_argument('--resume_ckpt', type=str, default=None, help='显式指定要恢复的 checkpoint 路径')
    # 推理
    ap.add_argument('--infer_json', type=str, default=None, help='在该数据集上运行离线推理，支持 json/jsonl/目录')
    ap.add_argument('--infer_ckpt', type=str, default=None, help='推理时加载的 checkpoint；默认取 out_dir 下最新的模型')
    ap.add_argument('--infer_out', type=str, default='./infer_out', help='推理结果输出目录')
    ap.add_argument('--infer_batches', type=int, default=0, help='推理时最多运行多少个 batch；0 表示全跑')
    ap.add_argument('--infer_vis', action='store_true', help='推理时保存可视化图像')
    ap.add_argument('--infer_save_npz', action='store_true', help='推理时保存 npz 预测结果')
    # 单 episode 评估
    ap.add_argument('--episode_json', type=str, default=None, help='用于单个 episode 评估的 JSON/JSONL 路径')
    ap.add_argument('--episode_eval_every', type=int, default=0, help='每隔多少个 step 做一次单 episode 评估；0 表示关闭')
    ap.add_argument('--episode_threshold', type=float, default=0.2, help='单 episode 跟随成功阈值半径')
    ap.add_argument('--episode_max_frames', type=int, default=256, help='单 episode 评估最多使用多少帧')
    ap.add_argument('--llm_name', type=str, default='Qwen/Qwen3-0.6B', help='LLM 主干使用的 HuggingFace 模型名')

    args = ap.parse_args()
    return TrainConfig(**vars(args))


if __name__ == '__main__':
    cfg = parse_args()
    # 纯推理模式：如果给了 infer_json 且 epochs==0，则不进入训练，直接加载模型做推理
    if cfg.infer_json and cfg.epochs == 0:
        _run_inference(cfg)
    else:
        train(cfg)
