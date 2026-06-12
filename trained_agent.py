import habitat
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from habitat.config.default_structured_configs import AgentConfig
from habitat.tasks.nav.nav import NavigationEpisode
from habitat_sim.gfx import LightInfo, LightPositionModel
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from tqdm import trange

import os
import os.path as osp
import imageio
import json
import traceback
import torch
from typing import Optional, List
from collections import deque


try:
    from model import OpenTrackVLA as PlannerModel, ModelConfig as PlannerConfig
    from cache_gridpool import VisionFeatureCacher, VisionCacheConfig, grid_pool_tokens
except Exception:
    PlannerModel = None  # type: ignore
    PlannerConfig = None  # type: ignore
    VisionFeatureCacher = None  # type: ignore
    VisionCacheConfig = None  # type: ignore
    grid_pool_tokens = None  # type: ignore

try:
    from open_trackvla_hf import OpenTrackVLAForWaypoint
except Exception:
    OpenTrackVLAForWaypoint = None  # type: ignore

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None  # type: ignore


def evaluate_agent(config, dataset_split, save_path) -> None:
    # robot definition
    robot_config = GTBBoxAgent(save_path)
    with habitat.TrackEnv(
        config=config,
        dataset=dataset_split
    ) as env:
        sim = env.sim
        robot_config.reset()
        
        num_episodes = len(env.episodes)
        for _ in trange(num_episodes):
            obs = env.reset()
            light_setup = [
                LightInfo(
                    vector=[10.0, -2.0, 0.0, 0.0],
                    color=[1.0, 1.0, 1.0],
                    model=LightPositionModel.Global,
                ),
                LightInfo(
                    vector=[-10.0, -2.0, 0.0, 0.0],
                    color=[1.0, 1.0, 1.0],
                    model=LightPositionModel.Global,
                ),
                LightInfo(
                    vector=[0.0, -2.0, 10.0, 0.0],
                    color=[1.0, 1.0, 1.0],
                    model=LightPositionModel.Global,
                ),
                LightInfo(
                    vector=[0.0, -2.0, -10.0, 0.0],
                    color=[1.0, 1.0, 1.0],
                    model=LightPositionModel.Global,
                ),
            ]
            sim.set_light_setup(light_setup)

            result = {}
            record_infos = []

            # Fetch instruction text for this episode if available
            try:
                instruction = env.current_episode.info.get('instruction', None)
            except Exception:
                instruction = None

            action_dict = dict()
            finished = False
            
            humanoid_agent_main = sim.agents_mgr[0].articulated_agent
            robot_agent = sim.agents_mgr[1].articulated_agent

            iter_step = 0
            followed_step = 0
            human_no_move = 0
            too_far_count = 0
            status = 'Normal'
            info = env.get_metrics()

            while not env.episode_over:
                record_info = {}
                
                obs = sim.get_sensor_observations()

                detector = env.task._get_observations(env.current_episode)
                action = robot_config.act(obs, detector, env.current_episode.episode_id, instruction)

                action_dict = {
                    "action": ("agent_0_humanoid_navigate_action", "agent_1_base_velocity", "agent_2_oracle_nav_randcoord_action_obstacle", "agent_3_oracle_nav_randcoord_action_obstacle", "agent_4_oracle_nav_randcoord_action_obstacle", "agent_5_oracle_nav_randcoord_action_obstacle"),
                    "action_args": {
                        "agent_1_base_vel" : action
                    }
                }
                
                iter_step += 1
                env.step(action_dict)

                info = env.get_metrics()
                if info['human_following'] == 1.0:
                    print("Followed")
                    followed_step += 1
                    too_far_count = 0
                else:
                    print("Lost")

                if np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos) > 4.0:
                    too_far_count += 1
                    if too_far_count > 20:
                        print("Too far from human!")
                        status = 'Lost'
                        finished = False
                        break

                record_info["step"] = iter_step
                record_info["dis_to_human"] = float(np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos))
                record_info["facing"] = info['human_following']
                record_info["base_velocity"] = action
                record_infos.append(record_info)

                if info['human_collision'] == 1.0:
                    print("Collision detected!")
                    status = 'Collision'
                    finished = False
                    break
                
                print(f"========== ID: {env.current_episode.episode_id} Step now is: {iter_step} action is: {action} dis_to_main_human: {np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos)} ============")

            print("finished episode id: ", env.current_episode.episode_id)
            info = env.get_metrics()
            robot_config.reset(env.current_episode)

            if env.episode_over:
                finished = True
            
            scene_key = osp.splitext(osp.basename(env.current_episode.scene_id))[0].split('.')[0]
            save_dir = os.path.join(save_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "{}_info.json".format(env.current_episode.episode_id)), "w") as f:
                json.dump(record_infos, f, indent=2) 
            result['finish'] = finished
            result['status'] = status
            if iter_step < 300:
                result['success'] = info['human_following_success'] and info['human_following']
            else:
                result['success'] = info['human_following']
            result['following_rate'] = followed_step / iter_step
            result['following_step'] = followed_step
            result['total_step'] = iter_step
            result['collision'] = info['human_collision']
            if instruction is not None:
                result['instruction'] = instruction
            with open(os.path.join(save_dir, "{}.json".format(env.current_episode.episode_id)), "w") as f:
                json.dump(result, f, indent=2) 


class GTBBoxAgent(AgentConfig):
    def __init__(self, result_path):
        super().__init__()
        print("Initialize gtbbox agent")

        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)
        
        self.rgb_list = []
        self.rgb_box_list = []

        # Frame buffer for TrackVLA model (stores normalized CHW tensors)
        self.frame_buffer: deque = deque(maxlen=32)
        self.norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.window_T = 8
        self.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trackvla_model = None

        # Planner (train_planner.py) integration
        self.history = 31  # number of previous frames (coarse history)
        self._vision_cache = None
        self.planner_model = None
        self.planner_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Maintain history of coarse tokens (per-frame 4 tokens)
        self._coarse_hist_tokens: deque = deque(maxlen=self.history)
        self._planner_ckpt_path: Optional[str] = None
        self._hf_model_dir: Optional[str] = None
        self._hf_model_id: Optional[str] = None
        self._resolve_planner_ckpt_once()
        self._init_planner_model()
        self._last_predicted_traj = None  # type: Optional[np.ndarray]

        self.kp_t = 2
        self.kd_t = 0
        self.kp_f = 1
        self.kd_f = 0
        # Separate lateral (y-axis) gains so y_speed != yaw_speed
        self.kp_y = 0.5
        self.kd_y = 0

        self.prev_error_t = 0
        self.prev_error_f = 0

        self.first_inside = True

        self.reset()

    def reset(self, episode: NavigationEpisode = None):
        if len(self.rgb_list) != 0:
            scene_key = osp.splitext(osp.basename(episode.scene_id))[0].split('.')[0]
            save_dir = os.path.join(self.result_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            output_video_path = os.path.join(save_dir, "{}.mp4".format(episode.episode_id))
            imageio.mimsave(output_video_path, self.rgb_list)

            print(f"Successfully save the episode video with episode id {episode.episode_id}")

            self.rgb_list = []
        
        self.first_inside = True

    def act(self, observations, detector, episode_id, instruction: Optional[str] = None):
        self.episode_id = episode_id
        
        rgb = observations["agent_1_articulated_agent_jaw_rgb"]
        print (rgb.shape)
        rgb_ = rgb[:, :, :3]
        image = np.asarray(rgb_[:, :, ::-1])
        height, width = image.shape[:2]
        # Push current frame into buffer (normalize to CHW, [0,1] -> norm)
        try:
            chw = np.transpose(rgb_[:, :, ::-1].astype(np.float32) / 255.0, (2, 0, 1))
            chw = (chw - self.norm_mean[:, None, None]) / self.norm_std[:, None, None]
            self.frame_buffer.append(torch.from_numpy(chw))
        except Exception:
            pass
        
        
        # Prefer planner action (train_planner) if available, then TrackVLA, else PID
        planner_action = self._planner_action(rgb_, instruction)
        action = planner_action
        if action is None:
            # 避免把 None 传给 Habitat 动作接口导致直接崩溃。
            # 这里先保守回退为静止动作，同时把真正错误交给 _planner_action 内部日志打印。
            action = [0.0, 0.0, 0.0]
            print("[planner] planner_action is None, fallback to zero velocity action.")
        
        print (f"Planner action: {action}")

        self.last_action = action
        # Draw predicted trajectory overlay (if available)
        frame_out = self._render_frame_with_traj(rgb_, self._last_predicted_traj)
        self.rgb_list.append(frame_out)

        return action

    def _ensure_vision_cache(self):
        if VisionFeatureCacher is None or VisionCacheConfig is None:
            print(f"[planner] Vision cache is not available")
            return None
        if self._vision_cache is None:
            try:
                cfg = VisionCacheConfig(image_size=384, batch_size=1, device=('cuda' if torch.cuda.is_available() else 'cpu'))
                self._vision_cache = VisionFeatureCacher(cfg)
                self._vision_cache.eval()
            except Exception as e:
                print(f"[planner] Failed to initialize vision cache: {e}")
                self._vision_cache = None
        return self._vision_cache

    def _is_hf_dir(self, path: str) -> bool:
        cfg = os.path.join(path, 'config.json')
        weights = os.path.join(path, 'pytorch_model.bin')
        return os.path.isdir(path) and os.path.isfile(cfg) and os.path.isfile(weights)

    def _resolve_planner_ckpt_once(self):
        if self._planner_ckpt_path is not None or self._hf_model_dir is not None:
            return
        hf_env = os.environ.get('HF_MODEL_DIR')
        if hf_env and self._is_hf_dir(hf_env):
            self._hf_model_dir = hf_env
            return
        hf_id_env = os.environ.get('HF_MODEL_ID')
        if hf_id_env:
            local = self._download_hf_model(hf_id_env)
            if local:
                self._hf_model_dir = local
                self._hf_model_id = hf_id_env
                return
        ckpt_path = None
        try:
            ckpt_dir = os.path.join(os.path.dirname(__file__), 'ckpts')
            env_ckpt = os.environ.get('CKPT')
            if env_ckpt and os.path.isfile(env_ckpt):
                ckpt_path = env_ckpt
            elif os.path.isdir(ckpt_dir):
                cands = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.startswith('model_epoch') and f.endswith('.pt')]
                if cands:
                    ckpt_path = sorted(cands, key=lambda p: os.path.getmtime(p))[-1]
        except Exception:
            ckpt_path = None
        if ckpt_path:
            self._planner_ckpt_path = ckpt_path
            return
        default_hf = os.path.join(os.path.dirname(__file__), 'open_trackvla_hf')
        if self._is_hf_dir(default_hf):
            self._hf_model_dir = default_hf

    def _download_hf_model(self, repo_id: str) -> Optional[str]:
        if snapshot_download is None:
            print("[planner] huggingface_hub is not installed; cannot download HF_MODEL_ID.")
            return None
        cache_root = os.path.join(os.path.dirname(__file__), 'hf_downloads')
        os.makedirs(cache_root, exist_ok=True)
        safe_name = repo_id.replace('/', '__')
        local_dir = os.path.join(cache_root, safe_name)
        if self._is_hf_dir(local_dir):
            return local_dir
        try:
            snapshot_download(
                repo_id,
                repo_type='model',
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
            if self._is_hf_dir(local_dir):
                return local_dir
        except Exception as exc:
            print(f"[planner] Failed to download HF repo {repo_id}: {exc}")
        return None

    def _init_planner_model(self):
        if PlannerModel is None or PlannerConfig is None:
            return
        if self.planner_model is not None:
            return
        # Vision feature dim used in train_planner by default
        vision_feat_dim = 1536
        if self._hf_model_dir and OpenTrackVLAForWaypoint is not None:
            try:
                print(f"[planner] Loading HuggingFace checkpoint: {self._hf_model_dir}")
                model = OpenTrackVLAForWaypoint.from_pretrained(self._hf_model_dir)
                model = model.to(self.planner_device).eval()
                self.planner_model = model
                return
            except Exception as exc:
                print(f"[planner] Failed to load HF checkpoint ({exc}), falling back to legacy ckpt.")
        if PlannerModel is None or PlannerConfig is None:
            return
        cfg = PlannerConfig()
        cfg.n_waypoints = 8
        model = PlannerModel(cfg, vision_feat_dim=vision_feat_dim).to(self.planner_device).eval()
        print (self._planner_ckpt_path)
        if self._planner_ckpt_path:
            obj = torch.load(self._planner_ckpt_path, map_location=self.planner_device)
            msd = obj.get('model_state') or obj.get('model_state_dict')
            if msd:
                model.load_state_dict(msd, strict=False)
                print (model)
                print ("Planner model loaded successfully")
        self.planner_model = model


    def _encode_frame_tokens(self, rgb_np: np.ndarray):
        """Encode an RGB frame (H,W,3, uint8-like) into (Vcoarse(4,C), Vfine(64,C))."""
        if grid_pool_tokens is None:
            return None, None
        enc = self._ensure_vision_cache()
        if enc is None:
            return None, None
        try:
            from PIL import Image
            pil = Image.fromarray(rgb_np.astype(np.uint8))
            tok_dino, Hp, Wp = enc._encode_dino([pil])
            tok_sigl = enc._encode_siglip([pil], out_hw=(Hp, Wp))
            Vt_cat = torch.cat([tok_dino, tok_sigl], dim=-1)  # (1, P, C_total)
            Vfine = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=64)[0].float()   # (64, C)
            Vcoarse = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=4)[0].float()  # (4, C)
            return Vcoarse, Vfine
        except Exception:
            return None, None

    def _planner_action(self, rgb_frame_np: np.ndarray, instruction: Optional[str]) -> Optional[List[float]]:
        """Use NavFoM planner to predict waypoints and convert to [vx, vy, wz]."""
        if self.planner_model is None and PlannerModel is None:
            return None
        # Encode current frame tokens
        Vc, Vf = self._encode_frame_tokens(rgb_frame_np)
        if Vc is None or Vf is None:
            self._last_predicted_traj = None
            print(f"[planner] Frame encoding failed")
            return None
        # Require planner model to be initialized once
        if self.planner_model is None:
            self._last_predicted_traj = None
            print(f"[planner] Planner model is not initialized")
            return None
        try:
            # Append current coarse tokens to history
            self._coarse_hist_tokens.append(Vc.cpu())
            # Build coarse history up to H frames with left padding using earliest token
            H = self.history
            hist = list(self._coarse_hist_tokens)
            if len(hist) < H:
                pad_needed = H - len(hist)
                first = hist[0] if hist else Vc
                hist = [first] * pad_needed + hist
            else:
                hist = hist[-H:]
            coarse_list = []
            coarse_tidx = []
            for t, tok4 in enumerate(hist):
                tok4 = tok4.to(self.planner_device)
                coarse_list.append(tok4)
                coarse_tidx.append(torch.full((tok4.size(0),), fill_value=t, dtype=torch.long, device=self.planner_device))
            coarse_tokens = torch.cat(coarse_list, dim=0).unsqueeze(0)  # (1, H*4, C)
            coarse_tidx = torch.cat(coarse_tidx, dim=0).unsqueeze(0)    # (1, H*4)
            # Fine tokens for current frame; time index H
            fine_tokens = Vf.to(self.planner_device).unsqueeze(0)       # (1, 64, C)
            fine_tidx = torch.full((1, fine_tokens.size(1)), fill_value=H, dtype=torch.long, device=self.planner_device)
            instr = [instruction or 'follow the person']
            with torch.inference_mode():
                tau = self.planner_model(
                    coarse_tokens, coarse_tidx,
                    fine_tokens, fine_tidx,
                    instr
                )  # (1, Mw, D)
            # Convert first waypoint to velocities using dt
            tau_cpu = tau.detach().float().cpu().numpy()
            print (tau_cpu)
            self._last_predicted_traj = tau_cpu[0]            
            wp0 = tau[0, 1]
            x, y = float(wp0[0].item()), float(wp0[1].item())
            theta = float(wp0[2].item()) if wp0.numel() >= 3 else 0.0
            dt = 0.1            
            vx = x / dt
            vy = y / dt
            wz = theta / dt            
            print (f"Planner action: {vx}, {vy}, {wz}")
            return [float(vx), float(vy), float(wz)]
            
        except Exception as exc:
            self._last_predicted_traj = None
            print(f"[planner] _planner_action failed: {exc}")
            traceback.print_exc()
            return None

    def _render_frame_with_traj(self, rgb_frame_np: np.ndarray, traj_xyz: Optional[np.ndarray]) -> np.ndarray:
        """Overlay predicted trajectory onto the frame and return the composited RGB frame."""
        try:
            if traj_xyz is None or not isinstance(traj_xyz, np.ndarray) or traj_xyz.size == 0:
                return rgb_frame_np
            from PIL import Image, ImageDraw
            img = Image.fromarray(rgb_frame_np.astype(np.uint8), mode='RGB')
            draw = ImageDraw.Draw(img)
            w, h = img.size
            base_x = w // 2
            base_y = int(h * 0.86)
            scale = 120.0  # px per meter
            pts = []
            npts = min(int(traj_xyz.shape[0]), 64)
            for i in range(npts):
                x = float(traj_xyz[i, 0])
                y = float(traj_xyz[i, 1]) if traj_xyz.shape[1] >= 2 else 0.0
                px = base_x - int(y * scale)
                py = base_y - int(x * scale)
                pts.append((px, py))
            # outline thicker black for visibility
            for i in range(1, len(pts)):
                draw.line([pts[i-1], pts[i]], fill=(0, 0, 0), width=8)
            # main colored line
            for i in range(1, len(pts)):
                draw.line([pts[i-1], pts[i]], fill=(0, 255, 180), width=4)
            # start point marker
            if pts:
                r = 4
                sx, sy = pts[0]
                draw.ellipse([sx-r, sy-r, sx+r, sy+r], fill=(0, 255, 0))
            return np.asarray(img)
        except Exception:
            return rgb_frame_np

    
