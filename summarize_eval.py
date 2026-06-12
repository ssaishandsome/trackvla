#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 EVT benchmark 评测结果。

读取 `sim_data/eval/<task>` 下每个 scene/episode 的结果 json，
计算整体与按场景的聚合指标。

单 episode 结果默认来自 `trained_agent.py` / `baseline_agent.py`，字段形如：
    {
      "finish": false,
      "status": "Collision",
      "success": 0.0,
      "following_rate": 0.69,
      "following_step": 52,
      "total_step": 75,
      "collision": 1.0,
      "instruction": "..."
    }

python /home/ssa/code/EVT/OpenTrackVLA/OpenTrackVLA/summarize_eval.py \
  --eval_dir /home/ssa/code/EVT/OpenTrackVLA/OpenTrackVLA/sim_data/eval/dt
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class EpisodeResult:
    scene: str
    episode_id: str
    finish: float
    success: float
    following_rate: float
    following_step: float
    total_step: float
    collision: float
    status: str
    instruction: str


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if isinstance(x, bool):
            return float(int(x))
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_mean(vals: Iterable[float]) -> float:
    vals = list(vals)
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _iter_result_files(eval_dir: Path) -> Iterable[Path]:
    for fp in sorted(eval_dir.rglob("*.json")):
        if fp.name.endswith("_info.json"):
            continue
        yield fp


def load_results(eval_dir: Path) -> List[EpisodeResult]:
    results: List[EpisodeResult] = []
    for fp in _iter_result_files(eval_dir):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as exc:
            print(f"[warn] skip invalid json: {fp} ({exc})")
            continue

        scene = fp.parent.name
        episode_id = fp.stem
        results.append(
            EpisodeResult(
                scene=scene,
                episode_id=episode_id,
                finish=_to_float(obj.get("finish", 0.0)),
                success=_to_float(obj.get("success", 0.0)),
                following_rate=_to_float(obj.get("following_rate", 0.0)),
                following_step=_to_float(obj.get("following_step", 0.0)),
                total_step=_to_float(obj.get("total_step", 0.0)),
                collision=_to_float(obj.get("collision", 0.0)),
                status=str(obj.get("status", "")),
                instruction=str(obj.get("instruction", "")),
            )
        )
    return results


def summarize_group(results: List[EpisodeResult]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {
            "episodes": 0,
            "SR": float("nan"),
            "TR": float("nan"),
            "CR": float("nan"),
            "FR": float("nan"),
            "avg_following_step": float("nan"),
            "avg_total_step": float("nan"),
        }
    return {
        "episodes": n,
        "SR": _safe_mean(r.success for r in results),
        "TR": _safe_mean(r.following_rate for r in results),
        "CR": _safe_mean(r.collision for r in results),
        "FR": _safe_mean(r.finish for r in results),
        "avg_following_step": _safe_mean(r.following_step for r in results),
        "avg_total_step": _safe_mean(r.total_step for r in results),
    }


def format_metric(x: Any) -> str:
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.4f}"
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser(description="汇总 sim_data/eval 下的 episode 评测结果")
    ap.add_argument("--eval_dir", type=str, required=True, help="例如 sim_data/eval/dt 或 sim_data/eval/stt")
    ap.add_argument("--topk_scenes", type=int, default=10, help="打印前多少个 scene 的统计")
    ap.add_argument("--save_json", type=str, default=None, help="可选：把汇总结果保存成 json")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        raise FileNotFoundError(f"eval_dir not found: {eval_dir}")

    results = load_results(eval_dir)
    if len(results) == 0:
        print(f"[summary] no episode result json found under {eval_dir}")
        return

    overall = summarize_group(results)

    by_scene: Dict[str, List[EpisodeResult]] = {}
    for r in results:
        by_scene.setdefault(r.scene, []).append(r)

    scene_rows = []
    for scene, items in by_scene.items():
        row = {"scene": scene, **summarize_group(items)}
        scene_rows.append(row)
    scene_rows.sort(key=lambda x: (-int(x["episodes"]), x["scene"]))

    status_counts: Dict[str, int] = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    print(f"[summary] eval_dir = {eval_dir}")
    print(f"[summary] episodes = {overall['episodes']}")
    print(
        "[summary] "
        f"SR={format_metric(overall['SR'])}  "
        f"TR={format_metric(overall['TR'])}  "
        f"CR={format_metric(overall['CR'])}  "
        f"FR={format_metric(overall['FR'])}  "
        f"avg_following_step={format_metric(overall['avg_following_step'])}  "
        f"avg_total_step={format_metric(overall['avg_total_step'])}"
    )
    print(f"[summary] unique_scenes = {len(by_scene)}")
    print(f"[summary] status_counts = {status_counts}")

    topk = max(0, int(args.topk_scenes))
    if topk > 0:
        print("[summary] top scene stats:")
        for row in scene_rows[:topk]:
            print(
                "  "
                f"{row['scene']}: "
                f"episodes={row['episodes']} "
                f"SR={format_metric(row['SR'])} "
                f"TR={format_metric(row['TR'])} "
                f"CR={format_metric(row['CR'])} "
                f"FR={format_metric(row['FR'])}"
            )

    if args.save_json:
        out = {
            "eval_dir": str(eval_dir),
            "overall": overall,
            "status_counts": status_counts,
            "by_scene": scene_rows,
        }
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[summary] saved json -> {out_path}")


if __name__ == "__main__":
    main()
