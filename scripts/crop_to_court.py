#!/usr/bin/env python3
"""
コート中心クロップ実験 — 入力 framing を詰めると検出がどう変わるかを安く試す。

現状の課題: 3840×800 のウルトラワイドを 640 に letterbox すると scale≈0.167 で
奥側選手が ~10px に潰れて検出不能。コート領域は横 ~1760px・縦 ~370px しか
占めておらず、水平方向に大きな無駄がある。

このスクリプトは court.json の4端点からコート外接矩形＋マージンを計算し、
その領域だけにクロップした mp4 を書き出す。あわせて端点をクロップ原点ぶん
オフセットした court.json を出力するので、#1/#2 の評価をクロップ動画で
そのまま再実行できる。

⚠ できること/できないこと:
  - ✅ 水平の無駄を削る効果（letterbox scale 改善）を再現
  - ❌ 垂直解像度の増加は再現できない（元映像に 800px 分の縦情報しかない）
  実効解像度 = 選手の実px × (640 / 長辺)。クロップは長辺を縮めて scale を
  上げるが、選手の実pxは増えない。アップスケールは補間で情報ゼロなので行わない。
  → これは「改善の下限」を測る実験（新カメラ実機はさらに上）。

Usage:
  python scripts/crop_to_court.py \\
    --source videos/volley-2.mp4 --court configs/court.json \\
    --out-video videos/volley-2_courtcrop.mp4 \\
    --out-court configs/court_crop.json \\
    --margin-top 100 --margin-bottom 40 --margin-side 80
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from alt_pix.stream import iter_frames


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crop video to the court ROI")
    p.add_argument("--source", required=True)
    p.add_argument("--court", required=True, help="court.json (4 corners)")
    p.add_argument("--out-video", required=True)
    p.add_argument("--out-court", required=True,
                   help="remapped court.json for the cropped video")
    # マージン（px）: 奥側選手は遠ライン上に立ち頭が上に伸びるので top を厚めに、
    # 手前選手はジャンプ/頭上余白で側方より上が必要。
    p.add_argument("--margin-top", type=float, default=100.0)
    p.add_argument("--margin-bottom", type=float, default=40.0)
    p.add_argument("--margin-side", type=float, default=80.0)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data = json.loads(Path(args.court).read_text())
    corners = np.array(data["corners"], dtype=np.float64)  # (4,2)

    # ── クロップ矩形（コート外接 + マージン）────────────────────────────────────
    x_min, y_min = corners.min(axis=0)
    x_max, y_max = corners.max(axis=0)
    x0 = x_min - args.margin_side
    x1 = x_max + args.margin_side
    y0 = y_min - args.margin_top
    y1 = y_max + args.margin_bottom

    # 整数化（フレーム境界クリップは最初のフレームサイズ確定後）
    crop = [x0, y0, x1, y1]

    writer: cv2.VideoWriter | None = None
    cw = ch = 0
    n = 0

    for frame_id, ts, frame in iter_frames(args.source):
        h, w = frame.shape[:2]
        if writer is None:
            cx0 = max(0, int(round(crop[0])))
            cy0 = max(0, int(round(crop[1])))
            cx1 = min(w, int(round(crop[2])))
            cy1 = min(h, int(round(crop[3])))
            cw, ch = cx1 - cx0, cy1 - cy0
            crop = [cx0, cy0, cx1, cy1]

            # 端点をクロップ原点ぶんオフセットして保存
            new_corners = [[float(cxy[0] - cx0), float(cxy[1] - cy0)] for cxy in corners]
            out_court = dict(data)
            out_court["corners"] = new_corners
            Path(args.out_court).write_text(json.dumps(out_court, indent=2))

            # 旧 scale との比較を表示
            old_scale = 640.0 / max(w, h)
            new_scale = 640.0 / max(cw, ch)
            court_w = x_max - x_min
            court_h = y_max - y_min
            print("── クロップ実験 セットアップ ──")
            print(f"  元フレーム      : {w}x{h}  (letterbox scale {old_scale:.3f})")
            print(f"  クロップ領域    : {cw}x{ch}  @ ({cx0},{cy0})  "
                  f"(letterbox scale {new_scale:.3f})")
            print(f"  コート内寸(元)  : {court_w:.0f}x{court_h:.0f}  "
                  f"アスペクト {court_w/max(court_h,1):.2f}:1")
            print(f"  クロップ アスペクト: {cw/max(ch,1):.2f}:1")
            print(f"  実効解像度向上  : ×{new_scale/old_scale:.2f}")
            print(f"  → 奥側選手 60px 相当: {60*old_scale:.0f}px → {60*new_scale:.0f}px @640入力")
            print(f"  出力動画        : {args.out_video}")
            print(f"  出力court       : {args.out_court}")
            writer = cv2.VideoWriter(args.out_video,
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (cw, ch))

        cropped = frame[crop[1]:crop[3], crop[0]:crop[2]]
        writer.write(cropped)
        n += 1
        if n % 100 == 0:
            print(f"  wrote {n} frames")
        if args.max_frames and n >= args.max_frames:
            break

    if writer:
        writer.release()
    print(f"\n完了: {n} frames → {args.out_video}")
    print("次の評価:")
    print(f"  python scripts/eval_court_homography.py --source {args.out_video} "
          f"--court {args.out_court} --person-model models/yolox_m.onnx "
          f"--out videos/courtcrop_homography.mp4 --max-frames 300")
    print(f"  python scripts/debug_player_tracking.py --source {args.out_video} "
          f"--court {args.out_court} --person-model models/yolox_m.onnx "
          f"--out videos/courtcrop_tracking.mp4 --max-frames 600")


if __name__ == "__main__":
    main()
