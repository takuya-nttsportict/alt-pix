#!/usr/bin/env python3
"""
Phase 4 評価 ① — コートホモグラフィの検証

2つのパネルを並べた動画を出力する：

  [ 左 ] 元映像 + オーバーレイ
    - 4端点と外枠（水色）
    - H⁻¹ でコート平面の 1m / 3m グリッド線を画像に逆投影（白細線 / 白太線）
    - 検出した人物の足元ドット：場内=緑 / 場外=赤（margin 付き）
    - 足元のコート座標 (u,v) [m] をラベル表示（デバッグ用）

  [ 右 ] 鳥瞰ビュー (18m × 9m スケール)
    - コート外枠（白）
    - 3m/6m/9m/12m/15m の縦ライン（センターラインを太く）
    - 3m の攻撃ライン（破線）
    - 検出した人物の足元を対応色のドットでプロット

何を見るか:
  - 左: グリッド線が実際のコートライン/目地/床面マーカーに重なるか
         → ずれが大きければ 4端点の指定ミスか findHomography の符号間違い
  - 右: 選手がコートの正しい位置に落ちるか（ネット際の選手がコート中央付近に）
         → 変なポジションに飛んでいたら変換行列の方向が逆か端点順序の間違い
  - 足元ラベル: 画像端/画面外の点が out-of-range な座標を返していないか

人物検出器がなくても --no-persons で純粋に幾何だけ評価できる。

Usage:
  # 幾何のみ（人物検出器不要）
  python scripts/eval_court_homography.py \\
    --source videos/volley-2.mp4 --court configs/court.json \\
    --out videos/court_eval.mp4 --max-frames 300 --no-persons

  # 人物検出あり（足元のコート座標も確認）
  python scripts/eval_court_homography.py \\
    --source videos/volley-2.mp4 --court configs/court.json \\
    --person-model models/yolox_m.onnx \\
    --out videos/court_eval.mp4 --max-frames 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from alt_pix.court import CourtCalibration, _COURT_W, _COURT_H
from alt_pix.log_config import setup_logging
from alt_pix.stream import iter_frames

# ── バレーコートの規格ライン定義（メートル座標） ────────────────────────────────

# センターライン（ネット下） u = 9.0m
# 攻撃ライン（Attack line） u = 6.0m / 12.0m
# u 方向の定義: TL=0, TR=18, BL=0, BR=18 （court.py の _COURT_W=18）
_VERT_LINES = [
    (3.0,  False, "3m"),
    (6.0,  True,  "ATK"),   # 攻撃ライン
    (9.0,  True,  "NET"),   # センター/ネット
    (12.0, True,  "ATK"),
    (15.0, False, "15m"),
]
_HORIZ_LINES = [3.0, 6.0]   # v 方向（_COURT_H=9 の 1/3 ずつ）


def _court_to_image(H_inv: np.ndarray, u: float, v: float) -> tuple[int, int]:
    """コート座標 (u,v)[m] → 画像ピクセル (x,y)。"""
    pt = np.array([[[u, v]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(pt, H_inv)
    return int(round(dst[0, 0, 0])), int(round(dst[0, 0, 1]))


def _draw_grid(frame: np.ndarray, H_inv: np.ndarray) -> None:
    """H⁻¹ でコート平面のグリッドを画像に描画する。"""
    h, w = frame.shape[:2]

    def line(u0, v0, u1, v1, color, thick):
        p0 = _court_to_image(H_inv, u0, v0)
        p1 = _court_to_image(H_inv, u1, v1)
        # 画面外の点があっても cv2.line はクリッピングしてくれる
        cv2.line(frame, p0, p1, color, thick, cv2.LINE_AA)

    # 水平ライン（v 方向）— コート幅 3 等分
    for v in _HORIZ_LINES:
        line(0, v, _COURT_W, v, (200, 200, 200), 1)

    # 垂直ライン（u 方向）
    for u, is_key, label in _VERT_LINES:
        color = (255, 255, 255) if is_key else (160, 160, 160)
        thick = 2 if is_key else 1
        line(u, 0, u, _COURT_H, color, thick)
        # ラベルを外枠の上に添える
        lx, ly = _court_to_image(H_inv, u, 0)
        cv2.putText(frame, label, (lx - 16, max(ly - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


def _make_birdseye(
    bv_h: int,
    bv_w: int,
    feet: list[tuple[float, float]],
    on_court_flags: list[bool],
) -> np.ndarray:
    """選手の足元をコート俯瞰図にプロットした画像を返す。"""
    img = np.zeros((bv_h, bv_w, 3), dtype=np.uint8)

    margin_px = 30  # 俯瞰図の外枠マージン
    court_px_w = bv_w - 2 * margin_px
    court_px_h = bv_h - 2 * margin_px

    def m2px(u: float, v: float) -> tuple[int, int]:
        """コート座標 (u,v)[m] → 鳥瞰画像ピクセル。"""
        px = margin_px + int(round(u / _COURT_W * court_px_w))
        py = margin_px + int(round(v / _COURT_H * court_px_h))
        return px, py

    # コート外枠
    cv2.rectangle(img, (margin_px, margin_px),
                  (bv_w - margin_px, bv_h - margin_px), (200, 200, 200), 2)

    # グリッドライン
    for v in _HORIZ_LINES:
        p0 = m2px(0, v)
        p1 = m2px(_COURT_W, v)
        cv2.line(img, p0, p1, (100, 100, 100), 1)

    for u, is_key, label in _VERT_LINES:
        p0 = m2px(u, 0)
        p1 = m2px(u, _COURT_H)
        color = (180, 180, 180) if is_key else (80, 80, 80)
        thick = 2 if is_key else 1
        cv2.line(img, p0, p1, color, thick)
        lx, ly = m2px(u, 0)
        cv2.putText(img, label, (lx - 14, max(ly - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)

    # 選手足元ドット
    for (u, v), on in zip(feet, on_court_flags):
        if not (-3 < u < _COURT_W + 3 and -3 < v < _COURT_H + 3):
            continue  # 大きく外れた点は描かない（変換エラー確認のため）
        px = m2px(u, v)
        color = (0, 220, 0) if on else (0, 80, 220)
        cv2.circle(img, px, 6, color, -1)

    # 凡例
    cv2.circle(img, (margin_px + 10, bv_h - margin_px + 14), 5, (0, 220, 0), -1)
    cv2.putText(img, "on-court", (margin_px + 18, bv_h - margin_px + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1)
    cv2.circle(img, (margin_px + 90, bv_h - margin_px + 14), 5, (0, 80, 220), -1)
    cv2.putText(img, "off-court", (margin_px + 98, bv_h - margin_px + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 80, 220), 1)
    return img


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Court homography evaluation")
    p.add_argument("--source", required=True)
    p.add_argument("--court", required=True, help="JSON from pick_court_corners.py")
    p.add_argument("--out", default="court_eval.mp4")
    p.add_argument("--person-model", default="models/yolox_m.onnx")
    p.add_argument("--conf-person", type=float, default=0.4)
    p.add_argument("--no-persons", action="store_true",
                   help="Skip person detection (geometry-only evaluation)")
    p.add_argument("--scene-margin", type=float, default=400.0,
                   help="Margin (px) for on/off court labelling at the feet")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--frame-skip", type=int, default=0,
                   help="Process every (N+1)-th frame")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)

    court = CourtCalibration.load(args.court)
    # H maps image → court(m); H_inv maps court(m) → image
    H_inv = np.linalg.inv(court._H).astype(np.float32)

    person_det = None
    if not args.no_persons:
        ort_device = "cuda" if args.device.startswith("cuda") else "cpu"
        from alt_pix.detector import YOLOXDetector, _COCO_PERSON
        person_det = YOLOXDetector(args.person_model, conf_thr=args.conf_person,
                                   detect_classes={_COCO_PERSON}, device=ort_device)

    writer: cv2.VideoWriter | None = None
    n = 0
    n_on = n_off = 0

    for frame_id, ts, frame in iter_frames(args.source, skip_frames=args.frame_skip):
        h, w = frame.shape[:2]
        vis = frame.copy()

        # ── 元映像オーバーレイ ──────────────────────────────────────────────
        court.draw(vis)
        _draw_grid(vis, H_inv)

        feet: list[tuple[float, float]] = []
        on_flags: list[bool] = []

        if person_det is not None:
            dets = person_det.detect(frame)
            for d in dets:
                x1, y1, x2, y2 = d.bbox
                fx = (x1 + x2) / 2
                fy = y2
                on = court.is_on_court(d.bbox, args.scene_margin)
                on_flags.append(on)
                u, v = court.image_to_court(fx, fy)
                feet.append((u, v))

                # 足元ドット（場内=緑、場外=赤）
                dot_color = (0, 220, 0) if on else (0, 80, 220)
                cv2.circle(vis, (int(fx), int(fy)), 5, dot_color, -1)

                # コート座標ラベル（DEBUG: --log-level DEBUG 時のみ描画しても良いが
                # ここでは常時小さく表示して変換値を目視しやすくする）
                label = f"({u:.1f},{v:.1f})"
                cv2.putText(vis, label, (int(fx) + 7, int(fy) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, dot_color, 1, cv2.LINE_AA)

                n_on += int(on)
                n_off += int(not on)

        # ── 鳥瞰ビュー ──────────────────────────────────────────────────────
        # 鳥瞰図の高さは元フレームに合わせる、幅は 18:9 比から決める
        bv_h = h
        bv_w = int(round(h * _COURT_W / _COURT_H))   # 18:9 アスペクト
        bv = _make_birdseye(bv_h, bv_w, feet, on_flags)

        # ── 情報テキスト ─────────────────────────────────────────────────────
        cv2.putText(vis, f"f={frame_id}  on={n_on}  off={n_off}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(bv, "Bird's-eye (18m x 9m)",
                    (10, bv_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # ── 合成して出力 ─────────────────────────────────────────────────────
        combined = np.hstack([vis, bv])
        if writer is None:
            ch, cw = combined.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     25.0, (cw, ch))
        writer.write(combined)

        n += 1
        if n % 100 == 0:
            print(f"[f={frame_id:5d}] on={n_on}  off={n_off}")
        if args.max_frames and n >= args.max_frames:
            break

    if writer:
        writer.release()

    print(f"\n── コートホモグラフィ評価まとめ ──")
    print(f"  出力動画  : {args.out}")
    print(f"  処理フレーム: {n}")
    print(f"  累計 on-court 足元: {n_on}")
    print(f"  累計 off-court 足元: {n_off}")
    print()
    print("  チェックポイント:")
    print("  [左パネル] グリッドの縦線がコートのラインに重なるか？")
    print("             ずれる → 4端点の指定位置を見直す（TL/TR/BR/BL の順序確認）")
    print("  [左パネル] 足元(u,v) が 0–18 / 0–9 の範囲内に収まるか？")
    print("             大きくはみ出す → H の方向か端点順序が逆")
    print("  [右パネル] 選手が正しいポジション（ネット際・バックライン）に落ちるか？")
    print("             センターライン両側に対称に散らばれば変換は正常")


if __name__ == "__main__":
    main()
