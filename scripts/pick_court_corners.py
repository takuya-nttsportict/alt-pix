#!/usr/bin/env python3
"""Interactive tool to click 4 court corners from a video frame.

Usage:
  python scripts/pick_court_corners.py \
    --source /data/volley-1.mp4 \
    --out    configs/court.json \
    [--frame 30]   # which frame to use (default: 0)

Click order: top-left, top-right, bottom-right, bottom-left (clockwise).
Press 'r' to reset, 's' to save, 'q' to quit without saving.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_frame(source: str, frame_no: int) -> np.ndarray:
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_no} from {source}")
    return frame


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--out", default="configs/court.json")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--player-margin", type=float, default=80.0)
    p.add_argument("--ball-margin", type=float, default=200.0)
    args = p.parse_args()

    frame = get_frame(args.source, args.frame)
    display = frame.copy()
    corners: list[tuple[int, int]] = []
    labels = ["1: top-left", "2: top-right", "3: bottom-right", "4: bottom-left"]

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append((x, y))
            cv2.circle(display, (x, y), 6, (0, 255, 255), -1)
            cv2.putText(display, str(len(corners)), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if len(corners) > 1:
                cv2.line(display, corners[-2], corners[-1], (0, 255, 255), 2)
            if len(corners) == 4:
                cv2.line(display, corners[-1], corners[0], (0, 255, 255), 2)
                cv2.putText(display, "Press 's' to save or 'r' to reset",
                            (10, display.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.namedWindow("Pick court corners")
    cv2.setMouseCallback("Pick court corners", mouse_cb)

    while True:
        hint = labels[len(corners)] if len(corners) < 4 else "Done — press 's' to save"
        canvas = display.copy()
        cv2.putText(canvas, hint, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow("Pick court corners", canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            corners.clear()
            display[:] = frame
        elif key == ord('s') and len(corners) == 4:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "corners": corners,
                "player_margin": args.player_margin,
                "ball_margin": args.ball_margin,
            }
            out.write_text(json.dumps(data, indent=2))
            print(f"Saved to {out}: {corners}")
            break
        elif key == ord('q'):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
