#!/usr/bin/env python3
"""
alt-pix pipeline: volleyball player + ball tracking.

Usage examples:
  # Analyse a local mp4 (batch mode)
  python scripts/run_pipeline.py --source game.mp4 --out-json out.jsonl --out-video out.mp4

  # Live SRT stream
  python scripts/run_pipeline.py --source "srt://192.168.1.10:9000?mode=caller"

  # Live RTMP stream
  python scripts/run_pipeline.py --source "rtmp://server/live/stream"
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
from loguru import logger

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from alt_pix.ball_tracker import BallTracker
from alt_pix.detector import YOLOXDetector, _COCO_PERSON, _COCO_SPORTS_BALL
from alt_pix.framing import FramingCalculator
from alt_pix.jersey_ocr import JerseyOCR
from alt_pix.output import JSONLWriter, VideoWriter, _make_record
from alt_pix.stream import iter_frames
from alt_pix.tracker import PlayerTracker
from alt_pix.visualizer import annotate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="alt-pix volleyball tracking pipeline")
    p.add_argument("--source", required=True, help="SRT URL, RTMP URL, or mp4 path")
    p.add_argument(
        "--person-model",
        default="models/yolox_m.onnx",
        help="YOLOX-m ONNX model for person detection",
    )
    p.add_argument(
        "--ball-model",
        default="models/yolox_s_ball.onnx",
        help="YOLOX-s ONNX model fine-tuned for volleyball detection",
    )
    p.add_argument("--conf-person", type=float, default=0.4, help="Person detection threshold")
    p.add_argument("--conf-ball", type=float, default=0.35, help="Ball detection threshold")
    p.add_argument(
        "--tracker",
        choices=["bytetrack"],
        default="bytetrack",
        help="MOT algorithm",
    )
    p.add_argument("--track-thresh", type=float, default=0.5,
                   help="Min detection confidence to start a track")
    p.add_argument("--track-buffer", type=int, default=30,
                   help="Frames to keep a lost track alive")
    p.add_argument("--skip-frames", type=int, default=0, help="Process every N+1 frames")
    p.add_argument("--no-ocr", action="store_true", help="Disable jersey number OCR")
    p.add_argument("--framing-mode", choices=["auto", "ball", "wide"], default="auto")
    p.add_argument("--out-json", default=None, help="Output JSONL log path")
    p.add_argument("--out-video", default=None, help="Output annotated mp4 path")
    p.add_argument("--device", default="cuda:0", help="Torch/ONNX device")
    p.add_argument("--show", action="store_true", help="Display live preview window")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger.info(f"Source: {args.source}")

    # --- Init components ---
    ort_device = "cuda" if args.device.startswith("cuda") else "cpu"

    logger.info("Loading person detector …")
    person_det = YOLOXDetector(
        args.person_model,
        conf_thr=args.conf_person,
        detect_classes={_COCO_PERSON},
        device=ort_device,
    )

    logger.info("Loading ball detector …")
    ball_det = YOLOXDetector(
        args.ball_model,
        conf_thr=args.conf_ball,
        detect_classes={_COCO_SPORTS_BALL},
        device=ort_device,
    )

    logger.info(f"Init tracker: {args.tracker}")
    tracker = PlayerTracker(
        method=args.tracker,
        track_activation_threshold=args.track_thresh,
        lost_track_buffer=args.track_buffer,
    )
    ball_tracker = BallTracker(ball_class_id=_COCO_SPORTS_BALL)

    ocr = None if args.no_ocr else JerseyOCR(use_gpu=ort_device == "cuda")
    jersey_map: dict[int, str] = {}

    json_writer: JSONLWriter | None = None
    if args.out_json:
        json_writer = JSONLWriter(args.out_json)

    video_writer: VideoWriter | None = None
    framing: FramingCalculator | None = None

    fps_counter = {"t0": time.time(), "n": 0}
    logger.info("Pipeline running. Press Ctrl-C to stop.")

    try:
        for frame_id, ts, frame in iter_frames(args.source, skip_frames=args.skip_frames):
            h, w = frame.shape[:2]

            # Init framing and video writer on first frame
            if framing is None:
                framing = FramingCalculator(w, h, mode=args.framing_mode)
            if video_writer is None and args.out_video:
                video_writer = VideoWriter(args.out_video, fps=30.0, size=(w, h))

            # Detection
            person_dets = person_det.detect(frame)
            ball_dets = ball_det.detect(frame)

            # Tracking
            tracks = tracker.update(person_dets, frame)
            ball_state = ball_tracker.update(ball_dets)

            # OCR (every frame; voting averages over time)
            if ocr is not None:
                jersey_map = ocr.update(frame, tracks)

            # Framing
            roi = framing.compute(ball_state, tracks)

            # Output
            record = _make_record(frame_id, ts * 1000, ball_state, tracks, jersey_map, roi)
            if json_writer:
                json_writer.write(record)

            # Visualization
            if args.out_video or args.show:
                vis = annotate(frame, tracks, ball_state, jersey_map, roi)
                if video_writer:
                    video_writer.write(vis)
                if args.show:
                    cv2.imshow("alt-pix", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            # FPS logging
            fps_counter["n"] += 1
            if fps_counter["n"] % 100 == 0:
                elapsed = time.time() - fps_counter["t0"]
                logger.info(
                    f"frame={frame_id}  ts={ts:.1f}s  fps={fps_counter['n']/elapsed:.1f}"
                    f"  players={len(tracks)}  ball={'✓' if ball_state.visible else '✗'}"
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        if json_writer:
            json_writer.close()
        if video_writer:
            video_writer.close()
        cv2.destroyAllWindows()
        logger.info("Done.")


if __name__ == "__main__":
    main()
