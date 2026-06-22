#!/usr/bin/env python3
"""
alt-pix pipeline: volleyball player + ball tracking.

Usage examples:
  # Analyse a local mp4 (batch mode), with court calibration
  python scripts/run_pipeline.py \\
    --source game.mp4 \\
    --person-model models/yolox_m.onnx \\
    --ball-model   models/yolox_s_ball.onnx \\
    --court        configs/court.json \\
    --out-json     out.jsonl \\
    --out-video    out.mp4

  # Live SRT / RTMP stream
  python scripts/run_pipeline.py \\
    --source "srt://192.168.1.10:9000?mode=caller" \\
    --person-model models/yolox_m.onnx \\
    --ball-model   models/yolox_s_ball.onnx

  # Pick court corners interactively first
  python scripts/pick_court_corners.py --source game.mp4 --out configs/court.json
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from alt_pix.ball_tracker import BallTracker
from alt_pix.court import CourtCalibration
from alt_pix.detector import YOLOXDetector, _COCO_PERSON, _COCO_SPORTS_BALL
from alt_pix.framing import FramingCalculator
from alt_pix.game_state import GameStateEstimator
from alt_pix.jersey_ocr import JerseyOCR
from alt_pix.log_config import setup_logging
from alt_pix.output import JSONLWriter, VideoWriter, _make_record
from alt_pix.roles import RoleClassifier
from alt_pix.stream import iter_frames
from alt_pix.team_assign import make_team_assigner
from alt_pix.team_classifier import TeamClassifier
from alt_pix.tracknet import TrackNetDetector
from alt_pix.tracker import PlayerTracker
from alt_pix.visualizer import annotate

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="alt-pix volleyball tracking pipeline")
    p.add_argument("--source", required=True,
                   help="SRT URL, RTMP URL, or mp4 file path")

    # Models
    p.add_argument("--person-model", default="models/yolox_m.onnx")
    p.add_argument("--ball-model",   default="models/tracknet_volleyball.pt",
                   help="Ball detector. .pt = TrackNet (recommended), .onnx = YOLOX")
    p.add_argument("--conf-person",  type=float, default=0.4)
    p.add_argument("--conf-ball",    type=float, default=0.5,
                   help="Ball threshold (heatmap peak for TrackNet, score for YOLOX)")

    # Court calibration
    p.add_argument("--court", default=None,
                   help="Path to court calibration JSON (from pick_court_corners.py). "
                        "If omitted, court filtering is disabled.")

    # Tracking
    p.add_argument("--track-thresh",  type=float, default=0.5)
    p.add_argument("--track-buffer",  type=int,   default=30)

    # Perception core (Phase 4): team classification + role filtering
    p.add_argument("--sport", choices=["volleyball", "basketball", "generic"],
                   default="volleyball",
                   help="Team-assignment strategy by sport: volleyball uses the "
                        "court-half (net) boundary; others use uniform colour clustering")
    p.add_argument("--no-team", action="store_true",
                   help="Disable team assignment")
    p.add_argument("--team-backend", choices=["siglip", "lab", "hsv"], default="siglip",
                   help="Uniform embedding backend for colour clustering "
                        "(siglip falls back to lab if unavailable)")
    p.add_argument("--no-role", action="store_true",
                   help="Disable role classification (field/bench/referee). "
                        "Requires --court.")
    p.add_argument("--scene-margin", type=float, default=300.0,
                   help="Keep people within this many px of the court (drops far "
                        "spectators); requires --court. Bench/referee fall inside it.")

    # OCR
    p.add_argument("--no-ocr", action="store_true", help="Disable jersey OCR")

    # Framing
    p.add_argument("--framing-mode", choices=["auto", "ball", "wide"], default="auto")
    p.add_argument("--framing-style", choices=["normal", "dynamic"], default="normal",
                   help="normal=放送的で安定。dynamic=有人カメラ的に寄り強め・機敏。")

    # Output
    p.add_argument("--out-json",  default=None, help="Output JSONL log path")
    p.add_argument("--out-video", default=None, help="Output annotated mp4 path")
    p.add_argument("--show",      action="store_true", help="Display live preview")

    # Performance
    p.add_argument("--skip-frames", type=int, default=0,
                   help="Process every (N+1)-th frame for speed")
    p.add_argument("--device", default="cuda")

    # Logging
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-file", default=None, help="Also write logs to this file")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level, log_file=args.log_file)

    logger.info("=" * 60)
    logger.info(f"alt-pix pipeline starting")
    logger.info(f"  source       : {args.source}")
    logger.info(f"  person-model : {args.person_model}")
    logger.info(f"  ball-model   : {args.ball_model}")
    logger.info(f"  court        : {args.court or '(disabled)'}")
    logger.info(f"  device       : {args.device}")
    logger.info(f"  log-level    : {args.log_level}")
    logger.info("=" * 60)

    ort_device = "cuda" if args.device.startswith("cuda") else "cpu"

    # ── Models ────────────────────────────────────────────────────────────────
    logger.info("Loading person detector …")
    person_det = YOLOXDetector(
        args.person_model, conf_thr=args.conf_person,
        detect_classes={_COCO_PERSON}, device=ort_device,
    )

    logger.info("Loading ball detector …")
    if Path(args.ball_model).suffix == ".pt":
        ball_det = TrackNetDetector(
            args.ball_model, conf_thr=args.conf_ball, device=ort_device,
        )
    else:
        ball_det = YOLOXDetector(
            args.ball_model, conf_thr=args.conf_ball,
            detect_classes={_COCO_SPORTS_BALL}, device=ort_device,
        )

    logger.info("Initialising tracker …")
    tracker = PlayerTracker(
        track_activation_threshold=args.track_thresh,
        lost_track_buffer=args.track_buffer,
    )
    ball_tracker = BallTracker()

    # ── Court calibration ─────────────────────────────────────────────────────
    court: CourtCalibration | None = None
    if args.court:
        logger.info(f"Loading court calibration from {args.court} …")
        court = CourtCalibration.load(args.court)
    else:
        logger.warning(
            "No court calibration specified (--court). "
            "All detections will be used without spatial filtering."
        )

    # ── Team assignment + role filtering (Phase 4 perception core) ─────────────
    # Team-assignment strategy depends on the sport (principle 4): net sports
    # split deterministically by court half; invasion sports by uniform colour.
    assigner = None
    if not args.no_team:
        team_clf = TeamClassifier(backend=args.team_backend, device=args.device)
        assigner = make_team_assigner(args.sport, court, team_clf)
        logger.info(f"Team assigner: {assigner.method} (sport={args.sport})")

    role_clf: RoleClassifier | None = None
    if not args.no_role:
        if court is not None:
            logger.info("Initialising role classifier (court geometry + colour outlier) …")
            role_clf = RoleClassifier(court)
        else:
            logger.warning(
                "Role classification requested but no --court given; disabled. "
                "Roles need court geometry to separate field / bench / referee."
            )

    # ── OCR ───────────────────────────────────────────────────────────────────
    ocr = None
    if not args.no_ocr:
        logger.info("Initialising jersey OCR …")
        ocr = JerseyOCR(use_gpu=(ort_device == "cuda"))
    jersey_map: dict[int, str] = {}

    # ── Output writers (initialised on first frame) ───────────────────────────
    json_writer: JSONLWriter | None = JSONLWriter(args.out_json) if args.out_json else None
    video_writer: VideoWriter | None = None
    framing: FramingCalculator | None = None

    # ── Game-state estimator (drives state-aware framing) ──────────────────────
    # Shares the role classifier's participation tracker (on-court count EMA) so
    # NO_PLAY (timeout / set break) and SERVICE (ball at endline) gate framing.
    game_state_est = GameStateEstimator(
        court, role_clf.participation if role_clf is not None else None)

    # ── Stats ──────────────────────────────────────────────────────────────────
    t_start = time.time()
    n_frames = 0
    n_persons_total = 0
    ball_visible_frames = 0
    LOG_INTERVAL = 100  # frames

    logger.info("Pipeline running. Press Ctrl-C to stop.")

    try:
        for frame_id, ts, frame in iter_frames(args.source, skip_frames=args.skip_frames):
            h, w = frame.shape[:2]

            if framing is None:
                framing = FramingCalculator(w, h, mode=args.framing_mode,
                                            style=args.framing_style)
                logger.info(f"First frame: {w}x{h} (framing style={args.framing_style})")
            if video_writer is None and args.out_video:
                video_writer = VideoWriter(args.out_video, fps=30.0, size=(w, h))

            # ── Detection ─────────────────────────────────────────────────────
            person_dets = person_det.detect(frame)
            ball_dets   = ball_det.detect(frame)

            # ── Court filtering ────────────────────────────────────────────────
            if court is not None:
                # Drop far spectators. When role classification is on we keep a
                # wider band (scene margin) so bench/referee survive to be
                # *labelled* rather than discarded — analytics needs them too.
                keep_margin = args.scene_margin if role_clf is not None else court.player_margin
                person_dets = [d for d in person_dets
                               if court.is_on_court(d.bbox, keep_margin)]

                b_mask = court.filter_ball([d.bbox for d in ball_dets])
                ball_dets = [d for d, ok in zip(ball_dets, b_mask) if ok]

            # ── Tracking ──────────────────────────────────────────────────────
            tracks    = tracker.update(person_dets, frame)
            ball_state = ball_tracker.update(ball_dets)

            # ── Team assignment + role labelling (perception core) ─────────────
            dist_map: dict[int, float] = {}
            if assigner is not None:
                result = assigner.update(frame, tracks)
                dist_map = result.dist_map
                for t in tracks:
                    tm = result.team_map.get(t.track_id, -1)
                    t.team = tm if tm >= 0 else None
                    t.team_reason = result.reason_map.get(t.track_id)
            if role_clf is not None:
                team_map_safe = {t.track_id: (t.team if t.team is not None else -1)
                                 for t in tracks}
                role_map = role_clf.classify(tracks, team_map_safe, dist_map)
                for t in tracks:
                    t.role = role_map.get(t.track_id)
                    t.role_reason = role_clf.last_reasons.get(t.track_id)

            # Field players (framing-relevant). Without role info, treat all as field.
            field_tracks = [t for t in tracks if t.role in (None, "field")]

            # ── OCR ───────────────────────────────────────────────────────────
            if ocr is not None:
                jersey_map = ocr.update(frame, tracks)

            # ── Framing ───────────────────────────────────────────────────────
            # State-aware camera work: RALLY follows ball+players at mid zoom;
            # SERVICE points at the server (assume_server) at a mid zoom rather
            # than a loose wide; NO_PLAY (timeout / set break) freezes the pan and
            # holds a wide shot. Critically-damped spring smoothing removes jitter.
            game_state = game_state_est.update(ball_state, field_tracks)
            focus_xy = None
            if game_state == "service" and role_clf is not None:
                sid = role_clf.participation.server_track_id()
                if sid is not None:
                    st = next((t for t in tracks if t.track_id == sid), None)
                    if st is not None:
                        x1, _y1, x2, y2 = st.bbox
                        focus_xy = ((x1 + x2) / 2.0, float(y2))  # server's feet
            roi = framing.compute(ball_state, field_tracks,
                                  game_state=game_state, focus_xy=focus_xy)

            # ── Output ────────────────────────────────────────────────────────
            record = _make_record(frame_id, ts * 1000, ball_state, tracks, jersey_map, roi)
            if json_writer:
                json_writer.write(record)

            if args.out_video or args.show:
                vis = annotate(frame, tracks, ball_state, jersey_map, roi)
                if court is not None:
                    court.draw(vis)
                if video_writer:
                    video_writer.write(vis)
                if args.show:
                    cv2.imshow("alt-pix", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            # ── Per-frame stats (DEBUG) ────────────────────────────────────────
            n_frames += 1
            n_persons_total += len(tracks)
            if ball_state.visible:
                ball_visible_frames += 1

            n_field = sum(1 for t in tracks if t.role in (None, "field"))
            n_ref = sum(1 for t in tracks if t.role == "referee")
            logger.debug(
                f"frame={frame_id:5d} ts={ts:7.2f}s "
                f"persons={len(person_dets):2d} "
                f"tracks={len(tracks):2d} field={n_field:2d} ref={n_ref:1d} "
                f"team_ready={'Y' if (assigner and assigner.ready) else 'N'} "
                f"ball={'Y' if ball_state.visible else 'N'} "
                f"ball_conf={ball_state.conf:.2f} "
                f"state={game_state} "
                f"roi=({roi.x},{roi.y},{roi.w},{roi.h})"
            )

            # ── Per-interval summary (INFO) ────────────────────────────────────
            if n_frames % LOG_INTERVAL == 0:
                elapsed = time.time() - t_start
                fps = n_frames / elapsed
                ball_rate = ball_visible_frames / n_frames * 100
                avg_persons = n_persons_total / n_frames
                logger.info(
                    f"[frame {frame_id:5d}] "
                    f"fps={fps:5.1f}  "
                    f"avg_tracks={avg_persons:4.1f}  "
                    f"ball_visible={ball_rate:4.1f}%  "
                    f"elapsed={elapsed:.0f}s"
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        elapsed = time.time() - t_start
        fps = n_frames / max(elapsed, 1e-3)
        ball_rate = ball_visible_frames / max(n_frames, 1) * 100
        logger.info("=" * 60)
        logger.info(f"Pipeline finished")
        logger.info(f"  frames processed : {n_frames}")
        logger.info(f"  elapsed          : {elapsed:.1f}s")
        logger.info(f"  avg fps          : {fps:.1f}")
        logger.info(f"  ball visible     : {ball_rate:.1f}%")
        logger.info("=" * 60)

        if json_writer:
            json_writer.close()
        if video_writer:
            video_writer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
