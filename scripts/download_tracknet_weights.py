#!/usr/bin/env python3
"""Download pre-trained TrackNetV3 weights.

Two weight sources are available:
  (A) qaz812345/TrackNetV3 — badminton shuttlecock, BMVC 2023
      Google Drive: https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view
      This is the official TrackNetV3 (two-module: TrackNet + InpaintNet).
      We only need the TrackNet module (.pt file named TrackNet_best.pt inside the zip).

  (B) nttcom/WASB-SBDT — volleyball-specific, BMVC 2023 (MIT licence)
      Google Drive: https://drive.google.com/file/d/103jOdYp4k20avid4uyB9USCuwiphI4Kz/view
      TrackNetV2 architecture trained on volleyball footage.
      Recommended starting point since it is already volleyball-domain.

Usage:
  # Volleyball-specific weights (recommended):
  python scripts/download_tracknet_weights.py --source wasb --out models/tracknet_volleyball.pt

  # Shuttlecock weights (badminton, general starting point):
  python scripts/download_tracknet_weights.py --source tracknetv3 --out models/tracknet_shuttlecock.pt

  # After downloading, run the pipeline with:
  python scripts/run_pipeline.py \\
    --source game.mp4 \\
    --person-model models/yolox_m.onnx \\
    --ball-model   models/tracknet_volleyball.pt \\
    --court        configs/court.json \\
    --out-video    out.mp4

Requires gdown:
  pip install gdown
"""

import argparse
import sys
import zipfile
from pathlib import Path

# ── Source definitions ────────────────────────────────────────────────────────

_SOURCES = {
    "wasb": {
        "description": "WASB-SBDT volleyball weights (nttcom, MIT, BMVC 2023)",
        "gdrive_id": "103jOdYp4k20avid4uyB9USCuwiphI4Kz",
        "is_zip": False,
        "zip_entry": None,
    },
    "tracknetv3": {
        "description": "TrackNetV3 shuttlecock weights (qaz812345, BMVC 2023)",
        "gdrive_id": "1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA",
        "is_zip": True,
        "zip_entry": "TrackNet_best.pt",   # file inside the zip
    },
}


def _check_gdown() -> None:
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("ERROR: gdown is required for Google Drive downloads.", file=sys.stderr)
        print("  pip install gdown", file=sys.stderr)
        sys.exit(1)


def _download_gdrive(gdrive_id: str, dst: Path) -> None:
    import gdown
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    print(f"Downloading from Google Drive (id={gdrive_id})")
    print(f"  → {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(dst), quiet=False)
    if not dst.exists() or dst.stat().st_size < 1024:
        print("ERROR: Download failed or file too small.", file=sys.stderr)
        print(
            "If gdown shows 'Permission denied', the file may require sign-in.\n"
            "Manual download:\n"
            f"  1. Open https://drive.google.com/uc?id={gdrive_id}\n"
            f"  2. Save to: {dst}",
            file=sys.stderr,
        )
        sys.exit(1)


def _extract_from_zip(zip_path: Path, entry: str, dst: Path) -> None:
    print(f"Extracting '{entry}' from {zip_path} …")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        matches = [n for n in names if n.endswith(entry)]
        if not matches:
            print(f"ERROR: '{entry}' not found in zip. Contents: {names}", file=sys.stderr)
            sys.exit(1)
        with zf.open(matches[0]) as src, open(dst, "wb") as out:
            out.write(src.read())
    print(f"Extracted → {dst}")


def _convert_checkpoint(src: Path, dst: Path, device: str) -> None:
    """Strip DataParallel 'module.' prefix if present."""
    import torch

    print(f"Loading checkpoint {src} …")
    ckpt = torch.load(str(src), map_location=device, weights_only=False)

    # Unwrap common wrapper formats
    if isinstance(ckpt, dict):
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    else:
        state = ckpt

    # Strip DataParallel prefix
    cleaned = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }

    torch.save(cleaned, str(dst))
    print(f"Saved converted checkpoint → {dst}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download pre-trained TrackNet weights for volleyball ball detection"
    )
    p.add_argument(
        "--source",
        choices=list(_SOURCES.keys()),
        default="wasb",
        help=(
            "wasb: volleyball-specific from nttcom/WASB-SBDT (recommended). "
            "tracknetv3: shuttlecock from qaz812345/TrackNetV3."
        ),
    )
    p.add_argument("--out", default="models/tracknet_volleyball.pt",
                   help="Output path for the .pt file")
    p.add_argument("--cpu", action="store_true",
                   help="Use CPU for checkpoint conversion (default: auto-detect)")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip download if raw file already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src_cfg = _SOURCES[args.source]
    device = "cpu" if args.cpu else ("cuda" if _has_cuda() else "cpu")

    print(f"Source : {args.source} — {src_cfg['description']}")
    print(f"Output : {args.out}")
    print()

    _check_gdown()

    dst = Path(args.out)
    raw_suffix = ".zip" if src_cfg["is_zip"] else ".pt"
    raw = dst.parent / ("_raw_" + dst.stem + raw_suffix)

    # ── Download ──────────────────────────────────────────────────────────────
    if raw.exists() and args.skip_download:
        print(f"Skipping download — {raw} already exists.")
    else:
        _download_gdrive(src_cfg["gdrive_id"], raw)

    # ── Extract (if zip) ──────────────────────────────────────────────────────
    if src_cfg["is_zip"]:
        extracted = dst.parent / ("_extracted_" + dst.stem + ".pt")
        _extract_from_zip(raw, src_cfg["zip_entry"], extracted)
        src_pt = extracted
    else:
        src_pt = raw

    # ── Convert ───────────────────────────────────────────────────────────────
    _convert_checkpoint(src_pt, dst, device)

    print()
    print(f"✓  TrackNet weights ready at: {dst}")
    print(f"   Use with: --ball-model {dst}")


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
