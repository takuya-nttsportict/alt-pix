"""Stream ingestion via PyAV: supports SRT, RTMP, and local mp4."""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Generator

import av
import numpy as np


def _open_container(source: str) -> av.container.InputContainer:
    options: dict[str, str] = {}

    if source.startswith("srt://"):
        options = {"fflags": "nobuffer", "flags": "low_delay"}
    elif source.startswith("rtmp://"):
        options = {"fflags": "nobuffer", "rtmp_live": "live"}
    elif Path(source).is_file():
        pass  # local file: no special options needed

    return av.open(source, options=options)


def iter_frames(
    source: str,
    skip_frames: int = 0,
) -> Generator[tuple[int, float, np.ndarray], None, None]:
    """Yield (frame_index, timestamp_sec, BGR ndarray) from any supported source.

    Args:
        source: SRT URL, RTMP URL, or local mp4 path.
        skip_frames: Yield every (skip_frames+1)-th frame (0 = every frame).
    """
    container = _open_container(source)
    video_stream = next(s for s in container.streams if s.type == "video")
    video_stream.thread_type = "AUTO"

    frame_idx = 0
    for packet in container.demux(video_stream):
        for av_frame in packet.decode():
            if skip_frames and frame_idx % (skip_frames + 1) != 0:
                frame_idx += 1
                continue

            bgr = av_frame.to_ndarray(format="bgr24")
            ts = float(av_frame.pts * av_frame.time_base) if av_frame.pts is not None else 0.0
            yield frame_idx, ts, bgr
            frame_idx += 1

    container.close()


class AsyncFrameQueue:
    """Background thread that decodes frames into a queue for downstream processing."""

    def __init__(self, source: str, maxsize: int = 32, skip_frames: int = 0) -> None:
        self._source = source
        self._skip = skip_frames
        self._q: queue.Queue[tuple[int, float, np.ndarray] | None] = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            for item in iter_frames(self._source, skip_frames=self._skip):
                self._q.put(item)
        finally:
            self._q.put(None)  # sentinel

    def __iter__(self) -> Generator[tuple[int, float, np.ndarray], None, None]:
        while True:
            item = self._q.get()
            if item is None:
                break
            yield item
