"""TrackingStats（GT 不要プロキシ指標）の単体テスト。numpy のみで動く。

Run:  python tests/test_track_eval.py
  or: python -m pytest tests/test_track_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alt_pix.track_eval import TrackingStats


def test_stable_tracks_high_lifetime_low_churn():
    """同じ 3 ID が 100 フレーム継続 → 安定（unique=3, short=0）。"""
    st = TrackingStats(short_threshold=5)
    for f in range(100):
        st.update(f, [1, 2, 3])
    s = st.summary()
    assert s["unique_ids"] == 3
    assert s["active_mean"] == 3.0
    assert s["active_std"] == 0.0
    assert s["lifetime_mean"] == 100.0
    assert s["short_lived"] == 0
    assert s["short_ratio"] == 0.0
    # 新規 ID は最初の 1 フレームで 3 個だけ。
    assert abs(s["new_id_rate_per100"] - 3.0) < 1e-9


def test_id_churn_inflates_unique_and_short_ratio():
    """毎フレーム新しい ID（switch だらけ）→ unique 過大・short_ratio=1。"""
    st = TrackingStats(short_threshold=5)
    for f in range(50):
        st.update(f, [f])  # 毎フレーム別 ID
    s = st.summary()
    assert s["unique_ids"] == 50
    assert s["short_lived"] == 50
    assert s["short_ratio"] == 1.0
    assert s["lifetime_mean"] == 1.0
    assert abs(s["new_id_rate_per100"] - 100.0) < 1e-9  # 毎フレーム 1 新規


def test_lifetime_histogram_buckets():
    st = TrackingStats()
    # ID 1: 3 フレーム（1-4 バケット）, ID 2: 20 フレーム（15-29 バケット）
    for f in range(3):
        st.update(f, [1])
    for f in range(20):
        st.update(100 + f, [2])
    hist = dict(st.lifetime_histogram())
    assert hist["1-4"] == 1
    assert hist["15-29"] == 1
    assert sum(hist.values()) == 2


def test_active_count_variation():
    """track 数が乱高下する → active_std が大きい。"""
    st = TrackingStats()
    for f in range(10):
        st.update(f, list(range(f % 5)))  # 0,1,2,3,4,0,1,...
    s = st.summary()
    assert s["active_max"] == 4
    assert s["active_min"] == 0
    assert s["active_std"] > 0


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
