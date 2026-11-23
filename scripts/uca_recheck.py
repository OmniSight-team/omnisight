"""K-20's re-check: a fresh sample of UCA-timed windows, after D-49's 8 s margin (OMNI-901).

Two steps, no database (UCF-Crime is not loaded in the product; the clips are watched from
the Drive copy in any player):

  uv run python scripts/uca_recheck.py draw     # writes datasets/hand_check/uca_recheck.csv
  uv run python scripts/uca_recheck.py report   # after the checker filled visible_start_s

**draw:** 15 clips whose 30 s window is UCA-timed (train or validation, as rebuilt by D-49),
none of them in OMNI-609's sample, spread over the classes round-robin like D-26, seed 901
(a new seed). Like the other samples it is fixed: a redraw that differs writes nothing.

**What the checker records, per clip** (D-49, written before any clip is watched): in
`visible_start_s`, the second of the video where the act visibly begins, judged by D-27's
per-class rules (the act itself, not who will do it); `none` if the act is never seen;
`?` if the footage does not show it clearly. A short `note` is optional.

**report:** per clip, and in total with a 95% Wilson interval:
- **clean** (the K-20 question): the window shows none of the event, i.e. the act begins no
  earlier than 1.5 s (D-27's tolerance) before the window's end
- **UCA time correct** (D-27 applied to the UCA time itself, comparable with OMNI-609's
  10 of 15): the act begins within 1.5 s of the UCA time
- how far before the visible act the window ends
"""

import hashlib
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

from omnisight_common.datasets.windows import UCA_MARGIN, UCA_MARGIN_S

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "datasets" / "windows" / "ucf_crime.csv"
FIRST_SAMPLE = ROOT / "datasets" / "hand_check" / "sample.csv"
OUT = ROOT / "datasets" / "hand_check" / "uca_recheck.csv"
SEED = 901
SIZE = 15
TOLERANCE_S = 1.5  # D-27
HEADER = "clip_key,video_relpath,event_class,split,uca_event_s,window_start_s,window_end_s,visible_start_s,note"


def rows(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    head = lines[0].split(",")
    return [dict(zip(head, ln.split(",", len(head) - 1))) for ln in lines[1:] if ln.strip()]


def mmss(s: float) -> str:
    return f"{int(s // 60)}:{s % 60:04.1f}"


def draw() -> int:
    checked = {r["clip_key"] for r in rows(FIRST_SAMPLE)}
    pool = [
        r for r in rows(WINDOWS)
        if r["kind"] == "positive" and r["nominal_length_s"] == "30" and UCA_MARGIN in r["flags"].split(";")
        and r["clip_key"] not in checked
    ]
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in sorted(pool, key=lambda r: r["clip_key"]):
        by_class[r["event_class"]].append(r)
    rng = random.Random(SEED)
    for cls in sorted(by_class):
        rng.shuffle(by_class[cls])
    picked: list[dict[str, str]] = []
    while len(picked) < SIZE and any(by_class.values()):
        for cls in sorted(by_class):
            if by_class[cls] and len(picked) < SIZE:
                picked.append(by_class[cls].pop(0))

    lines = [HEADER]
    for r in sorted(picked, key=lambda r: r["clip_key"]):
        uca = round(float(r["window_end_s"]) + UCA_MARGIN_S, 3)
        lines.append(
            f"{r['clip_key']},{r['video_relpath']},{r['event_class']},{r['split']},{uca:g},"
            f"{r['window_start_s']},{r['window_end_s']},,"
        )
    data = ("\n".join(lines) + "\n").encode("utf-8")

    def drawn(text: str) -> list[list[str]]:  # the sample without the checker's two columns
        return [ln.split(",", 7)[:7] for ln in text.splitlines()]

    if OUT.exists() and drawn(OUT.read_text(encoding="utf-8")) != drawn(data.decode()):
        print(f"{OUT} exists and the redrawn sample differs: the sample is fixed. Nothing written.")
        return 1
    if not OUT.exists():
        OUT.write_bytes(data)
    print(f"seed {SEED}, {len(picked)} of {len(pool)} UCA-timed clips; OMNI-609's sample left out")
    print(f"  {'clip':28s}{'class':15s}{'window':>18s}{'UCA time':>10s}")
    for r in sorted(picked, key=lambda r: r["clip_key"]):
        s, e = float(r["window_start_s"]), float(r["window_end_s"])
        print(f"  {r['clip_key']:28s}{r['event_class']:15s}{mmss(s):>8s} - {mmss(e):>7s}{mmss(e + UCA_MARGIN_S):>10s}")
    print(f"{OUT.relative_to(ROOT).as_posix()} sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}")
    return 0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def report() -> int:
    checked = [r for r in rows(OUT) if r["visible_start_s"].strip()]
    clear = [r for r in checked if r["visible_start_s"].strip() not in ("none", "?")]
    print(f"{len(checked)} of {len(rows(OUT))} clips checked; {len(checked) - len(clear)} marked none or ?")
    clean = correct = 0
    print(f"  {'clip':28s}{'window end':>11s}{'UCA':>8s}{'seen':>8s}{'end before act':>16s}  clean  UCA ok  note")
    for r in checked:
        end, uca, seen = float(r["window_end_s"]), float(r["uca_event_s"]), r["visible_start_s"].strip()
        if seen in ("none", "?"):
            print(f"  {r['clip_key']:28s}{mmss(end):>11s}{mmss(uca):>8s}{seen:>8s}{'':>16s}  -      -       {r['note']}")
            continue
        v = float(seen)
        is_clean, is_correct = v >= end - TOLERANCE_S, abs(v - uca) <= TOLERANCE_S
        clean += is_clean
        correct += is_correct
        print(
            f"  {r['clip_key']:28s}{mmss(end):>11s}{mmss(uca):>8s}{mmss(v):>8s}{v - end:>14.1f} s  "
            f"{'yes' if is_clean else 'NO ':5s}  {'yes' if is_correct else 'no ':6s}  {r['note']}"
        )
    n = len(clear)
    for name, k in (("clean (window shows none of the event)", clean), ("UCA time correct (D-27)", correct)):
        lo, hi = wilson(k, n)
        print(f"{name}: {k} of {n}" + (f" ({100 * k / n:.0f}%, 95% {100 * lo:.0f}-{100 * hi:.0f}%)" if n else ""))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in ("draw", "report"):
        print(__doc__)
        sys.exit(2)
    sys.exit(draw() if cmd == "draw" else report())
