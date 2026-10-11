"""
src/preprocessing/parse_inhouse.py

Loader for files exported by demo/collector.html (JSON Lines, one sample per line).

Each line:
  {writer, hand, device, pointer_types, prompt_id, text, canvas:[w,h], dpr,
   saved_at, strokes:[[ [x, y, t_seconds, pressure], ... ], ...]}

Unlike UJIpenchars2 these samples DO carry timestamps and pressure, and are
real words/lines, so they are the project's only source of continuous,
timed handwriting.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

Pt = Tuple[float, float, float, float]   # x, y, t, pressure


@dataclass
class InHouseSample:
    writer: str
    text: str
    device: str
    hand: str
    canvas: Tuple[float, float]
    strokes: List[List[Pt]]

    @property
    def num_points(self):
        return sum(len(s) for s in self.strokes)

    @property
    def duration(self):
        ts = [p[2] for s in self.strokes for p in s]
        return max(ts) - min(ts)


def parse_inhouse_file(path) -> List[InHouseSample]:
    out = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            strokes = [[tuple(float(v) for v in p) for p in s] for s in d["strokes"]]
            text = d["text"]
            writer = d["writer"]
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"{path}:{n}: malformed sample ({e})")
        if not text.strip():
            raise ValueError(f"{path}:{n}: empty text")
        if not strokes or any(len(s) == 0 for s in strokes):
            raise ValueError(f"{path}:{n}: empty stroke(s)")
        for s in strokes:
            for p in s:
                if len(p) != 4 or any(math.isnan(v) for v in p):
                    raise ValueError(f"{path}:{n}: bad point {p}")
            ts = [p[2] for p in s]
            if ts != sorted(ts):
                raise ValueError(f"{path}:{n}: timestamps go backwards within a stroke")
        out.append(InHouseSample(writer, text, d.get("device", ""), d.get("hand", ""),
                                 tuple(d.get("canvas", (0, 0))), strokes))
    if not out:
        raise ValueError(f"No samples in {path}")
    return out


def load_dir(directory) -> List[InHouseSample]:
    """Load every *.jsonl in a directory (one file per writer/session)."""
    files = sorted(Path(directory).glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files in {directory}")
    return [s for f in files for s in parse_inhouse_file(f)]


def writer_split(samples, test_writers):
    """Hold out whole writers. Raises if a requested test writer is absent."""
    tw = set(test_writers)
    missing = tw - {s.writer for s in samples}
    if missing:
        raise ValueError(f"Unknown writers: {sorted(missing)}")
    return [s for s in samples if s.writer not in tw], [s for s in samples if s.writer in tw]
