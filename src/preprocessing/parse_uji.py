"""
src/preprocessing/parse_uji.py

Parser for UJIpenchars2 (UCI ML Repository, dataset 177).

File format (verified against the real file):
    // comment lines
    WORD <char> <sample_id>          e.g.  WORD a trn_UJI_W01-01
      NUMSTROKES <n>
      POINTS <k> # x1 y1 x2 y2 ...   (one POINTS line per stroke)

Facts that matter for the pipeline:
  * UTF-8 encoded, CRLF line endings.
  * Only (x, y) per point. NO timestamps. Units: 100 per millimetre.
  * sample_id = <split>_<site>_W<writer>-<repetition>; split is 'trn' or 'tst'
    and the two sets use different writers (writer-disjoint by construction).
  * Consecutive duplicate points are common (handled in a later step).
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

Point = Tuple[float, float]

_ID_RE = re.compile(r"^(trn|tst)_(UJI|UPV)_W(\d+)-(\d+)$")


@dataclass
class UJISample:
    label: str                 # the character, e.g. "a", "Ñ", "€"
    sample_id: str             # e.g. "trn_UJI_W01-01"
    split: str                 # "trn" or "tst"
    site: str                  # "UJI" or "UPV"
    writer: int                # 1..60
    repetition: int            # 1 or 2
    strokes: List[List[Point]]

    @property
    def num_strokes(self) -> int:
        return len(self.strokes)

    @property
    def num_points(self) -> int:
        return sum(len(s) for s in self.strokes)


def parse_uji_file(path) -> List[UJISample]:
    """Parse ujipenchars2.txt. Fails loudly on any structural inconsistency."""
    text = Path(path).read_text(encoding="utf-8").replace("\r", "")
    samples: List[UJISample] = []

    cur = None          # (label, sample_id)
    expected = None     # NUMSTROKES for the current sample
    strokes: List[List[Point]] = []

    def finish():
        if cur is None:
            return
        label, sid = cur
        if expected is None or len(strokes) != expected:
            raise ValueError(
                f"{sid}: NUMSTROKES={expected} but found {len(strokes)} POINTS lines")
        m = _ID_RE.match(sid)
        if not m:
            raise ValueError(f"Unrecognised sample id: {sid!r}")
        split, site, writer, rep = m.groups()
        samples.append(UJISample(label, sid, split, site, int(writer), int(rep),
                                 [list(s) for s in strokes]))

    for lineno, raw in enumerate(text.split("\n"), 1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("WORD "):
            finish()
            parts = line.split(" ")
            if len(parts) != 3:
                raise ValueError(f"line {lineno}: malformed WORD line: {line!r}")
            cur = (parts[1], parts[2])
            expected, strokes = None, []
        elif line.startswith("NUMSTROKES"):
            expected = int(line.split()[1])
        elif line.startswith("POINTS"):
            head, _, body = line.partition("#")
            n = int(head.split()[1])
            vals = body.split()
            if len(vals) != 2 * n:
                raise ValueError(
                    f"line {lineno}: POINTS says {n} points but has {len(vals)} values")
            xs = [float(v) for v in vals]
            strokes.append(list(zip(xs[0::2], xs[1::2])))
        else:
            raise ValueError(f"line {lineno}: unrecognised line: {line[:60]!r}")
    finish()

    if not samples:
        raise ValueError(f"No samples found in {path}")
    return samples


def make_splits(samples: List[UJISample], val_writers=(1, 2, 3, 4, 5, 6)):
    """
    Writer-disjoint train/val/test split.

    test  = every 'tst' sample (20 writers, defined by the dataset authors)
    val   = 'trn' samples from `val_writers` (carved out of the train writers)
    train = remaining 'trn' samples

    Raises if any writer appears in more than one split.
    """
    test = [s for s in samples if s.split == "tst"]
    trn = [s for s in samples if s.split == "trn"]
    val_set = set(val_writers)
    val = [s for s in trn if s.writer in val_set]
    train = [s for s in trn if s.writer not in val_set]

    w = lambda xs: {s.writer for s in xs}
    if w(train) & w(val) or w(train) & w(test) or w(val) & w(test):
        raise ValueError("Writer leakage between splits")
    if not val:
        raise ValueError("val_writers matched no training writers")
    return train, val, test


if __name__ == "__main__":
    data = parse_uji_file("data/raw/ujipenchars2/ujipenchars2.txt")
    train, val, test = make_splits(data)
    print(f"total={len(data)} train={len(train)} val={len(val)} test={len(test)}")
