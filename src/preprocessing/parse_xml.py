"""
src/preprocessing/parse_xml.py

Parses IAM-OnDB "lineStrokes" XML files (WhiteboardCaptureSession format)
into Python data structures, and matches each parsed line against its
ground-truth transcription from the accompanying ascii file.

This is the atomic first step of the preprocessing pipeline (Step C1.1-C1.2
in the ScribbleAI build guide). Every later preprocessing function
(duplicate-point removal, slant correction, resampling, feature encoding)
takes its input from what this module produces.
"""

from pathlib import Path
from lxml import etree


class ParsedLine:
    """One parsed handwriting line: its raw strokes + ground-truth text."""

    def __init__(self, form_id, line_id, strokes, text):
        self.form_id = form_id
        self.line_id = line_id
        self.strokes = strokes  # list of list of (x, y, t) tuples
        self.text = text

    @property
    def num_strokes(self):
        return len(self.strokes)

    @property
    def num_points(self):
        return sum(len(s) for s in self.strokes)

    def __repr__(self):
        return (f"ParsedLine(form={self.form_id}, line={self.line_id}, "
                f"strokes={self.num_strokes}, points={self.num_points}, "
                f"text={self.text!r})")


def parse_stroke_xml(xml_path):
    """
    Parse a single IAM-OnDB lineStrokes XML file.

    Returns: list of strokes, where each stroke is a list of (x, y, t) tuples,
    in the exact order they appear in the file.

    Raises:
        ValueError if the file has no strokes or malformed points — fail loud
        rather than silently returning empty data (this is what Step C1.3's
        unit tests check for).
    """
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    stroke_set = root.find("StrokeSet")
    if stroke_set is None:
        raise ValueError(f"No <StrokeSet> found in {xml_path}")

    strokes = []
    for stroke_elem in stroke_set.findall("Stroke"):
        points = []
        for point_elem in stroke_elem.findall("Point"):
            try:
                x = float(point_elem.get("x"))
                y = float(point_elem.get("y"))
                t = float(point_elem.get("time"))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Malformed Point in {xml_path}, stroke starting at "
                    f"time={stroke_elem.get('start_time')}: {e}"
                )
            points.append((x, y, t))

        if len(points) == 0:
            raise ValueError(
                f"Empty <Stroke> (no points) in {xml_path} — "
                f"start_time={stroke_elem.get('start_time')}"
            )
        strokes.append(points)

    if len(strokes) == 0:
        raise ValueError(f"No strokes found in {xml_path}")

    return strokes


def parse_ascii_transcription(ascii_path):
    """
    Parse an IAM-OnDB ascii transcription file into an ordered dict of
    {line_number_str: text}.

    Real files look like:
        a01-000u

        CSR:

        the quick brown fox
        jumps over the lazy dog
        handwriting recognition is fun

    Lines after "CSR:" correspond, in order, to line 01, 02, 03, ...
    """
    lines = Path(ascii_path).read_text().splitlines()

    try:
        csr_idx = lines.index("CSR:")
    except ValueError:
        raise ValueError(f"No 'CSR:' marker found in {ascii_path}")

    text_lines = [line for line in lines[csr_idx + 1:] if line.strip() != ""]

    if len(text_lines) == 0:
        raise ValueError(f"No transcription lines found after CSR: in {ascii_path}")

    return {f"{i+1:02d}": text for i, text in enumerate(text_lines)}


def load_form(form_id, linestrokes_dir, ascii_dir):
    """
    Load every parsed line for a given form_id, matching strokes to their
    ground-truth transcription.

    Returns: list of ParsedLine objects, one per line file found on disk.
    """
    linestrokes_dir = Path(linestrokes_dir)
    ascii_path = Path(ascii_dir) / f"{form_id}.txt"

    if not ascii_path.exists():
        raise FileNotFoundError(f"No ascii transcription file at {ascii_path}")

    transcriptions = parse_ascii_transcription(ascii_path)

    line_files = sorted(linestrokes_dir.glob(f"{form_id}-*.xml"))
    if len(line_files) == 0:
        raise FileNotFoundError(
            f"No line XML files found for form '{form_id}' in {linestrokes_dir}"
        )

    parsed_lines = []
    for xml_path in line_files:
        # filename pattern: <form_id>-<line_number>.xml
        line_id = xml_path.stem.split("-")[-1]
        strokes = parse_stroke_xml(xml_path)

        if line_id not in transcriptions:
            raise ValueError(
                f"Line {line_id} from {xml_path.name} has no matching "
                f"transcription in {ascii_path}"
            )

        parsed_lines.append(
            ParsedLine(form_id, line_id, strokes, transcriptions[line_id])
        )

    return parsed_lines


if __name__ == "__main__":
    # Quick manual smoke test against the synthetic data.
    lines = load_form(
        form_id="a01-000u",
        linestrokes_dir="data/raw/synthetic/lineStrokes",
        ascii_dir="data/raw/synthetic/ascii",
    )
    for line in lines:
        print(line)
