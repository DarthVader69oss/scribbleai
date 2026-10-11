"""
tests/test_parse_xml.py

Unit tests for src/preprocessing/parse_xml.py — Step C1.3 from the build guide.

Run with:
    pytest tests/test_parse_xml.py -v

These catch silent corruption (malformed points, empty strokes, mismatched
transcriptions) before it propagates into preprocessing or training, where
bugs are much harder to trace back to their source.
"""

import sys
import os
import pytest
from pathlib import Path
from lxml import etree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing.parse_xml import (
    parse_stroke_xml,
    parse_ascii_transcription,
    load_form,
    ParsedLine,
)

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "iamondb"
LINESTROKES_DIR = FIXTURES_DIR / "lineStrokes"
ASCII_DIR = FIXTURES_DIR / "ascii"


# ---------------------------------------------------------------------------
# parse_stroke_xml
# ---------------------------------------------------------------------------

def test_parse_stroke_xml_returns_correct_stroke_count():
    strokes = parse_stroke_xml(LINESTROKES_DIR / "a01-000u-01.xml")
    assert len(strokes) == 4, "Expected 4 strokes in the synthetic line 01 file"


def test_parse_stroke_xml_points_are_float_triples():
    strokes = parse_stroke_xml(LINESTROKES_DIR / "a01-000u-01.xml")
    for stroke in strokes:
        assert len(stroke) > 0
        for point in stroke:
            assert len(point) == 3, "Each point must be an (x, y, t) triple"
            x, y, t = point
            assert isinstance(x, float)
            assert isinstance(y, float)
            assert isinstance(t, float)


def test_parse_stroke_xml_no_nans():
    import math
    strokes = parse_stroke_xml(LINESTROKES_DIR / "a01-000u-01.xml")
    for stroke in strokes:
        for x, y, t in stroke:
            assert not math.isnan(x)
            assert not math.isnan(y)
            assert not math.isnan(t)


def test_parse_stroke_xml_timestamps_monotonic_within_stroke():
    """Timestamps should never go backwards within a single stroke —
    a violation here would indicate a data acquisition or parsing bug."""
    strokes = parse_stroke_xml(LINESTROKES_DIR / "a01-000u-01.xml")
    for stroke in strokes:
        times = [t for (_, _, t) in stroke]
        assert times == sorted(times), "Timestamps must be non-decreasing"


def test_parse_stroke_xml_raises_on_missing_strokeset(tmp_path):
    """A file with no <StrokeSet> at all should fail loudly, not silently
    return an empty list."""
    bad_xml = tmp_path / "bad.xml"
    bad_xml.write_text(
        '<?xml version="1.0"?><WhiteboardCaptureSession>'
        '<WhiteboardDescription/></WhiteboardCaptureSession>'
    )
    with pytest.raises(ValueError, match="No <StrokeSet>"):
        parse_stroke_xml(bad_xml)


def test_parse_stroke_xml_raises_on_empty_stroke(tmp_path):
    """A <Stroke> with zero <Point> children should be rejected, not
    silently treated as a valid stroke."""
    bad_xml = tmp_path / "empty_stroke.xml"
    bad_xml.write_text(
        '<?xml version="1.0"?><WhiteboardCaptureSession>'
        '<StrokeSet><Stroke colour="black" start_time="0" end_time="0">'
        '</Stroke></StrokeSet></WhiteboardCaptureSession>'
    )
    with pytest.raises(ValueError, match="Empty <Stroke>"):
        parse_stroke_xml(bad_xml)


def test_parse_stroke_xml_raises_on_malformed_point(tmp_path):
    """A Point missing a required attribute (e.g. y) should fail with a
    clear error rather than crashing deep inside downstream code."""
    bad_xml = tmp_path / "malformed_point.xml"
    bad_xml.write_text(
        '<?xml version="1.0"?><WhiteboardCaptureSession>'
        '<StrokeSet><Stroke colour="black" start_time="0" end_time="1">'
        '<Point x="100" time="0.01"/>'  # missing y
        '</Stroke></StrokeSet></WhiteboardCaptureSession>'
    )
    with pytest.raises(ValueError, match="Malformed Point"):
        parse_stroke_xml(bad_xml)


def test_parse_stroke_xml_raises_on_no_strokes(tmp_path):
    bad_xml = tmp_path / "no_strokes.xml"
    bad_xml.write_text(
        '<?xml version="1.0"?><WhiteboardCaptureSession>'
        '<StrokeSet></StrokeSet></WhiteboardCaptureSession>'
    )
    with pytest.raises(ValueError, match="No strokes found"):
        parse_stroke_xml(bad_xml)


# ---------------------------------------------------------------------------
# parse_ascii_transcription
# ---------------------------------------------------------------------------

def test_parse_ascii_transcription_correct_line_count():
    transcriptions = parse_ascii_transcription(ASCII_DIR / "a01-000u.txt")
    assert len(transcriptions) == 3


def test_parse_ascii_transcription_text_non_empty():
    transcriptions = parse_ascii_transcription(ASCII_DIR / "a01-000u.txt")
    for line_id, text in transcriptions.items():
        assert text.strip() != "", f"Line {line_id} has empty transcription"


def test_parse_ascii_transcription_expected_content():
    transcriptions = parse_ascii_transcription(ASCII_DIR / "a01-000u.txt")
    assert transcriptions["01"] == "the quick brown fox"
    assert transcriptions["02"] == "jumps over the lazy dog"
    assert transcriptions["03"] == "handwriting recognition is fun"


def test_parse_ascii_transcription_raises_without_csr_marker(tmp_path):
    bad_ascii = tmp_path / "bad.txt"
    bad_ascii.write_text("a01-000u\n\nsome text without the marker\n")
    with pytest.raises(ValueError, match="No 'CSR:' marker"):
        parse_ascii_transcription(bad_ascii)


# ---------------------------------------------------------------------------
# load_form (end-to-end: strokes + transcription matched together)
# ---------------------------------------------------------------------------

def test_load_form_returns_all_lines():
    parsed_lines = load_form("a01-000u", LINESTROKES_DIR, ASCII_DIR)
    assert len(parsed_lines) == 3


def test_load_form_lines_match_correct_text():
    parsed_lines = load_form("a01-000u", LINESTROKES_DIR, ASCII_DIR)
    by_line_id = {pl.line_id: pl for pl in parsed_lines}
    assert by_line_id["01"].text == "the quick brown fox"
    assert by_line_id["02"].text == "jumps over the lazy dog"
    assert by_line_id["03"].text == "handwriting recognition is fun"


def test_load_form_every_line_has_strokes_and_points():
    parsed_lines = load_form("a01-000u", LINESTROKES_DIR, ASCII_DIR)
    for pl in parsed_lines:
        assert pl.num_strokes > 0
        assert pl.num_points > 0


def test_load_form_raises_on_missing_ascii_file():
    with pytest.raises(FileNotFoundError, match="No ascii transcription"):
        load_form("nonexistent-form", LINESTROKES_DIR, ASCII_DIR)


def test_load_form_raises_on_no_matching_xml_files(tmp_path):
    # ascii file exists (reuse real one via a temp copy) but no XML lines
    ascii_copy_dir = tmp_path / "ascii"
    ascii_copy_dir.mkdir()
    (ascii_copy_dir / "ghost-form.txt").write_text(
        "ghost-form\n\nCSR:\n\nsome text\n"
    )
    empty_linestrokes_dir = tmp_path / "lineStrokes"
    empty_linestrokes_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No line XML files found"):
        load_form("ghost-form", empty_linestrokes_dir, ascii_copy_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
