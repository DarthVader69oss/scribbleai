"""
Generates a synthetic file matching the REAL IAM-OnDB lineStrokes XML schema,
confirmed against the official data format page:
https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database/data-format

Real schema:
    WhiteboardCaptureSession
      WhiteboardDescription
        SensorLocation (corner attr)
        DiagonallyOppositeCoords (x, y attrs)
        VerticallyOppositeCoords (x, y attrs)
        HorizontallyOppositeCoords (x, y attrs)
      StrokeSet
        Stroke (colour, start_time, end_time attrs)
          Point (x, y, time attrs) -- one or more per stroke

The real dataset also ships plain-text ascii transcriptions in a separate
directory, matched by form/line ID. We generate one of those too, so the
parser can be tested end-to-end: strokes in, (strokes, text) pairs out.

Naming convention matches the real dataset: <form_id>-<line_number>.xml
e.g. a01-000u-01.xml is line 1 of form a01-000u.
"""

import random
import math

random.seed(42)  # reproducible synthetic data


def generate_stroke_points(start_x, start_y, num_points, start_time, dt=0.01):
    """Generate a single stroke's points as a smooth-ish wandering path,
    loosely simulating a pen tracing a letter shape."""
    points = []
    x, y = start_x, start_y
    t = start_time
    for i in range(num_points):
        x += random.uniform(-15, 25)
        y += random.uniform(-10, 10)
        t += dt
        points.append((round(x, 1), round(y, 1), round(t, 3)))
    return points


def build_line_xml(strokes):
    """strokes: list of list-of-(x,y,t) tuples. Returns the full XML string."""
    lines = []
    lines.append('<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>')
    lines.append('<WhiteboardCaptureSession>')
    lines.append('  <WhiteboardDescription>')
    lines.append('    <SensorLocation corner="top_left"/>')
    lines.append('    <DiagonallyOppositeCoords x="8869" y="5937"/>')
    lines.append('    <VerticallyOppositeCoords x="90" y="5937"/>')
    lines.append('    <HorizontallyOppositeCoords x="8869" y="112"/>')
    lines.append('  </WhiteboardDescription>')
    lines.append('  <StrokeSet>')
    for stroke in strokes:
        start_t = stroke[0][2]
        end_t = stroke[-1][2]
        lines.append(f'    <Stroke colour="black" start_time="{start_t}" end_time="{end_t}">')
        for (x, y, t) in stroke:
            lines.append(f'      <Point x="{x}" y="{y}" time="{t}"/>')
        lines.append('    </Stroke>')
    lines.append('  </StrokeSet>')
    lines.append('</WhiteboardCaptureSession>')
    return "\n".join(lines)


def main():
    out_dir = "/home/claude/scribbleai/data/raw/synthetic/lineStrokes"
    ascii_dir = "/home/claude/scribbleai/data/raw/synthetic/ascii"

    # Simulate a form "a01-000u" with 3 lines, matching real IAM-OnDB naming.
    form_id = "a01-000u"
    line_texts = {
        "01": "the quick brown fox",
        "02": "jumps over the lazy dog",
        "03": "handwriting recognition is fun",
    }

    t = 0.0
    for line_num, text in line_texts.items():
        # crude simulation: one stroke per word, few points per stroke
        strokes = []
        x_cursor = 500.0
        y_base = 2000.0 + int(line_num) * 400  # lines stack vertically
        for word in text.split():
            num_points = random.randint(15, 30)
            stroke = generate_stroke_points(x_cursor, y_base, num_points, t)
            strokes.append(stroke)
            x_cursor = stroke[-1][0] + 60  # gap before next word
            t = stroke[-1][2] + 0.15  # pen-up pause between strokes

        xml_content = build_line_xml(strokes)
        out_path = f"{out_dir}/{form_id}-{line_num}.xml"
        with open(out_path, "w") as f:
            f.write(xml_content)
        print(f"Wrote {out_path} ({len(strokes)} strokes, "
              f"{sum(len(s) for s in strokes)} total points)")

    # Write the matching ascii transcription file, mimicking the real
    # dataset's "CSR:" section format used in the ascii/ directory.
    ascii_path = f"{ascii_dir}/{form_id}.txt"
    with open(ascii_path, "w") as f:
        f.write(f"{form_id}\n\n")
        f.write("CSR:\n\n")
        for line_num in sorted(line_texts.keys()):
            f.write(line_texts[line_num] + "\n")
    print(f"Wrote {ascii_path}")


if __name__ == "__main__":
    main()
