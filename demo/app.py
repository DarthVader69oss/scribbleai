"""
demo/app.py -- ScribbleAI live recognition demo (Objective 8).

    python demo/app.py --ckpt checkpoints/best.pt                 # http://127.0.0.1:7860
    python demo/app.py --ckpt checkpoints/best.pt --host 0.0.0.0  # open from a tablet/phone on the same Wi-Fi
    python demo/app.py --ckpt checkpoints/best.pt --share         # + public Gradio link for the file tab

Two front ends over the SAME Recognizer (same preprocessing as training):
  /         live canvas: transcription updates after every stroke; switch decoders;
            "compare decoders" shows greedy / beam / beam+LM / +spell side by side with latency;
            shows the preprocessed, resampled trajectory the model actually sees.
  /gradio   Gradio app: upload a collector .jsonl export and score every sample (CER per line).
Runs on CPU; the model never needs a GPU at inference time.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.inference import Recognizer, MODES

STATIC = Path(__file__).resolve().parent / "static"
REC: Recognizer = None
CKPT = ""


class RecognizeRequest(BaseModel):
    strokes: list
    mode: str = "greedy"
    beam_width: int | None = None
    compare: bool = False


def create_app(rec, ckpt_name):
    global REC, CKPT
    REC, CKPT = rec, ckpt_name
    app = FastAPI(title="ScribbleAI demo")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/info")
    def info():
        return {"checkpoint": CKPT, "epoch": REC.epoch, "val_cer": REC.best_cer,
                "modes": REC.available_modes(), "vocab_size": REC.vocab.size,
                "params": REC.model.num_parameters()}

    @app.post("/api/recognize")
    def recognize(req: RecognizeRequest):
        strokes = [s for s in req.strokes if len(s) > 0]
        if not strokes:
            return {"text": "", "ms": {"total": 0}, "frames": 0, "ink": [], "all": {}}
        if req.mode not in REC.available_modes():
            raise HTTPException(400, f"mode {req.mode!r} not available: {REC.available_modes()}")
        try:
            out = REC.recognize(strokes, req.mode, req.beam_width)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if req.compare:
            from src.preprocessing.pipeline import preprocess
            import time
            F, _ = preprocess(strokes, REC.pcfg)
            lp = REC.log_probs_from_features(F)
            out["all"] = {}
            for m in REC.available_modes():
                t0 = time.perf_counter()
                txt = REC.decode(lp, m, req.beam_width)
                out["all"][m] = {"text": txt, "decode_ms": (time.perf_counter() - t0) * 1000}
        return out

    return app


def build_gradio(rec):
    import gradio as gr
    from src.preprocessing.parse_inhouse import parse_inhouse_file
    from src.evaluation.metrics import cer, corpus_scores

    def score_file(file, mode):
        if file is None:
            return [], "Upload a .jsonl file exported from demo/collector.html"
        path = file if isinstance(file, str) else file.name
        samples = parse_inhouse_file(path)
        rows, refs, hyps = [], [], []
        for s in samples:
            r = rec.recognize(s.strokes, mode)
            ref = rec.vocab.normalize_text(s.text)
            refs.append(ref); hyps.append(r["text"])
            rows.append([s.writer, ref, r["text"], round(100 * cer(ref, r["text"]), 1), round(r["ms"]["total"], 1)])
        sc = corpus_scores(refs, hyps)
        return rows, (f"**{len(samples)} samples** — CER {100 * sc['cer']:.1f}% · WER {100 * sc['wer']:.1f}% · "
                      f"exact lines {100 * sc['line_acc']:.0f}% (decoder: {mode})")

    with gr.Blocks(title="ScribbleAI — score collected samples") as demo:
        gr.Markdown("## ScribbleAI — score a collector export\nUpload a `.jsonl` file from "
                    "`demo/collector.html`; every sample is run through the same pipeline as the live demo.")
        with gr.Row():
            f = gr.File(label="collector .jsonl", file_types=[".jsonl"])
            mode = gr.Dropdown(rec.available_modes(), value="greedy", label="decoder")
        btn = gr.Button("Recognise all", variant="primary")
        summary = gr.Markdown()
        table = gr.Dataframe(headers=["writer", "reference", "prediction", "CER %", "ms"], wrap=True)
        btn.click(score_file, [f, mode], [table, summary])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "best.pt"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-gradio", action="store_true")
    ap.add_argument("--share", action="store_true", help="also launch the Gradio tab with a public link")
    ap.add_argument("--config", default=None,
                    help="take decoding settings (lm_alpha, lm_beta, default_beam_width, char_lm_file) from this "
                         "config instead of the checkpoint's copy, e.g. after tuning alpha/beta")
    a = ap.parse_args()
    if not Path(a.ckpt).exists():
        sys.exit(f"checkpoint not found: {a.ckpt}\nDownload best.pt from Drive into checkpoints/ or pass --ckpt.")
    overrides = None
    if a.config:
        from src.utils import load_config
        overrides = load_config(a.config)["decoding"]
    rec = Recognizer(a.ckpt, device="cpu", decoding_overrides=overrides)
    print(f"loaded {a.ckpt} (epoch {rec.epoch}, val CER {rec.best_cer}); decoders: {rec.available_modes()}")
    for m in rec.available_modes():          # warm-up so the first stroke in front of the audience is fast
        rec.recognize([[(0, 0), (10, 12), (20, 0)], [(25, 5), (35, 5)]], m)
    app = create_app(rec, Path(a.ckpt).name)
    if not a.no_gradio:
        import gradio as gr
        gdemo = build_gradio(rec)
        if a.share:
            gdemo.launch(share=True, prevent_thread_lock=True)
        app = gr.mount_gradio_app(app, gdemo, path="/gradio")
    print(f"open http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}/")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
