# WELFake — Fake-News Detector (FastAPI + Agentic AI)

A web app and JSON API around the LSTM fake-news classifier trained in
`Compute.ipynb` (WELFake dataset, exported to `models/model.onnx`).

It wraps the model in a **two-agent pipeline** and a Tailwind UI:

```
title + text  →  Translator agent → preprocess  →  ONNX LSTM
              →  classify  →  Verification agent (only if FAKE)  →  result
```

* **Translator agent** — auto-detects non-English input (e.g. Indonesian) and
  translates it to English before classifying (handles long articles via chunking).
* **Verification agent** — when a story is flagged **FAKE**, it calls a web search
  (DuckDuckGo, no API key) to surface what reputable sources actually report.
  If `GEMINI_API_KEY` is set, it also adds an LLM-written summary.

This targets the assignment's **80+ "Model integration using Agentic AI"** tier:
a model exposed as an API, with agents that pass results to one another.

---

## Quick start

> Requirements already met on this machine: Python **3.14**, `.venv` created,
> dependencies installed, `models/model.onnx` + `models/tokenizer_word_index.json`
> in place. To reproduce from scratch:

```bash
# 1. Create the virtual environment
py -3.14 -m venv .venv

# 2. Activate it — pick the line for YOUR shell:
.venv\Scripts\activate            # Windows CMD
.venv\Scripts\Activate.ps1        # Windows PowerShell
source .venv/Scripts/activate     # Git Bash / MINGW64  (note: forward slashes + `source`)

# 3. Install dependencies (no TensorFlow needed)
pip install -r requirements.txt

# 4. Run the app
uvicorn app.main:app --port 8000 --reload
```

> **Already set up on this machine** — the `.venv`, dependencies, and
> `models/tokenizer_word_index.json` exist, so you can skip steps 1 and 3 and just
> activate + run. If you're in **Git Bash** and don't want to activate, run directly:
> `.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000 --reload`

Then open **http://localhost:8000** (interactive API docs at `/docs`).

---

## API

| Method | Path | Body | Description |
|--------|------|------|-------------|
| `GET`  | `/` | — | Tailwind web UI |
| `GET`  | `/health` | — | `{model_loaded, tokenizer_loaded, vocab_size}` |
| `POST` | `/api/predict` | `{title, text, translate?, verify?}` | Single prediction |
| `POST` | `/api/predict/batch?translate=&verify=` | `[{title, text}, ...]` | Batch (mirrors the notebook's JSON cell) |

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"title":"Breaking discovery","text":"New discovery found in the deep ocean.","verify":false}'
```

Response:

```json
{
  "label": "FAKE",
  "probability": 0.9655,
  "confidence": 0.9655,
  "input_was_translated": false,
  "detected_language": null,
  "translated_text": null,
  "verification": null
}
```

**Decision rule** (matches the notebook's validated logic): `probability ≥ 0.5 → FAKE`,
else `REAL`; `confidence` is the distance from 0.5. Configurable in `app/config.py`.

---

## The tokenizer (important)

`model.onnx` needs the exact Keras `Tokenizer` (`num_words=25000`, `maxlen=500`) that it was trained with. The tokenizer is exported directly by the notebook as `models/tokenizer_word_index.json` and loaded at startup.

---

## Project layout

```
app/
  main.py            FastAPI app, routes, startup model/tokenizer load
  config.py          hyperparameters (must match the notebook)
  preprocessing.py   pure-Python Keras tokenize + pad (train/serve parity)
  inference.py       ONNX Runtime session + classify()
  schemas.py         Pydantic request/response models
  agents/
    translator.py    Agent 1 — deep-translator auto→English
    verifier.py      Agent 2 — DuckDuckGo search (+ optional Gemini LLM)
web/
  templates/index.html   Tailwind single-page UI
  static/app.js          fetch + rendering
tests/test_preprocessing.py  Keras-parity + notebook-prediction tests
```

Run tests with `python -m pytest tests/ -q`.

## Notes
* **Python 3.14**: the stack is intentionally TensorFlow-free (`onnxruntime` only),
  which installs cleanly on 3.14. The UI is served via `FileResponse` (not Jinja2,
  which currently has a 3.14 cache bug).
* Optional LLM verification: copy `.env.example` to `.env` and set `GEMINI_API_KEY`.
  Without it, the verifier still returns search sources.
