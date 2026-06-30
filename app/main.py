"""FastAPI service: Tailwind UI + JSON API for the WELFake fake-news detector.

Request pipeline:  title+text  ->  (translator agent)  ->  preprocess  ->  ONNX
                   ->  classify  ->  (verifier agent, if flagged FAKE)  ->  response
"""
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .agents.translator import translate_to_english
from .agents.verifier import verify_claim
from .config import MODEL_PATH, VERIFY_ON_LABEL, WEB_DIR, WORD_INDEX_PATH
from .inference import FakeNewsModel
from .preprocessing import Tokenizer, preprocess
from .schemas import (
    BatchResponse,
    NewsItem,
    PredictRequest,
    PredictResponse,
    Verification,
    VerifyRequest,
)

logger = logging.getLogger("welfake")
logging.basicConfig(level=logging.INFO)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ONNX model once.
    if MODEL_PATH.exists():
        try:
            state["model"] = FakeNewsModel(MODEL_PATH)
            logger.info("Loaded ONNX model: %s", MODEL_PATH)
        except Exception as e:
            state["model"] = None
            logger.error("Failed to load model: %s", e)
    else:
        state["model"] = None
        logger.warning("Model file missing: %s", MODEL_PATH)

    # Load the tokenizer word_index once.
    if WORD_INDEX_PATH.exists():
        state["tokenizer"] = Tokenizer.from_json_file(WORD_INDEX_PATH)
        logger.info(
            "Loaded tokenizer (%d words): %s",
            len(state["tokenizer"].word_index),
            WORD_INDEX_PATH,
        )
    else:
        state["tokenizer"] = None
        logger.warning("Tokenizer file missing: %s", WORD_INDEX_PATH)

    yield
    state.clear()


app = FastAPI(title="WELFake Fake-News Detector", version="1.0", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")
INDEX_HTML = WEB_DIR / "templates" / "index.html"


def _ensure_ready():
    if state.get("model") is None:
        raise HTTPException(503, "Model not loaded. Place model.onnx in models/.")
    if state.get("tokenizer") is None:
        raise HTTPException(
            503,
            "Tokenizer not loaded. Place tokenizer_word_index.json in models/ "
            "(see README).",
        )


def _run_pipeline(title: str, text: str, translate: bool, verify: bool) -> PredictResponse:
    model: FakeNewsModel = state["model"]
    tokenizer: Tokenizer = state["tokenizer"]

    combined = f"{title or ''} {text or ''}".strip()
    used_text, was_translated, detected = combined, False, None
    if translate:
        used_text, was_translated, detected = translate_to_english(combined)

    padded = preprocess(used_text or combined, tokenizer)
    prob = model.predict_proba(padded)[0]
    label, confidence, prob = model.classify(prob)

    verification = None
    if verify and label == VERIFY_ON_LABEL:
        verification = verify_claim(used_text or combined, original_claim=combined)

    return PredictResponse(
        label=label,
        probability=prob,
        confidence=confidence,
        input_was_translated=was_translated,
        detected_language=detected,
        translated_text=used_text if was_translated else None,
        verification=verification,
    )


@app.get("/")
async def index():
    return FileResponse(INDEX_HTML)


@app.get("/health")
async def health():
    tok = state.get("tokenizer")
    return {
        "status": "ok",
        "model_loaded": state.get("model") is not None,
        "tokenizer_loaded": tok is not None,
        "vocab_size": len(tok.word_index) if tok else 0,
    }


@app.post("/api/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    _ensure_ready()
    return _run_pipeline(req.title, req.text, req.translate, req.verify)


@app.post("/api/verify", response_model=Verification)
async def verify(req: VerifyRequest):
    _ensure_ready()
    combined = f"{req.title or ''} {req.text or ''}".strip()
    used_text = combined
    if req.translate:
        used_text, _, _ = translate_to_english(combined)
    return verify_claim(used_text or combined, original_claim=combined)


@app.post("/api/predict/batch", response_model=BatchResponse)
async def predict_batch(
    items: List[NewsItem], translate: bool = True, verify: bool = False
):
    """Accepts a raw JSON array of {title, text} -- mirrors the notebook's batch cell."""
    _ensure_ready()
    results = [_run_pipeline(it.title, it.text, translate, verify) for it in items]
    return BatchResponse(results=results)
