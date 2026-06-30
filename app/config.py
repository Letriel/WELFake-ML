"""Central configuration. Hyperparameters MUST match Compute (1).ipynb."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
WEB_DIR = BASE_DIR / "web"

MODEL_PATH = MODELS_DIR / "model.onnx"
WORD_INDEX_PATH = MODELS_DIR / "tokenizer_word_index.json"

# --- Keras Tokenizer / training hyperparameters (from the notebook) ---
NUM_WORDS = 25000          # max_features
MAXLEN = 500               # maxlen
OOV_TOKEN = None           # the notebook used no oov_token
# Keras default Tokenizer filters (every char here is replaced with a space):
PAD_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

# --- Decision rule (matches the notebook's *validated* predict logic) ---
#   prob >= THRESHOLD -> FAKE, else REAL ; confidence = distance from 0.5
THRESHOLD = 0.5
FAKE_IS_HIGH = True        # flip if the team confirms the opposite orientation

# --- Verification agent ---
VERIFY_ON_LABEL = "FAKE"   # run the verifier only when prediction == this label
VERIFY_MAX_RESULTS = 5
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash")

