# WELFake — Fake-News Detector (FastAPI + Agentic AI)

A complete web application and JSON API constructed around the LSTM fake-news classifier trained on the WELFake dataset (72k articles). The trained model has been exported to `models/model.onnx`.

This project implements a multi-agent system that bridges a deep learning model with external APIs to classify text and verify claims in real-time.

---

## 🛠️ Architecture & Pipeline

The system is designed with a **two-agent pipeline** and a Tailwind-styled single-page UI:

```
                  ┌───────────────────────┐
                  │      Input Text       │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │   Translator Agent    │ (Translate to English if needed)
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │ Preprocessing Layer   │ (Tokenize & Pad Sequence)
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │    ONNX LSTM Model    │ (Classify FAKE vs REAL)
                  └───────────┬───────────┘
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
        [REAL]                               [FAKE]
            │                                   │
      (Optional)                                │
┌───────────────────────┐             ┌─────────▼─────────────┐
│  Verification Agent   │             │  Verification Agent   │ (Search in original
│ (Triggered Manually)  │             │ (Triggered Auto/Man)  │  input language)
└───────────────────────┘             └───────────────────────┘
```

1. **Translator Agent** (`app/agents/translator.py`): Automatically detects input language. If non-English (e.g., Indonesian) and auto-translation is enabled, it translates the text into English. For long articles exceeding 4,500 characters, it automatically chunks the article before translating to bypass Google Translate's size limitations.
2. **Model Classification** (`app/inference.py`): Performs inference using `onnxruntime` on the processed tokens.
3. **Verification Agent** (`app/agents/verifier.py`): If a story is flagged **FAKE** (or manually requested by the user), it queries DuckDuckGo for reputable sources. If an LLM API key (like `GEMINI_API_KEY`) is provided, it generates an LLM-written summary comparing the claim with the search results.

---

## ✨ Features and Enhancements

* **Multilingual Web Search Querying**: The Verification Agent executes search queries in the **original language** of the input rather than its English translation. This improves retrieval effectiveness for local news stories.
* **Flexible Tokenizer Loading**: The tokenizer loader is updated to handle both raw token-to-index dictionary files (e.g., `tokenizer_word_index.json.henry`) and fully-exported Keras Tokenizer configurations (which contains config values and stringified JSON dictionaries). It handles double-serialized JSON strings seamlessly.
* **Interactive Manual Verification**: For cases where the model flags a story as `REAL` (or flags it as `FAKE` but auto-verification was disabled), the UI displays a call-to-action button allowing the user to run the verification agent manually:
  * **REAL**: Shows *"Having trust issue?"* followed by a **[Verify using Web Search]** button.
  * **FAKE** (with auto-verify disabled): Shows *"Want to verify?"* followed by a **[Verify using Web Search]** button.

---

## 📂 Project Layout & Core Files

### Backend Components

* **app/main.py**: The FastAPI server entry point. It sets up routes, manages startup configurations (loading the model and tokenizer), and orchestrates the inference/verification pipeline.
* **app/config.py**: Central configuration holding hyperparameters (vocab size, max sequence length, decision threshold) and API credentials.
* **app/preprocessing.py**: Pure-Python replication of the Keras text preprocessing pipeline (Tokenization, texts_to_sequences, and padding). Ensures exact byte-for-byte alignment with the training script without requiring a heavy TensorFlow dependency.
* **app/inference.py**: Handles ONNX Runtime execution (`CpuExecutionProvider`) and maps raw sigmoid output probabilities to classification labels and confidence ratings.
* **app/schemas.py**: Pydantic request/response structures enforcing strict validation for single/batch predictions and verification requests.
* **app/agents/translator.py)**: Translator Agent implementation using `deep-translator` (Google Translate backend) and language auto-detection via `langdetect`.
* **app/agents/verifier.py**: Verification Agent implementation utilizing `duckduckgo-search` (no API key required) and LLM summarization.

### Frontend Components

* **web/templates/index.html**: Simple, sleek Tailwind CSS Single-Page Application (SPA). Provides panels for both single prediction and batch predictions (JSON).
* **web/static/app.js**: Handles user interaction, form submissions, asynchronous API calls to `/api/predict` and `/api/verify`, and dynamically renders the prediction cards and verification results.

---

## ⚡ Quick Start

### Prerequisites
* Python **3.14** (or 3.10+)
* Model artifacts: `models/model.onnx` and `models/tokenizer_word_index.json` placed in their respective directory.

### Running the Server

If you are using the pre-configured virtual environment on this machine, activate it and start the development server:

```bash
# Activate virtual environment
source .venv/bin/activate

# Start the FastAPI server
uvicorn app.main:app --port 8000 --reload
```

Open [**http://localhost:8000**](http://localhost:8000) in your browser. You can access the interactive Swagger API documentation at [**http://localhost:8000/docs**](http://localhost:8000/docs).

### Running Tests

Unit tests are included to ensure preprocessing parity with Keras and check core classification outputs.

```bash
python -m pytest tests/ -q
```

---

## 📡 API Endpoints

### 1. Health Status
* **Method**: `GET`
* **Path**: `/health`
* **Response**:
  ```json
  {
    "status": "ok",
    "model_loaded": true,
    "tokenizer_loaded": true,
    "vocab_size": 25000
  }
  ```

### 2. Predict (Single)
* **Method**: `POST`
* **Path**: `/api/predict`
* **Request Body**:
  ```json
  {
    "title": "Breaking News Headline",
    "text": "Full article text...",
    "translate": true,
    "verify": true
  }
  ```
* **Response**:
  ```json
  {
    "label": "REAL",
    "probability": 0.034,
    "confidence": 0.966,
    "input_was_translated": false,
    "detected_language": "en",
    "translated_text": null,
    "verification": null
  }
  ```

### 3. Verify Claim (Manual)
* **Method**: `POST`
* **Path**: `/api/verify`
* **Request Body**:
  ```json
  {
    "title": "Headline",
    "text": "Text...",
    "translate": true
  }
  ```
* **Response**: Returns a `Verification` object containing:
  ```json
  {
    "checked": true,
    "method": "search",
    "summary": "Compare the claim against the sources below to assess its accuracy.",
    "sources": [
      {
        "title": "Reputable Source Article Title",
        "url": "https://example.com/article",
        "snippet": "Snippet containing relevant verification information..."
      }
    ]
  }
  ```

### 4. Batch Prediction
* **Method**: `POST`
* **Path**: `/api/predict/batch`
* **Query Parameters**: `translate` (bool), `verify` (bool)
* **Request Body**: An array of `NewsItem` objects:
  ```json
  [
    { "title": "Headline 1", "text": "Content..." },
    { "title": "Headline 2", "text": "Content..." }
  ]
  ```
