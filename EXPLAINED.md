# Lightweight Embedding Service

A lightweight HTTP embedding service using **FastAPI + ONNX Runtime + all-MiniLM-L6-v2**.

This service is designed to provide text embeddings without loading PyTorch or the full `sentence-transformers` stack into memory.

## Why ONNX?

The original `sentence-transformers` setup was too memory-intensive for a 512 MB Render instance.

The approximate memory usage observed with the PyTorch/Sentence Transformers setup was:

```text
Python startup       ~13 MB
sentence-transformers import  ~495 MB
Model loaded         ~517 MB
After inference      ~575 MB
```

Because of this, the service uses:

```text
FastAPI
   ↓
ONNX Runtime
   ↓
all-MiniLM-L6-v2
```

instead of:

```text
FastAPI
   ↓
Sentence Transformers
   ↓
Transformers
   ↓
PyTorch
   ↓
all-MiniLM-L6-v2
```

The ONNX approach removes the large PyTorch dependency from the runtime.

## Project Structure

```text
embedding-service/
│
├── main.py
├── model.onnx
├── tokenizer.json
├── requirements.txt
└── README.md
```

### Files

`main.py`

The FastAPI application and embedding logic.

`model.onnx`

The ONNX version of `all-MiniLM-L6-v2`.

Only one ONNX model file is required.

`tokenizer.json`

The tokenizer used to convert text into model input tokens.

`requirements.txt`

Contains only the runtime dependencies required by the service.

## Requirements

The service uses:

```text
fastapi
uvicorn
onnxruntime
tokenizers
numpy
```

Install them with:

```bash
pip install fastapi uvicorn onnxruntime tokenizers numpy
```

Or:

```bash
pip install -r requirements.txt
```

## Model

The service uses:

```text
all-MiniLM-L6-v2
```

The model produces:

```text
384-dimensional embeddings
```

The service uses a maximum sequence length of:

```text
256 tokens
```

## Downloading the Model

You need only:

```text
model.onnx
tokenizer.json
```

Place both files in the project root:

```text
embedding-service/
├── main.py
├── model.onnx
├── tokenizer.json
└── requirements.txt
```

Do not download multiple ONNX model files for the initial setup.

Later, if memory optimization is necessary, `model.onnx` can be replaced with an appropriate quantized ONNX model.

## Running Locally

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Start the service:

```bash
python main.py
```

The service runs on:

```text
http://127.0.0.1:8000
```

## Health Check

Open:

```text
http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

## Generate Embeddings

Send a POST request to:

```text
POST /embed
```

Request:

```json
{
  "texts": [
    "What is your refund policy?",
    "How can I contact support?"
  ]
}
```

Example using `curl`:

```bash
curl -X POST http://127.0.0.1:8000/embed \
  -H "Content-Type: application/json" \
  -d "{\"texts\":[\"What is your refund policy?\",\"How can I contact support?\"]}"
```

The response looks like:

```json
{
  "embeddings": [
    [
      0.0123,
      -0.0456,
      "... 384 values ..."
    ],
    [
      0.0234,
      0.0012,
      "... 384 values ..."
    ]
  ]
}
```

Each input text produces one 384-dimensional embedding.

## How Embedding Works

The request goes through the following process:

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Attention Mask
 ↓
ONNX Model
 ↓
Token Embeddings
 ↓
Mean Pooling
 ↓
L2 Normalization
 ↓
384-dimensional Embedding
```

Mean pooling combines the token-level embeddings into a single sentence embedding.

The final embeddings are normalized so they can be used effectively with cosine similarity.

## Memory Monitoring

Because this service is intended for a low-memory Render instance, `main.py` includes memory measurements.

It reports:

```text
RAM at startup
RAM after tokenizer
RAM after ONNX model
RAM after inference
```

For example:

```text
RAM at startup: 35.20 MB
RAM after tokenizer: 38.50 MB
RAM after ONNX model: 180.70 MB
RAM after inference: 210.40 MB
```

The most important measurements are:

```text
RAM after ONNX model
RAM after inference
```

The Render service has a 512 MiB memory limit, so these values should remain comfortably below that limit.

Memory usage should be measured locally before deployment.

## Thread Configuration

The ONNX Runtime session is configured with:

```python
session_options.intra_op_num_threads = 1
session_options.inter_op_num_threads = 1
```

This keeps CPU and memory usage more predictable on a small Render instance.

The service should also run with only one worker.

Do not start it with:

```bash
--workers 2
```

because each worker can load its own copy of the model.

## Render Deployment

For Render, use:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Do not use multiple workers.

Render provides the port through the `$PORT` environment variable.

The application also supports the same behavior when started directly with:

```bash
python main.py
```

## Environment Variables

The model and tokenizer paths can optionally be configured with:

```text
MODEL_PATH
TOKENIZER_PATH
```

By default:

```text
MODEL_PATH=model.onnx
TOKENIZER_PATH=tokenizer.json
```

For example:

```bash
MODEL_PATH=model.onnx
TOKENIZER_PATH=tokenizer.json
```

Normally, these variables do not need to be changed.

## Connecting the Chatbot Backend

The main chatbot backend should no longer install or import:

```text
sentence-transformers
torch
transformers
```

Instead, the chatbot sends text to this embedding service over HTTP.

For example:

```text
Chatbot Backend
      │
      │ POST /embed
      │
      ▼
Embedding Service
      │
      ▼
ONNX Runtime
      │
      ▼
all-MiniLM-L6-v2
      │
      ▼
384-dimensional embeddings
      │
      ▼
Chatbot Backend
```

This keeps the embedding model isolated from the chatbot process.

## Example Client Request

From the chatbot backend:

```python
import requests


response = requests.post(
    "https://YOUR-EMBEDDING-SERVICE.onrender.com/embed",
    json={
        "texts": [
            "What is your refund policy?",
            "How can I contact support?"
        ]
    },
    timeout=60
)

response.raise_for_status()

embeddings = response.json()["embeddings"]
```

The chatbot can then use those vectors for semantic search, retrieval, or vector database operations.
