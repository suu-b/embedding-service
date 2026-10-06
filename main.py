import os
import psutil

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from tokenizers import Tokenizer
import uvicorn

process = psutil.Process(os.getpid())

def memory_mb():
    return process.memory_info().rss / 1024 / 1024


print(f"RAM at startup: {memory_mb():.2f} MB")


MODEL_PATH = os.getenv("MODEL_PATH", "model.onnx")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "tokenizer.json")

# tokenizer
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

tokenizer.enable_truncation(max_length=256)

tokenizer.enable_padding(
    pad_id=0,
    pad_token="[PAD]"
)

print(f"RAM after tokenizer: {memory_mb():.2f} MB")


# onxx model
session_options = ort.SessionOptions()

session_options.intra_op_num_threads = 1
session_options.inter_op_num_threads = 1

session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=session_options,
    providers=["CPUExecutionProvider"],
)

print(f"RAM after ONNX model: {memory_mb():.2f} MB")


INPUT_NAMES = {
    item.name
    for item in session.get_inputs()
}



app = FastAPI()

class EmbedRequest(BaseModel):
    texts: list[str]


def create_embeddings(texts: list[str]):
    encodings = tokenizer.encode_batch(texts)
    input_ids = np.array(
        [encoding.ids for encoding in encodings],
        dtype=np.int64
    )
    attention_mask = np.array(
        [encoding.attention_mask for encoding in encodings],
        dtype=np.int64
    )
    token_type_ids = np.array(
        [encoding.type_ids for encoding in encodings],
        dtype=np.int64
    )
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if "token_type_ids" in INPUT_NAMES:
        inputs["token_type_ids"] = token_type_ids

    outputs = session.run(None, inputs)
    token_embeddings = outputs[0]
    mask = attention_mask[..., None].astype(np.float32)
    summed = np.sum(
        token_embeddings * mask,
        axis=1
    )
    counts = np.clip(
        mask.sum(axis=1),
        1e-9,
        None
    )

    embeddings = summed / counts
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )
    embeddings = embeddings / np.clip(
        norms,
        1e-12,
        None
    )
    print(f"RAM after inference: {memory_mb():.2f} MB")
    return embeddings.astype(np.float32).tolist()

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

@app.post("/embed")
def embed(request: EmbedRequest):
    embeddings = create_embeddings(
        request.texts
    )
    return {
        "embeddings": embeddings
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )