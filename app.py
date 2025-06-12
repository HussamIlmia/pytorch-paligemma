"""FastAPI wrapper around Google PaliGemma (3-B) for lightweight image-classification prompts.

Start with:
    pip install "uvicorn[standard]" fastapi transformers torch Pillow
    # optional: accelerate if you want device_map="auto" inference

Run with hot-reload for development:
    uvicorn app:app --host 0.0.0.0 --port 8080 --reload

The /classify endpoint takes an image file (multipart/form-data) and an optional
prompt string.  It returns a JSON payload like {"label": "red-bellied woodpecker"}.
"""

import os
from io import BytesIO
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)

# ────────────────────────────── Model initialisation ─────────────────────────────

MODEL_NAME: str = os.getenv("PALIGEMMA_MODEL", "google/paligemma-3b-pt-224")
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Prefer bfloat16 on Hopper/Ampere (SM80+) when available, else fall back.
try:
    _cap = torch.cuda.get_device_capability(0) if DEVICE == "cuda" else (0, 0)
    DTYPE = torch.bfloat16 if _cap[0] >= 8 else torch.float16 if DEVICE == "cuda" else torch.float32
except Exception:
    DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"Loading {MODEL_NAME} on {DEVICE} (dtype = {DTYPE}). This can take ~30 s the first time…")

processor: PaliGemmaProcessor = PaliGemmaProcessor.from_pretrained(MODEL_NAME) # type: ignore
model: PaliGemmaForConditionalGeneration = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype=DTYPE, device_map="auto" if DEVICE == "cuda" else None
) # type: ignore
model.to(DEVICE) # type: ignore
model.eval()

# ─────────────────────────────────── API ─────────────────────────────────────────

app = FastAPI(title="PaliGemma Classification API", version="0.1.0")


@app.get("/health")
async def health():
    """Kubernetes-friendly liveness probe."""
    return {"status": "ok"}


@app.post("/classify")
async def classify(
    image: Annotated[UploadFile, File(description="RGB image file (jpg/png)")],
    prompt: Annotated[str, Form()] = "What is in this image?",
):
    """Return the model’s best label (or free-form answer) for *prompt* about *image*."""

    try:
        raw = await image.read()
        img = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid image: {e}"})

    inputs = processor(images=[img], text=[prompt], return_tensors="pt").to(DEVICE, dtype=DTYPE)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            temperature=0.0,
        )

    answer: str = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
    return {"label": answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False)
