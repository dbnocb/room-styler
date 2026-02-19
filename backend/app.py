from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import base64
import io
from PIL import Image
from dotenv import load_dotenv
import os
from xai_sdk import Client

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Client(api_key=os.getenv("XAI_API_KEY"))

@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    history: str = Form("[]"),  # JSON str of [{"prompt": "...", "image_b64": "..."}]
    image_b64: str = Form(None),  # data:image/jpeg;base64,...
    ref_b64: str = Form(None),
):
    try:
        # Parse history
        history_list = eval(history) if history else []

        # Build full prompt with history
        full_prompt = prompt
        for h in history_list[-3:]:  # Last 3 for context
            full_prompt = f"Previous: {h['prompt']}. Current: {full_prompt}"

        if ref_b64:
            full_prompt += f". Use style/objects from reference image."

        params = {
            "model": "grok-imagine-image",
            "prompt": full_prompt,
            "aspect_ratio": "1:1",  # Mobile square
            "resolution": "1k",
        }

        if image_b64:
            # Extract base64 from data URL
            if image_b64.startswith("data:"):
                image_b64 = image_b64.split(",")[1]
            params["image_url"] = f"data:image/jpeg;base64,{image_b64}"

        if ref_b64:
            if ref_b64.startswith("data:"):
                ref_b64 = ref_b64.split(",")[1]
            # For ref, append to prompt (no multi-image yet)
            full_prompt += f" Reference: (imagine ref here)"

        response = client.image.sample(**params)

        # Return image URL + history update
        new_history = history_list + [{"prompt": prompt, "image_b64": response.image.split(",")[1] if response.image.startswith("data:") else ""}]

        return {
            "image_url": response.url,
            "image_b64": response.image,
            "history": new_history[-10:],  # Keep last 10
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
