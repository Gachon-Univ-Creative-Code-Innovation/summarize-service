from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests
import json

# .env
load_dotenv()

app = FastAPI()

VLLM_SERVER_URL = os.getenv("VLLM_SERVER_URL")

PROMPT_PREFIX = "넌 요약전문가야. 다음 글을 요약해줘. ////////////"
PROMPT_SUFFIX = "\n\n//////////// 넌 요약전문가야. 앞선 글을 6줄 이하로 최대한 간결하게 요약해줘."

class TextRequest(BaseModel):
    text: str

class SummaryResponse(BaseModel):
    summary: str

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_text(request: TextRequest):
    try:
        full_prompt = PROMPT_PREFIX + request.text + PROMPT_SUFFIX

        payload = {
            "model": "google/gemma-3-1b-it",
            "prompt": full_prompt,
            "max_tokens": 300,
            "temperature": 0.3
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(VLLM_SERVER_URL, headers=headers, data=json.dumps(payload))

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"vLLM 서버 오류: {response.text}")

        result = response.json()
        summary_text = result["choices"][0]["text"].strip()

        return SummaryResponse(summary=summary_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
