from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import httpx

# .env 파일 불러오기
load_dotenv()

app = FastAPI()

# 환경변수에서 vLLM 서버 URL 불러오기
VLLM_SERVER_URL = os.getenv("VLLM_SERVER_URL")

# 요청 모델
class SummarizeRequest(BaseModel):
    post_id: int
    context: str

# 응답 모델
class CommonResponse(BaseModel):
    status: int
    message: str
    data: str


@app.get("/api/summarize-service/health-check")
async def health_check():
    return JSONResponse(
        status_code=200,
        content={
            "status": 200,
            "message": "서버 상태 확인",
            "data": "Working"
        }
    )

@app.post("/api/summarize-service/summarize", response_model=CommonResponse)
async def summarizeText(request: SummarizeRequest):
    try:
        # messages 포맷
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a summarizing expert. Summarize the following text in Korean.\n"
                    "Important: Output must be only the Korean summary text itself. No explanation, no label, no preamble, no English text.\n"
                    "Conditions:\n"
                    "1. The summary must be complete and end with a full sentence.\n"
                    "2. Keep it within 500 characters.\n"
                    "3. Avoid repeated words or redundant expressions.\n"
                    "4. Only include the key information.\n"
                    "Example:\n"
                    "Input: \"이것은 샘플 텍스트입니다.\"\n"
                    "Output: \"샘플 요약입니다.\"\n"
                    "Now summarize this text:\n"
                )
            },
            {
                "role": "user",
                "content": request.context
            }
        ]

        payload = {
            "model": "google/gemma-3-4b-it",
            "messages": messages,
            "temperature": 0.3
        }

        # ✅ 비동기 HTTP 요청으로 변경
        async with httpx.AsyncClient() as client:
            response = await client.post(VLLM_SERVER_URL, json=payload)

        if response.status_code != 200:
            return CommonResponse(
                status=500,
                message=f"vLLM 서버 오류: {response.text}",
                data=""
            )

        result = response.json()
        summaryText = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        if not summaryText:
            return CommonResponse(
                status=500,
                message="vLLM 응답에서 content를 찾을 수 없습니다.",
                data=""
            )

        return CommonResponse(
            status=200,
            message="요청에 성공하였습니다",
            data=summaryText
        )

    except Exception as e:
        return CommonResponse(                    # ← ③
            status=500,
            message=str(e),
            data=""
        )

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],   # FE 주소(여러 개면 리스트로 추가로 넣어주도록 하자.)
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],  # 또는 ["*"], get은 필요해서 추가함.
    allow_headers=["*"],                       # Authorization 포함
)



#cd Desktop/vLLM
#uvicorn SummarizeServer:app --host 0.0.0.0 --port 8500 --reload