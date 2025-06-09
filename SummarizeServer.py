from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import httpx
from bs4 import BeautifulSoup
from prometheus_fastapi_instrumentator import Instrumentator
import asyncio
import logging


# .env 파일 불러오기
load_dotenv()

app = FastAPI()
instrumentator = Instrumentator().instrument(app).expose(app)

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

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text(separator="\n")
    return clean_text.strip()

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
                "content": strip_html_tags(request.context)
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
        return CommonResponse(
            status=500,
            message=str(e),
            data=""
        )


# 로거 설정 (optional)
logger = logging.getLogger("uvicorn.error")

# Dummy 요청용 메시지
dummy_messages = [
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
        "content": "이것은 더미 요청용 텍스트입니다. 서버가 정상 동작하는지 확인합니다."
    }
]

async def dummy_warmup_task():
    while True:
        try:
            logger.info("[Warmup] Dummy 요청 전송 시작")
            payload = {
                "model": "google/gemma-3-4b-it",
                "messages": dummy_messages,
                "temperature": 0.3
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(VLLM_SERVER_URL, json=payload)
            
            if response.status_code == 200:
                logger.info("[Warmup] Dummy 요청 성공 ✅")
            else:
                logger.warning(f"[Warmup] Dummy 요청 실패 ❌ - 상태코드: {response.status_code}, 응답: {response.text}")

        except Exception as e:
            logger.error(f"[Warmup] Dummy 요청 중 예외 발생 ❗ - {str(e)}")

        # 5분(300초)마다 실행 → 필요시 600초(10분)로 변경 가능
        await asyncio.sleep(300)

@app.on_event("startup")
async def startup_event():
    logger.info("서버 시작 시 warmup task 실행")
    asyncio.create_task(dummy_warmup_task())




#cd Desktop/vLLM
#uvicorn SummarizeServer:app --host 0.0.0.0 --port 8500 --reload