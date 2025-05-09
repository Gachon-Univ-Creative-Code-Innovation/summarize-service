from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests

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

@app.post("/summarize", response_model=CommonResponse)
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
            "temperature": 0.2
        }

        # JSON 자동 인코딩 및 Content-Type 헤더 자동 설정
        response = requests.post(VLLM_SERVER_URL, json=payload)

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"vLLM 서버 오류: {response.text}")

        result = response.json()

        # 응답 구조에서 content 가져오기 (KeyError 방지용 get() 사용)
        summaryText = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        if not summaryText:
            raise HTTPException(status_code=500, detail="vLLM 응답에서 content를 찾을 수 없습니다.")

        return CommonResponse(
            status=200,
            message="요청에 성공하였습니다",
            data=summaryText
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#cd Desktop/vLLM
#uvicorn SummarizeServer:app --host 0.0.0.0 --port 8500 --reload
