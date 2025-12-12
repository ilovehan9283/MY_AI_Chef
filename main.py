import os
import sys
import subprocess
import json


# [가이드라인: 라이브러리 자동 설치]
def install_package(package):
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package])
    except Exception as e:
        print(f"❌ {package} 설치 실패: {e}")


required_packages = [
    "fastapi", "uvicorn", "google-generativeai", "pydantic", "python-multipart"
]

for package in required_packages:
    try:
        import_name = "google.generativeai" if package == "google-generativeai" else package
        if package == "python-multipart": import_name = "multipart"
        __import__(import_name.split('.')[0])
    except ImportError:
        print(f"📦 '{package}' 라이브러리가 없습니다. 자동 설치를 시작합니다...")
        install_package(package)

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import google.generativeai as genai
from google.generativeai import types

app = FastAPI()

# [가이드라인: CORS 설정]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 데이터 모델 (api_key 필드 필수)
class RecipeRequest(BaseModel):
    ingredients: str
    is_creative_mode: bool = False
    allow_seasoning: bool = True
    api_key: str  # 프론트엔드에서 API Key 수신


@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: templates/index.html 파일을 찾을 수 없습니다.</h1>"


@app.post("/register")
async def register_data(request: Request):
    data = await request.json()
    print(f"📥 [데이터 수신 로그]: {data}")
    return {"message": "등록 성공", "received_data": data}


@app.post("/generate")
async def generate_recipe(req: RecipeRequest):
    api_key = req.api_key

    if not api_key:
        return JSONResponse(status_code=401,
                            content={"error": "API Key가 필요합니다."})

    # [모델 우선순위 수정] gemini-2.5-flash를 최우선으로 설정하여 404 및 429 오류 회피
    # 2.0-flash-exp 대신 2.5-flash 사용 (더 안정적)
    model_priority = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-pro"]

    # 1. API Key 설정 및 첫 모델 선택
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        return JSONResponse(status_code=401,
                            content={"error": f"API Key 설정 실패: {str(e)}"})

    request_text = "창의적이고 특별한 요리 추천" if req.is_creative_mode else "대중적이고 실패 없는 정석 요리 추천"
    condition = "기본 조미료 사용 가능" if req.allow_seasoning else "오직 재료와 소금/후추만 사용 (엄격 모드)"

    system_instruction = f"""
    당신은 전문 요리사입니다.
    [입력 재료]: {req.ingredients}
    [요청 스타일]: {request_text}
    [제약 조건]: {condition}
    반드시 아래의 JSON 형식으로만 응답하세요. (마크다운 코드블럭 없이 순수 JSON만)
    """

    json_schema = {
        "type": "OBJECT",
        "properties": {
            "recommendations": {
                "type": "ARRAY",
                "items": {
                    "type":
                    "OBJECT",
                    "properties": {
                        "id": {
                            "type": "INTEGER"
                        },
                        "dish_name": {
                            "type": "STRING"
                        },
                        "dish_name_en": {
                            "type": "STRING"
                        },
                        "style": {
                            "type": "STRING"
                        },
                        "difficulty": {
                            "type": "STRING"
                        },
                        "calories": {
                            "type": "STRING"
                        },
                        "reasoning": {
                            "type": "STRING"
                        },
                        "recipe_steps": {
                            "type": "ARRAY",
                            "items": {
                                "type": "STRING"
                            }
                        }
                    },
                    "required": [
                        "id", "dish_name", "dish_name_en", "style",
                        "difficulty", "calories", "reasoning", "recipe_steps"
                    ]
                }
            }
        }
    }

    # 2. 우선순위대로 모델을 순회하며 시도
    last_error = None
    for model_id in model_priority:
        try:
            print(f"🔄 /generate 시도 모델: {model_id}")
            model = genai.GenerativeModel(model_name=model_id)

            response = model.generate_content(
                f"{system_instruction}\n\n재료: {req.ingredients}. 레시피 3가지를 추천해줘.",
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=json_schema))
            return json.loads(response.text)  # 성공 시 즉시 반환

        except Exception as e:
            error_msg = str(e)
            last_error = error_msg
            print(f"❌ 모델 {model_id} 실패: {error_msg}")

            # 403 Forbidden, 404 Not Found, 429 Quota Exceeded 발생 시 다음 모델로 넘어감
            if "403" in error_msg or "429" in error_msg or "404" in error_msg:
                continue
            else:
                # 기타 치명적인 오류 (JSON 파싱 실패, 서버 내부 오류 등)
                return JSONResponse(status_code=500,
                                    content={"error": error_msg})

    # 3. 모든 모델 시도 실패 후 처리
    if "429" in last_error or "Quota" in last_error:
        return JSONResponse(
            status_code=503,
            content={"error": "현재 모든 AI 모델의 사용량이 초과되었습니다. 잠시 후 다시 시도해주세요."})
    elif "404" in last_error or "403" in last_error or "API key" in last_error:
        return JSONResponse(status_code=401,
                            content={"error": "API Key가 유효하지 않거나 할당량이 부족합니다."})
    else:
        return JSONResponse(status_code=500,
                            content={"error": f"AI 응답 처리 불가: {last_error}"})


@app.post("/ask")
async def ask_chef(request: Request):
    try:
        data = await request.json()
        api_key = data.get('api_key')

        if not api_key:
            return JSONResponse(status_code=401,
                                content={"error": "API Key 없음"})

        # 질문 기능은 2.5-flash로 고정하여 비용 최소화 및 안정성 확보
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        response = model.generate_content(
            f"요리 '{data.get('dish_name')}' 관련 질문: {data.get('question')}. 친절하고 짧게(3문장 이내) 답변해줘."
        )
        return {"answer": response.text.strip()}
    except Exception as e:
        if "403" in str(e) or "API key not valid" in str(e) or "404" in str(e):
            return JSONResponse(status_code=401,
                                content={"error": "유효하지 않은 API Key입니다."})
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
