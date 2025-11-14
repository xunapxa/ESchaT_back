import os
from typing import Any, Dict, List, Optional, Tuple
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv

# LangChain community integrations
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant as LCQdrant
from langchain_core.documents import Document

# 엑셀 파싱 모듈
from excel_parser import read_excel_flex, parse_qa_data


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    matched_question: Optional[str] = None
    score: Optional[float] = None


APP_NAME = "eschaT-qa-backend"

# .env 자동 로드 (프로젝트 루트) - 환경 변수 읽기 전에 먼저 로드해야 함
load_dotenv(dotenv_path=os.getenv("ENV_FILE", ".env"))

# 환경 변수 (배포/로컬 모두에서 작동)
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant-production-2c5c.up.railway.app")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "qa_collection")
EXCEL_PATH = os.getenv("QA_EXCEL_PATH", "Q&A.xlsx")
# 요구사항 명시 모델 고정
EMBEDDING_MODEL = "kimseongsan/ko-sbert-384-reduced"

logging.basicConfig(level=logging.INFO, format="%(message)s")
DEBUG_WORKFLOW = os.getenv("DEBUG_WORKFLOW", "0") == "1"


def _w(msg: str) -> None:
    if DEBUG_WORKFLOW:
        print(f"[WORKFLOW] {msg}")

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_qdrant_client() -> QdrantClient:
    # 타임아웃 설정 (초 단위)
    timeout = 30.0
    if QDRANT_API_KEY:
        return QdrantClient(
            url=QDRANT_URL, 
            api_key=QDRANT_API_KEY,
            timeout=timeout
        )
    return QdrantClient(url=QDRANT_URL, timeout=timeout)


def ensure_collection(
    client: QdrantClient, collection_name: str, vector_size: int
) -> None:
    try:
        client.get_collection(collection_name=collection_name)
        return
    except UnexpectedResponse:
        pass

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def build_embeddings() -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)


def initialize_db_and_data(filepath: str) -> Tuple[LCQdrant, int]:
    _w("=== DB initialization started ===")
    _w(f"target_excel='{filepath}'")
    df = read_excel_flex(filepath)
    qa_rows = parse_qa_data(df)
    _w(f"parsed_pairs={len(qa_rows)}")

    if DEBUG_WORKFLOW and qa_rows:
        sample = qa_rows[0]
        _w(f"sample_pair -> Q: {sample['question']}, A: {sample['answer']}")

    docs = [
        Document(page_content=it["question"], metadata={"answer": it["answer"]})
        for it in qa_rows
    ]

    embeddings = build_embeddings()

    vs = LCQdrant.from_documents(
        documents=docs,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION,
    )
    _w(f"qdrant_collection='{QDRANT_COLLECTION}' upserted={len(docs)}")
    _w("=== DB initialization finished ===")

    return vs, len(docs)


def get_vectorstore() -> LCQdrant:
    embeddings = build_embeddings()
    vs = LCQdrant(
        client=get_qdrant_client(),
        collection_name=QDRANT_COLLECTION,
        embeddings=embeddings,
    )
    return vs


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "app": APP_NAME}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    question = (request.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="질문이 필요합니다.")

    _w(f"/chat received question='{question}'")

    try:
        vs = get_vectorstore()
        results_with_score = vs.similarity_search_with_score(question, k=1)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"검색 실패: {exc}") from exc

    if not results_with_score:
        _w("no result found for query")
        raise HTTPException(
            status_code=404,
            detail="죄송합니다. 제공된 데이터베이스에서 관련 답변을 찾을 수 없습니다.",
        )

    doc, score = results_with_score[0]
    SIMILARITY_THRESHOLD = 0.7  # 코사인 거리 기준 (0.75 이상이면 관련 없음)

    if DEBUG_WORKFLOW:
        _w(f"similarity_score={score:.3f}, threshold={SIMILARITY_THRESHOLD}")

    if score < SIMILARITY_THRESHOLD:
        _w(f"score too high ({score:.3f}), rejecting query")
        raise HTTPException(
            status_code=404,
            detail="죄송합니다. 질문하신 내용과 관련된 답변을 제공할 수 없습니다. "
            "다른 질문을 해주시면 도와드리겠습니다.",
        )

    answer = doc.metadata.get("answer")
    matched_question = doc.page_content

    if DEBUG_WORKFLOW:
        _w(f"matched_question='{matched_question}'")
        _w(f"returning_answer='{answer}'")

    if answer is None:
        raise HTTPException(
            status_code=500, detail="검색 결과에 유효한 답변 메타데이터가 없습니다."
        )

    return ChatResponse(
        answer=answer, matched_question=matched_question, score=float(score)
    )


@app.on_event("startup")
def on_startup() -> None:
    _w("=== FastAPI startup ===")
    if DEBUG_WORKFLOW:
        _w(f"config -> QDRANT_URL={QDRANT_URL}, QDRANT_COLLECTION={QDRANT_COLLECTION}")
        _w(f"config -> QA_EXCEL_PATH={EXCEL_PATH}, EMBEDDING_MODEL={EMBEDDING_MODEL}")
    if not os.path.exists(EXCEL_PATH):
        _w(f"excel file not found at '{EXCEL_PATH}', skipping auto initialization")
        return

    try:
        client = get_qdrant_client()
        _ = client.get_collection(collection_name=QDRANT_COLLECTION)
        # 컬렉션이 있으면 벡터스토어만 연결 시도 (업서트는 생략)
        _ = get_vectorstore()
        _w("existing collection detected, startup completed without reimport")
    except Exception as exc:
        # 컬렉션이 없거나 연결 실패 시 재생성 및 업서트
        _w(f"Qdrant connection failed: {exc}")
        _w("Attempting to initialize database...")
        try:
            initialize_db_and_data(EXCEL_PATH)
        except Exception as init_exc:
            _w(f"Database initialization failed: {init_exc}")
            _w("Service will start but Qdrant operations will fail until connection is established")


@app.post("/admin/init")
def admin_init() -> Dict[str, Any]:
    if not os.path.exists(EXCEL_PATH):
        raise HTTPException(
            status_code=400,
            detail=f"엑셀 파일을 찾을 수 없습니다: {EXCEL_PATH}",
        )
    _w("=== /admin/init triggered ===")

    try:
        vs, cnt = initialize_db_and_data(EXCEL_PATH)
        return {
            "status": "ok",
            "collection": QDRANT_COLLECTION,
            "inserted": cnt,
        }
    except Exception as exc:
        _w(f"/admin/init failed: {exc}")
        raise HTTPException(status_code=500, detail=f"초기화 실패: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


