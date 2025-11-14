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
    answer: Optional[str] = None
    matched_question: Optional[str] = None
    score: Optional[float] = None
    confidence: Optional[str] = None  # high, medium, low, needs_confirmation
    threshold_used: Optional[float] = None
    needs_confirmation: Optional[bool] = None  # 사용자 확인 필요 여부
    alternative_questions: Optional[List[str]] = None  # 대안 질문들


APP_NAME = "eschaT-qa-backend"

# .env 자동 로드 (프로젝트 루트) - 환경 변수 읽기 전에 먼저 로드해야 함
load_dotenv(dotenv_path=os.getenv("ENV_FILE", ".env"))

# 환경 변수 (배포/로컬 모두에서 작동)
QDRANT_URL = os.getenv("QDRANT_URL")
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
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)


def get_qdrant_client() -> QdrantClient:
    """Qdrant 클라이언트 생성"""
    _w("Creating Qdrant client...")
    timeout = float(os.getenv("QDRANT_TIMEOUT", "120.0"))
    
    client_kwargs = {
        "url": QDRANT_URL,
        "timeout": timeout,
        "prefer_grpc": False,  # HTTP만 사용 (HTTPS와 더 호환)
    }
    
    if QDRANT_API_KEY:
        client_kwargs["api_key"] = QDRANT_API_KEY
        _w("Using API key authentication")
    else:
        _w("No API key provided, using unauthenticated connection")
    
    _w(f"Qdrant URL: {QDRANT_URL}, timeout: {timeout}s")
    return QdrantClient(**client_kwargs)


def build_embeddings() -> SentenceTransformerEmbeddings:
    """임베딩 모델 생성"""
    _w(f"Building embeddings with model: {EMBEDDING_MODEL}")
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)


def initialize_db_and_data(filepath: str) -> Tuple[LCQdrant, int]:
    """데이터베이스 초기화 및 데이터 로드"""
    import time
    start_time = time.time()
    
    _w("=" * 60)
    _w("=== DB INITIALIZATION START ===")
    _w(f"Target file: '{filepath}'")
    
    # 1. Qdrant 연결 테스트
    _w("[1/5] Testing Qdrant connection...")
    client = get_qdrant_client()
    try:
        collections = client.get_collections()
        _w(f"✓ Qdrant connected. Existing collections: {len(collections.collections)}")
    except Exception as conn_exc:
        _w(f"✗ Qdrant connection failed: {conn_exc}")
        raise ConnectionError(f"Failed to connect to Qdrant: {conn_exc}") from conn_exc
    
    # 2. 엑셀 파일 파싱
    _w("[2/5] Parsing Excel file...")
    parse_start = time.time()
    df = read_excel_flex(filepath)
    qa_rows = parse_qa_data(df)
    parse_time = time.time() - parse_start
    _w(f"✓ Parsed {len(qa_rows)} Q&A pairs in {parse_time:.3f}s")
    
    if DEBUG_WORKFLOW and qa_rows:
        _w("Sample Q&A pair:")
        sample = qa_rows[0]
        _w(f"  Q: {sample['question']}")
        _w(f"  A: {sample['answer'][:80]}...")

    # 3. 문서 생성
    _w("[3/5] Creating document objects...")
    docs = [
        Document(page_content=it["question"], metadata={"answer": it["answer"]})
        for it in qa_rows
    ]
    _w(f"✓ Created {len(docs)} documents")

    # 4. 임베딩 모델 로드
    _w("[4/5] Loading embedding model...")
    embed_start = time.time()
    embeddings = build_embeddings()
    embed_time = time.time() - embed_start
    _w(f"✓ Embedding model loaded in {embed_time:.3f}s")

    # 5. 컬렉션 재생성 및 데이터 업로드
    _w("[5/5] Uploading to Qdrant...")
    try:
        client.delete_collection(collection_name=QDRANT_COLLECTION)
        _w(f"  Deleted existing collection '{QDRANT_COLLECTION}'")
    except Exception:
        _w(f"  Collection '{QDRANT_COLLECTION}' does not exist, creating new one")

    upload_start = time.time()
    vs = LCQdrant.from_documents(
        documents=docs,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION,
    )
    upload_time = time.time() - upload_start
    _w(f"✓ Uploaded {len(docs)} documents to Qdrant in {upload_time:.3f}s")
    
    total_time = time.time() - start_time
    _w(f"=== DB INITIALIZATION COMPLETE (Total: {total_time:.3f}s) ===")
    _w("=" * 60)

    return vs, len(docs)


def get_vectorstore() -> LCQdrant:
    """벡터스토어 인스턴스 생성"""
    _w("Creating vectorstore instance...")
    embeddings = build_embeddings()
    vs = LCQdrant(
        client=get_qdrant_client(),
        collection_name=QDRANT_COLLECTION,
        embeddings=embeddings,
    )
    _w(f"Vectorstore ready for collection: {QDRANT_COLLECTION}")
    return vs


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "app": APP_NAME}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """챗봇 질의응답 엔드포인트"""
    import time
    start_time = time.time()
    
    _w("=" * 60)
    _w("=== CHAT REQUEST START ===")
    
    question = (request.question or "").strip()
    if not question:
        _w("ERROR: Empty question received")
        raise HTTPException(
            status_code=400, 
            detail="질문이 필요합니다.",
            headers={"Access-Control-Allow-Origin": "*"}
        )

    _w(f"[1/5] Received question: '{question}'")
    _w(f"Question length: {len(question)} chars")

    # 벡터 검색
    try:
        _w("[2/5] Getting vectorstore...")
        vs = get_vectorstore()
        
        TOP_K = int(os.getenv("TOP_K", "3"))
        _w(f"[3/5] Performing vector search (k={TOP_K})...")
        search_start = time.time()
        results_with_score = vs.similarity_search_with_score(question, k=TOP_K)
        search_time = time.time() - search_start
        _w(f"Vector search completed in {search_time:.3f}s")
        
    except Exception as exc:
        _w(f"ERROR: Vector search failed - {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500, 
            detail=f"검색 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            headers={"Access-Control-Allow-Origin": "*"}
        ) from exc

    if not results_with_score:
        _w("ERROR: No search results found")
        raise HTTPException(
            status_code=404,
            detail="죄송합니다. 제공된 데이터베이스에서 관련 답변을 찾을 수 없습니다.",
            headers={"Access-Control-Allow-Origin": "*"}
        )

    # 결과 분석
    _w(f"[4/5] Analyzing {len(results_with_score)} search results...")
    doc, score = results_with_score[0]
    answer = doc.metadata.get("answer")
    matched_question = doc.page_content
    
    # 신뢰도 레벨 계산
    if score >= 0.85:
        confidence = "high"
        needs_confirmation = False
    elif score >= 0.7:
        confidence = "medium"
        needs_confirmation = False
    elif score >= 0.5:
        confidence = "low"
        needs_confirmation = True
    else:
        _w(f"REJECTED: Score too low ({score:.3f} < 0.5)")
        raise HTTPException(
            status_code=404,
            detail="죄송합니다. 질문하신 내용과 관련된 답변을 제공할 수 없습니다. "
            "다른 질문을 해주시면 도와드리겠습니다.",
            headers={"Access-Control-Allow-Origin": "*"}
        )
    
    # 대안 질문 추출
    alternative_questions = None
    if len(results_with_score) > 1 and needs_confirmation:
        alternative_questions = [
            d.page_content for d, s in results_with_score[1:min(3, len(results_with_score))]
        ]
        _w(f"Alternative questions found: {len(alternative_questions)}")

    # 상세 로그
    _w(f"Top-{len(results_with_score)} search results:")
    for i, (d, s) in enumerate(results_with_score):
        _w(f"  [{i+1}] Score: {s:.4f} | Question: '{d.page_content[:60]}...'")
    
    _w(f"[5/5] Final decision:")
    _w(f"  - Matched question: '{matched_question}'")
    _w(f"  - Similarity score: {score:.4f}")
    _w(f"  - Confidence: {confidence}")
    _w(f"  - Needs confirmation: {needs_confirmation}")
    _w(f"  - Answer length: {len(answer) if answer else 0} chars")

    if answer is None:
        _w("ERROR: Answer metadata is None")
        raise HTTPException(
            status_code=500, 
            detail="검색 결과에 유효한 답변 메타데이터가 없습니다.",
            headers={"Access-Control-Allow-Origin": "*"}
        )

    total_time = time.time() - start_time
    _w(f"=== CHAT REQUEST COMPLETE (Total: {total_time:.3f}s) ===")
    _w("=" * 60)

    return ChatResponse(
        answer=answer,
        matched_question=matched_question,
        score=float(score),
        confidence=confidence,
        needs_confirmation=needs_confirmation,
        alternative_questions=alternative_questions,
        threshold_used=0.5
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

    # Qdrant 연결 시도 (최대 3번 재시도)
    import time
    max_retries = 3
    _w("Starting Qdrant connection attempts...")
    
    for attempt in range(1, max_retries + 1):
        try:
            _w(f"[Attempt {attempt}/{max_retries}] Connecting to Qdrant...")
            client = get_qdrant_client()
            collections = client.get_collections()
            _w(f"✓ Connected. Found {len(collections.collections)} collections.")
            
            # 컬렉션 확인
            try:
                collection_info = client.get_collection(collection_name=QDRANT_COLLECTION)
                _w(f"✓ Collection '{QDRANT_COLLECTION}' exists")
                _w(f"  Points count: {collection_info.points_count}")
                _w(f"  Vectors count: {collection_info.vectors_count}")
                _ = get_vectorstore()
                _w("✓ Startup completed - using existing collection")
                return
            except Exception as coll_exc:
                _w(f"✗ Collection '{QDRANT_COLLECTION}' not found: {coll_exc}")
                _w("Initializing database with Excel data...")
                initialize_db_and_data(EXCEL_PATH)
                _w("✓ Database initialized successfully")
                return
        except Exception as exc:
            _w(f"✗ Connection attempt {attempt} failed: {type(exc).__name__}: {exc}")
            if attempt < max_retries:
                wait_time = attempt * 2
                _w(f"  Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                _w("✗ All connection attempts failed")
                _w("⚠ Service will start but Qdrant operations will fail")
                _w("  Use /admin/init endpoint once Qdrant is available")


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
            "inserted": cnt
        }
    except Exception as exc:
        _w(f"/admin/init failed: {exc}")
        raise HTTPException(status_code=500, detail=f"초기화 실패: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


