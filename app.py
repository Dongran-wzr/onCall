from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from agent import OnCallAgent, to_sse
from semantic_search import MAX_RESULTS, SemanticSearchEngine
from utils import DocumentRecord, build_snippet, count_keyword_occurrences, html_to_document, iter_html_files


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TEMPLATES_DIR = BASE_DIR / "templates"

logger = logging.getLogger("oncall_search")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class DocumentCreateRequest(BaseModel):
    id: str = Field(..., min_length=1, description="Document identifier, e.g. sop-001")
    html: str = Field(..., min_length=1, description="Raw HTML payload")


class DocumentCreateResponse(BaseModel):
    id: str
    title: str


class SearchResult(BaseModel):
    id: str
    title: str
    snippet: str
    score: int


class SearchResponse(BaseModel):
    query: str
    total: int
    results: list[SearchResult]


class SemanticSearchResult(BaseModel):
    id: str
    title: str
    snippet: str
    score: float


class SemanticSearchResponse(BaseModel):
    query: str
    total: int
    results: list[SemanticSearchResult]


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[ChatHistoryMessage] = Field(default_factory=list)
    stream: bool = True


class ChatResponse(BaseModel):
    answer: str
    events: list[dict[str, Any]]


class DocumentStore:
    def __init__(self) -> None:
        self._documents: dict[str, DocumentRecord] = {}

    def upsert_document(self, document: DocumentRecord) -> DocumentRecord:
        self._documents[document.id] = document
        return document

    def all_documents(self) -> list[DocumentRecord]:
        return list(self._documents.values())

    def search(self, query: str) -> list[SearchResult]:
        normalized_query = query.strip()
        documents = self.all_documents()

        if not normalized_query:
            return [
                SearchResult(
                    id=document.id,
                    title=document.title,
                    snippet=build_snippet(document.clean_text, ""),
                    score=0,
                )
                for document in sorted(documents, key=lambda item: item.id)
            ]

        results: list[SearchResult] = []
        for document in documents:
            score = count_keyword_occurrences(document.clean_text, normalized_query)
            if score <= 0:
                continue

            results.append(
                SearchResult(
                    id=document.id,
                    title=document.title,
                    snippet=build_snippet(document.clean_text, normalized_query),
                    score=score,
                )
            )

        results.sort(key=lambda item: (-item.score, item.id))
        return results

    def load_from_directory(self, data_dir: Path) -> int:
        loaded = 0
        for html_file in iter_html_files(data_dir):
            html = html_file.read_text(encoding="utf-8")
            document_id = html_file.stem
            document = html_to_document(document_id=document_id, html=html, keep_original_html=True)
            self.upsert_document(document)
            loaded += 1
            logger.info("Loaded document %s from %s", document_id, html_file.name)
        return loaded


document_store = DocumentStore()
semantic_engine = SemanticSearchEngine()
agent: OnCallAgent | None = None


def list_document_manifest() -> list[dict[str, str]]:
    return [
        {
            "fname": f"{document.id}.html",
            "title": document.title,
        }
        for document in sorted(document_store.all_documents(), key=lambda item: item.id)
    ]


def suggest_agent_files(query: str) -> list[dict[str, str]]:
    suggestions: list[dict[str, str]] = []

    try:
        if semantic_engine.ready:
            semantic_results = semantic_engine.search(query, top_k=3, min_score=0.2)
            for item in semantic_results:
                suggestions.append({"fname": f"{item.id}.html", "title": item.title})
    except Exception:
        logger.exception("Failed to build semantic suggestions for agent query=%r", query)

    lowered = query.casefold()
    keyword_map = {
        "数据库": "sop-002.html",
        "主从": "sop-002.html",
        "延迟": "sop-002.html",
        "oom": "sop-001.html",
        "服务": "sop-001.html",
        "p0": "sop-004.html",
        "入侵": "sop-005.html",
        "黑客": "sop-005.html",
        "安全": "sop-005.html",
        "推荐": "sop-008.html",
        "模型": "sop-008.html",
        "质量": "sop-008.html",
    }
    manifest_by_name = {item["fname"]: item for item in list_document_manifest()}
    for token, fname in keyword_map.items():
        if token in lowered and fname in manifest_by_name:
            suggestions.append(manifest_by_name[fname])

    unique: dict[str, dict[str, str]] = {}
    for item in suggestions:
        unique[item["fname"]] = item
    return list(unique.values())[:4]


def handle_agent_file_write(path: Path) -> None:
    if path.suffix.lower() != ".html":
        logger.info("Agent created non-html file %s, skipping index rebuild", path.name)
        return

    document = html_to_document(document_id=path.stem, html=path.read_text(encoding="utf-8"), keep_original_html=True)
    document_store.upsert_document(document)
    try:
        semantic_engine.rebuild_index(document_store.all_documents())
    except Exception:
        logger.exception("Semantic index rebuild failed after agent file write: %s", path.name)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    loaded = document_store.load_from_directory(DATA_DIR)

    def _init_semantic_engine() -> None:
        try:
            semantic_engine.rebuild_index(document_store.all_documents())
            logger.info("Semantic search engine initialized successfully")
        except Exception:
            logger.exception("Semantic search engine failed to initialize")

    threading.Thread(target=_init_semantic_engine, daemon=True, name="semantic-init").start()

    global agent
    agent = OnCallAgent(
        data_dir=DATA_DIR,
        list_documents=list_document_manifest,
        suggest_files=suggest_agent_files,
        on_file_write=handle_agent_file_write,
    )
    logger.info("On-Call agent initialized successfully")
    logger.info("Application startup complete, loaded %s document(s)", loaded)
    yield


app = FastAPI(
    title="On-Call Assistant Search API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/v3")


@app.get("/v1", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "document_count": len(document_store.all_documents()),
            "api_path": "/v1/search",
            "page_title": "On-Call 助手关键词搜索",
            "page_heading": "On-Call 助手文档搜索",
            "page_description": "搜索已加载的 SOP 文档，支持中文、英文以及特殊字符关键词查询。",
            "search_placeholder": "输入关键词，例如：故障 / OOM / CDN / &",
            "score_label": "相关度分数",
        },
    )


@app.get("/v2", response_class=HTMLResponse)
async def semantic_home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "document_count": len(document_store.all_documents()),
            "api_path": "/v2/search",
            "page_title": "On-Call 助手语义搜索",
            "page_heading": "On-Call 助手语义搜索",
            "page_description": "输入自然语言问题，系统会用语义向量召回最相关的 SOP 文档。",
            "search_placeholder": "输入自然语言，例如：服务器挂了 / 黑客攻击 / 机器学习模型出问题",
            "score_label": "语义相似度",
        },
    )


@app.get("/v3", response_class=HTMLResponse)
async def agent_home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={},
    )


@app.post("/v1/documents", response_model=DocumentCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_document(payload: DocumentCreateRequest) -> DocumentCreateResponse:
    document_id = payload.id.strip()
    html = payload.html.strip()

    if not document_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Document id cannot be blank")
    if not html:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="HTML content cannot be blank")

    document = html_to_document(document_id=document_id, html=html, keep_original_html=True)
    document_store.upsert_document(document)
    try:
        semantic_engine.rebuild_index(document_store.all_documents())
    except Exception:
        logger.exception("Semantic index rebuild failed after document upsert")
    logger.info("Document upserted via API: %s", document.id)
    return DocumentCreateResponse(id=document.id, title=document.title)


@app.get("/v1/search", response_model=SearchResponse)
async def search_documents(q: str = Query(default="", description="Keyword query")) -> SearchResponse:
    results = document_store.search(q)
    logger.info("Search executed, query=%r, matched=%s", q, len(results))
    return SearchResponse(query=q, total=len(results), results=results)


@app.get("/v2/search", response_model=SemanticSearchResponse)
async def semantic_search_documents(q: str = Query(default="", description="Natural language query")) -> SemanticSearchResponse:
    if not semantic_engine.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Semantic search model is not ready. Check startup logs for model download or initialization errors.",
        )

    try:
        results = semantic_engine.search(q, top_k=MAX_RESULTS)
    except Exception as exc:
        logger.exception("Semantic search failed for query=%r", q)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    response_results = [
        SemanticSearchResult(
            id=result.id,
            title=result.title,
            snippet=result.snippet,
            score=result.score,
        )
        for result in results
    ]
    logger.info("Semantic search executed, query=%r, matched=%s", q, len(response_results))
    return SemanticSearchResponse(query=q, total=len(response_results), results=response_results)


@app.post("/v3/chat", response_model=ChatResponse)
async def agent_chat(payload: ChatRequest):
    if agent is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Agent is not initialized")

    history = [{"role": item.role, "content": item.content} for item in payload.history]

    try:
        events = agent.run(message=payload.message, history=history)
    except Exception as exc:
        logger.exception("Agent execution failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    if payload.stream:
        def event_generator():
            for event in events:
                yield to_sse([event])

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    final_answer = ""
    serialized_events: list[dict[str, Any]] = []
    for event in events:
        serialized_events.append({"type": event.type, "payload": event.payload})
        if event.type == "final":
            final_answer = str(event.payload.get("answer", ""))

    return ChatResponse(answer=final_answer, events=serialized_events)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
