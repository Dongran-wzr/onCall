from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from semantic_search import MAX_RESULTS, SemanticResult, SemanticSearchEngine
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


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    loaded = document_store.load_from_directory(DATA_DIR)
    try:
        semantic_engine.rebuild_index(document_store.all_documents())
        logger.info("Semantic search engine initialized successfully")
    except Exception:
        logger.exception("Semantic search engine failed to initialize")
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
    return RedirectResponse(url="/v2")


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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
