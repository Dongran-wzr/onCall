from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

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


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    loaded = document_store.load_from_directory(DATA_DIR)
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


@app.get("/v1", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "document_count": len(document_store.all_documents()),
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
    logger.info("Document upserted via API: %s", document.id)
    return DocumentCreateResponse(id=document.id, title=document.title)


@app.get("/v1/search", response_model=SearchResponse)
async def search_documents(q: str = Query(default="", description="Keyword query")) -> SearchResponse:
    results = document_store.search(q)
    logger.info("Search executed, query=%r, matched=%s", q, len(results))
    return SearchResponse(query=q, total=len(results), results=results)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
