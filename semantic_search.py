from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
from sentence_transformers import SentenceTransformer, util

from utils import DocumentRecord, truncate_text


logger = logging.getLogger("oncall_search.semantic")

DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
MIN_SCORE_THRESHOLD = 0.30
MAX_RESULTS = 10
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


@dataclass(slots=True)
class SemanticResult:
    id: str
    title: str
    snippet: str
    score: float


@dataclass(slots=True)
class SemanticDocumentEntry:
    document: DocumentRecord
    search_text: str
    segments: tuple[str, ...]
    segment_embeddings: torch.Tensor


class SemanticSearchEngine:
    """Small semantic retrieval layer for the current in-memory document store.

    With only 10 SOP documents, startup embedding + in-memory cosine similarity
    keeps the implementation simple while staying comfortably below the target
    latency after the model is loaded and cached locally.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self.model: SentenceTransformer | None = None
        self.entries: list[SemanticDocumentEntry] = []
        self.document_embeddings: torch.Tensor | None = None
        self.ready = False

    def initialize(self) -> None:
        if self.model is not None:
            return

        logger.info("Loading semantic model: %s", self.model_name)
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "15")
        self.model = SentenceTransformer(self.model_name)
        self.ready = True
        logger.info("Semantic model loaded successfully")

    def rebuild_index(self, documents: list[DocumentRecord]) -> None:
        if not documents:
            self.entries = []
            self.document_embeddings = None
            return

        self.initialize()
        assert self.model is not None

        prepared_entries: list[SemanticDocumentEntry] = []
        document_texts: list[str] = []
        all_segments: list[str] = []
        segment_lengths: list[int] = []

        for document in sorted(documents, key=lambda item: item.id):
            # 增加标题权重：重复标题 3 次
            search_text = f"{document.title}。{document.title}。{document.title}。{document.clean_text}"
            segments = document.segments or (document.clean_text,)
            document_texts.append(search_text)
            all_segments.extend(segments)
            segment_lengths.append(len(segments))
            prepared_entries.append(
                SemanticDocumentEntry(
                    document=document,
                    search_text=search_text,
                    segments=segments,
                    segment_embeddings=torch.empty(0),
                )
            )

        self.document_embeddings = self._encode(document_texts)
        segment_embeddings = self._encode(all_segments)

        offset = 0
        for index, length in enumerate(segment_lengths):
            prepared_entries[index].segment_embeddings = segment_embeddings[offset : offset + length]
            offset += length

        self.entries = prepared_entries
        logger.info("Semantic index rebuilt for %s document(s)", len(self.entries))

    def search(self, query: str, top_k: int = MAX_RESULTS, min_score: float = MIN_SCORE_THRESHOLD) -> list[SemanticResult]:
        normalized_query = query.strip()
        if not self.ready or self.model is None:
            raise RuntimeError("Semantic search engine is not ready")

        if not normalized_query:
            return [
                SemanticResult(
                    id=entry.document.id,
                    title=entry.document.title,
                    snippet=truncate_text(entry.segments[0] if entry.segments else entry.document.clean_text),
                    score=0.0,
                )
                for entry in self.entries[:top_k]
            ]

        if self.document_embeddings is None or not self.entries:
            return []

        expanded_query = self._expand_query(normalized_query)
        query_embedding = self._encode([self._prepare_query(expanded_query)])
        similarity_scores = util.cos_sim(query_embedding, self.document_embeddings)[0]

        results: list[SemanticResult] = []
        for index, entry in enumerate(self.entries):
            raw_score = float(similarity_scores[index].item())
            score = max(0.0, min(1.0, (raw_score + 1.0) / 2.0))
            if score < min_score:
                continue

            snippet = self._best_snippet(query_embedding, entry)
            results.append(
                SemanticResult(
                    id=entry.document.id,
                    title=entry.document.title,
                    snippet=snippet,
                    score=round(score, 4),
                )
            )

        results.sort(key=lambda item: (-item.score, item.id))
        return results[:top_k]

    def _encode(self, texts: list[str]) -> torch.Tensor:
        assert self.model is not None
        return self.model.encode(
            texts,
            batch_size=16,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

    def _prepare_query(self, query: str) -> str:
        if "bge" in self.model_name.casefold():
            return f"{BGE_QUERY_PREFIX}{query}"
        return query

    def _expand_query(self, query: str) -> str:
        lowered = query.casefold()
        expansions: list[str] = []

        synonym_groups = {
            "服务器": "后端服务 SRE 基础设施 集群 节点 实例 宕机 宕机 故障",
            "挂了": "宕机 崩溃 不可用 故障 无法访问 无法连接 挂死",
            "黑客": "安全攻击 入侵 恶意流量 DDoS 漏洞 网络安全",
            "攻击": "安全攻击 入侵 DDoS 漏洞 渗透",
            "模型": "机器学习 AI 算法 推理 效果 模型漂移 模型训练",
            "内存": "OOM 内存泄漏 内存溢出 Heap 堆内存",
            "爆炸": "OOM 崩溃 溢出 撑爆",
            "宕机": "服务不可用 故障 恢复 停止响应 挂掉",
            "流量": "流量突增 洪峰 高并发 突发流量 负载均衡",
            "洪峰": "流量突增 高并发 突发流量 压力测试",
            "漂移": "数据漂移 模型退化 效果下降 准确率下降",
            "SRE": "基础设施 运维 平台 架构 稳定性",
            "后端": "API 接口 服务 逻辑 数据 数据库",
        }

        for token, expansion in synonym_groups.items():
            if token in lowered:
                expansions.append(expansion)

        if not expansions:
            return query

        return f"{query} {' '.join(expansions)}"

    def _best_snippet(self, query_embedding: torch.Tensor, entry: SemanticDocumentEntry) -> str:
        if entry.segment_embeddings.nelement() == 0:
            return truncate_text(entry.document.clean_text)

        scores = util.cos_sim(query_embedding, entry.segment_embeddings)[0]
        best_index = int(torch.argmax(scores).item())
        return truncate_text(entry.segments[best_index], max_length=260)
