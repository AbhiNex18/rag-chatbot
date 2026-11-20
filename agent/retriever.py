import os
import glob
import json
import hashlib
import logging
import re
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger("support_agent")

INDEX_META_FILE = "index_meta.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "you", "your", "our", "we", "i", "me", "my",
    "on", "in", "at", "to", "for", "of", "and", "or", "with", "from",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "how", "when", "where", "why", "can", "could", "would", "should",
    "please", "offer", "offers", "about", "any", "if",
}


class FallbackEmbeddings(Embeddings):
    """Deterministic hashed bag-of-words fallback if sentence-transformers cannot load."""

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.model_name = "fallback-hash-v2"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def __call__(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        words = text.lower().split()
        vec = np.zeros(self.dim, dtype=np.float32)
        for w in words:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


class SentenceTransformerEmbeddings(Embeddings):
    """Local MiniLM embeddings with L2-normalized vectors (cosine-friendly)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()


def get_embeddings() -> Embeddings:
    """Prefer local sentence-transformers. HuggingFace token is optional for MiniLM."""
    try:
        embeddings = SentenceTransformerEmbeddings(EMBEDDING_MODEL)
        logger.info("Embeddings backend: sentence-transformers (%s)", EMBEDDING_MODEL)
        return embeddings
    except Exception as e1:
        logger.warning("sentence-transformers unavailable (%s). Trying OpenAI embeddings.", e1)

    if os.environ.get("OPENAI_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings()
            embeddings.model_name = "openai"
            logger.info("Embeddings backend: OpenAI")
            return embeddings
        except Exception as e:
            logger.warning("OpenAIEmbeddings failed: %s", e)

    logger.warning("Embeddings backend: hashed fallback (lower retrieval quality)")
    return FallbackEmbeddings()


def _embedding_name(embeddings: Embeddings) -> str:
    return getattr(embeddings, "model_name", embeddings.__class__.__name__)


def _l2_to_cosine(distance: float) -> float:
    """Convert FAISS L2 distance on normalized vectors to cosine similarity in [0, 1]."""
    dist = float(distance)
    if dist < 0:
        return max(0.0, min(1.0, -dist))
    cosine = 1.0 - (dist ** 2) / 2.0
    return round(max(0.0, min(1.0, cosine)), 4)


def _tokens(text: str) -> set:
    words = set(re.findall(r"[a-z0-9$]+", text.lower()))
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _content_tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9$]+", text.lower()) if t not in _STOPWORDS and len(t) > 1]


def _ngrams(tokens: List[str], n: int) -> set:
    return {" ".join(tokens[i : i + n]) for i in range(0, max(0, len(tokens) - n + 1))}


def _phrase_hits(query: str, content: str) -> int:
    q_toks = _content_tokens(query)
    c_toks = _content_tokens(content)
    hits = 0
    for n in (2, 3):
        hits += len(_ngrams(q_toks, n) & _ngrams(c_toks, n))
    return hits


def _id_like_tokens(text: str) -> set:
    """Order-ID tokens that should not drive policy retrieval."""
    found = set(re.findall(r"ord-?\d{4,6}", text.lower()))
    found.update(re.findall(r"\b\d{4,6}\b", text.lower()))
    return found


TOPIC_ROUTES = [
    {
        "query": [r"\breturn", r"\brefund", r"\bsend (?:it |them )?back"],
        "exclude": [r"\bdamag", r"\bdefect", r"\bbroken", r"\bmissing item"],
        "prefer": ["01_return_policy"],
        "avoid": ["08_damaged_items"],
    },
    {
        "query": [r"\bdamag", r"\bdefect", r"\bbroken", r"\bmissing item"],
        "prefer": ["08_damaged_items"],
        "avoid": [],
    },
    {
        "query": [r"\bfree shipping", r"\borders over", r"shipping take", r"\bshipping options?"],
        "prefer": ["02_shipping_timelines"],
        "avoid": ["01_return_policy"],
    },
]


def _topic_boost(query: str, source: str) -> float:
    source_l = source.lower()
    q = query.lower()
    boost = 0.0
    for route in TOPIC_ROUTES:
        if not any(re.search(p, q) for p in route["query"]):
            continue
        if any(re.search(p, q) for p in route.get("exclude", [])):
            continue
        if any(name in source_l for name in route.get("prefer", [])):
            boost += 0.28
        if any(name in source_l for name in route.get("avoid", [])):
            boost -= 0.22
    return boost


class KBRetriever:
    """Manages KB chunking, FAISS indexing, and filtered similarity search."""

    def __init__(self, kb_dir: Optional[str] = None, index_dir: Optional[str] = None):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        self.kb_dir = kb_dir or os.path.join(root_dir, "kb")
        self.index_dir = index_dir or os.path.join(root_dir, "data", "faiss_index")
        self.embeddings = get_embeddings()
        self.vectorstore: Optional[FAISS] = None

    def _kb_fingerprint(self) -> str:
        hasher = hashlib.md5()
        hasher.update(_embedding_name(self.embeddings).encode("utf-8"))
        hasher.update(b"|chunk-v3-wholeshort")
        for filepath in sorted(glob.glob(os.path.join(self.kb_dir, "*.md"))):
            hasher.update(os.path.basename(filepath).encode("utf-8"))
            with open(filepath, "rb") as f:
                hasher.update(f.read())
        return hasher.hexdigest()

    def _meta_path(self) -> str:
        return os.path.join(self.index_dir, INDEX_META_FILE)

    def load_kb_documents(self) -> List[Document]:
        documents = []
        md_files = glob.glob(os.path.join(self.kb_dir, "*.md"))
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "title"), ("##", "section")]
        )
        body_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        for filepath in sorted(md_files):
            filename = os.path.basename(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            title_line = content.strip().split("\n", 1)[0].lstrip("# ").strip() or filename

            # Keep short FAQ files whole so answers are not missing the eligibility rules.
            if len(content) <= 1800:
                prefix = f"Source: {filename} | Topic: {title_line}\n"
                documents.append(
                    Document(
                        page_content=prefix + content.strip(),
                        metadata={
                            "source": filename,
                            "title": title_line,
                            "section": "",
                            "filepath": filepath,
                        },
                    )
                )
                continue

            header_docs = header_splitter.split_text(content)
            if not header_docs:
                header_docs = [Document(page_content=content, metadata={"title": filename})]

            for doc in body_splitter.split_documents(header_docs):
                title = doc.metadata.get("title") or filename.replace("_", " ")
                section = doc.metadata.get("section") or ""
                prefix_parts = [f"Source: {filename}", f"Topic: {title}"]
                if section:
                    prefix_parts.append(f"Section: {section}")
                prefixed = " | ".join(prefix_parts) + "\n" + doc.page_content.strip()
                documents.append(
                    Document(
                        page_content=prefixed,
                        metadata={
                            "source": filename,
                            "title": title,
                            "section": section,
                            "filepath": filepath,
                        },
                    )
                )
        return documents

    def build_index(self) -> FAISS:
        raw_docs = self.load_kb_documents()
        if not raw_docs:
            raise ValueError(f"No markdown documents found in {self.kb_dir}")

        logger.info("Indexed %s chunks from %s", len(raw_docs), self.kb_dir)
        print(f"[Retriever] Indexed {len(raw_docs)} policy chunks from {self.kb_dir}")
        self.vectorstore = FAISS.from_documents(raw_docs, self.embeddings)

        os.makedirs(self.index_dir, exist_ok=True)
        self.vectorstore.save_local(self.index_dir)
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fingerprint": self._kb_fingerprint(),
                    "embedding": _embedding_name(self.embeddings),
                    "chunks": len(raw_docs),
                },
                f,
            )
        return self.vectorstore

    def load_or_build_index(self) -> FAISS:
        index_file = os.path.join(self.index_dir, "index.faiss")
        meta_file = self._meta_path()
        current_fp = self._kb_fingerprint()

        if os.path.exists(index_file) and os.path.exists(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("fingerprint") == current_fp:
                    self.vectorstore = FAISS.load_local(
                        self.index_dir,
                        self.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    logger.info("Loaded FAISS index (%s chunks)", meta.get("chunks"))
                    return self.vectorstore
                logger.info("KB or embedding model changed; rebuilding FAISS index.")
            except Exception as e:
                logger.warning("Could not load local index (%s); rebuilding.", e)

        return self.build_index()

    @staticmethod
    def _policy_query(query: str) -> str:
        """Drop order IDs so example IDs in unrelated policies cannot dominate retrieval."""
        cleaned = re.sub(
            r"\b(?:my\s+)?(?:ord(?:er)?|pkg|package)\s*(?:id|number|no\.?)?\s*(?:is|:)?\s*(?:ORD-?)?\d{4,6}\b",
            " ",
            query,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\bORD-?\d{4,6}\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned or query

    def search_with_confidence(
        self, query: str, top_k: int = 5
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Cosine search with duplicate removal and lexical filtering."""
        if self.vectorstore is None:
            self.load_or_build_index()

        retrieval_query = self._policy_query(query)
        fetch_k = max(top_k * 4, 12)
        results = self.vectorstore.similarity_search_with_score(retrieval_query, k=fetch_k)
        if not results:
            return [], 0.0

        query_terms = _tokens(retrieval_query)
        scored = []
        seen = set()
        for doc, dist in results:
            sim_score = _l2_to_cosine(dist)
            content = doc.page_content
            source = doc.metadata.get("source", "unknown")
            fingerprint = hashlib.md5((source + content).encode("utf-8")).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)

            chunk_terms = _tokens(content) | _tokens(
                source.replace("_", " ").replace(".md", "")
            )
            id_tokens = _id_like_tokens(query) | _id_like_tokens(retrieval_query)
            overlap = (query_terms - id_tokens) & (chunk_terms - id_tokens)
            coverage = len(overlap) / max(1, len(query_terms - id_tokens) or 1)
            phrase_hits = _phrase_hits(retrieval_query, content)
            rank_score = round(
                0.50 * sim_score
                + 0.35 * coverage
                + 0.15 * min(1.0, phrase_hits / 2.0)
                + _topic_boost(retrieval_query, source),
                4,
            )
            scored.append(
                {
                    "content": content,
                    "source": source,
                    "title": doc.metadata.get("title", source),
                    "section": doc.metadata.get("section", ""),
                    "score": sim_score,
                    "term_overlap": sorted(overlap),
                    "overlap_count": len(overlap),
                    "coverage": round(coverage, 4),
                    "phrase_hits": phrase_hits,
                    "rank_score": rank_score,
                }
            )

        if not scored:
            return [], 0.0

        scored.sort(key=lambda x: (x["rank_score"], x["overlap_count"], x["score"]), reverse=True)
        top_rank = scored[0]["rank_score"]

        filtered = []
        for item in scored:
            if item["rank_score"] < top_rank - 0.18:
                continue
            if item["overlap_count"] == 0 and item["score"] < 0.62:
                continue
            if item["score"] < 0.28:
                continue
            filtered.append(item)
            if len(filtered) >= top_k:
                break

        if not filtered:
            filtered = [scored[0]]

        confidence = filtered[0]["score"]
        if filtered[0]["overlap_count"] == 0:
            confidence = round(confidence * 0.7, 4)
        elif filtered[0].get("coverage", 0) >= 0.6:
            confidence = max(confidence, 0.56)
        elif (
            _topic_boost(retrieval_query, filtered[0]["source"]) > 0
            and filtered[0]["overlap_count"] >= 1
            and filtered[0]["score"] >= 0.32
        ):
            # Topic-aligned hit (e.g. return question -> return policy) even if cosine is moderate.
            confidence = max(confidence, 0.56)

        return filtered, confidence


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    retriever = KBRetriever()
    retriever.build_index()
    for q in [
        "What is your return policy?",
        "Do you offer free shipping on orders over $50?",
        "How long does shipping take?",
    ]:
        matches, score = retriever.search_with_confidence(q)
        print(f"\nQ: {q}\nTop score: {score}")
        for m in matches:
            print(f"- [{m['source']} {m['score']}] overlap={m['term_overlap']}: {m['content'][:120]}...")
