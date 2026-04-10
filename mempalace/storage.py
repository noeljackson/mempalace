"""
storage.py — storage adapters for MemPalace.

The rest of the codebase talks to a small Chroma-like collection interface
instead of importing Chroma directly. This keeps the current behavior while
making room for alternate backends such as Postgres + pgvector.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .config import MempalaceConfig


class StorageError(Exception):
    """Base error for storage adapter failures."""


class CollectionNotFoundError(StorageError):
    """Raised when a collection does not exist."""


def _is_postgres_dsn(value: Optional[str]) -> bool:
    if not value:
        return False
    lowered = value.lower()
    return lowered.startswith("postgres://") or lowered.startswith("postgresql://")


def _normalize_include(include: Optional[Sequence[str]]) -> List[str]:
    return list(include or [])


def _normalize_where(where: Optional[Dict[str, Any]]) -> List[tuple[str, Any]]:
    if not where:
        return []
    if "$and" in where:
        pairs: List[tuple[str, Any]] = []
        for clause in where["$and"]:
            pairs.extend(_normalize_where(clause))
        return pairs
    return list(where.items())


def _validate_metadata_key(key: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        raise StorageError(f"Unsupported metadata key: {key}")
    return key


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(value):.12g}" for value in values) + "]"


def _metadata_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def iter_collection_batches(
    collection,
    batch_size: int = 1000,
    where: Optional[Dict[str, Any]] = None,
    include: Optional[Sequence[str]] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield collection.get batches using a stable offset scan."""
    offset = 0
    include = list(include or [])

    while True:
        batch = collection.get(limit=batch_size, offset=offset, where=where, include=include)
        ids = batch.get("ids", [])
        if not ids:
            break
        yield batch
        offset += len(ids)
        if len(ids) < batch_size:
            break


def migrate_collection(
    source_collection,
    target_collection,
    batch_size: int = 1000,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    """Copy documents and metadata from one collection adapter to another."""
    scanned = 0
    written = 0

    for batch in iter_collection_batches(
        source_collection,
        batch_size=batch_size,
        where=where,
        include=["documents", "metadatas"],
    ):
        ids = batch.get("ids", [])
        documents = batch.get("documents", [])
        metadatas = batch.get("metadatas", [])
        if not ids:
            continue

        target_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        scanned += len(ids)
        written += len(ids)

    return {"scanned": scanned, "written": written}


class EmbeddingProvider:
    """Compute embeddings explicitly for non-Chroma backends."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self._impl = None
        self._mode = None

    def _load(self):
        if self._impl is not None:
            return

        try:
            from chromadb.utils import embedding_functions

            self._impl = embedding_functions.DefaultEmbeddingFunction()
            self._mode = "chroma-default"
            return
        except Exception:
            pass

        model_name = self.model_name or "sentence-transformers/all-MiniLM-L6-v2"
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise StorageError(
                "No embedding provider available. Install chromadb or sentence-transformers."
            ) from exc

        self._impl = SentenceTransformer(model_name)
        self._mode = "sentence-transformers"

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        self._load()
        if not texts:
            return []
        if self._mode == "chroma-default":
            return [list(vector) for vector in self._impl(list(texts))]
        return self._impl.encode(list(texts), normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        vectors = self.embed_documents([text])
        return vectors[0]


class ChromaCollectionAdapter:
    """Thin wrapper around a Chroma collection."""

    def __init__(self, collection):
        self._collection = collection

    def count(self) -> int:
        return self._collection.count()

    def get(
        self,
        ids: Optional[Sequence[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if ids is not None:
            kwargs["ids"] = list(ids)
        if where:
            kwargs["where"] = where
        if limit is not None:
            kwargs["limit"] = limit
        if offset is not None:
            kwargs["offset"] = offset
        if include is not None:
            kwargs["include"] = list(include)
        return self._collection.get(**kwargs)

    def query(
        self,
        query_texts: Sequence[str],
        n_results: int,
        include: Optional[Sequence[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "query_texts": list(query_texts),
            "n_results": n_results,
        }
        if include is not None:
            kwargs["include"] = list(include)
        if where:
            kwargs["where"] = where
        return self._collection.query(**kwargs)

    def add(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
    ) -> None:
        self._collection.add(ids=list(ids), documents=list(documents), metadatas=list(metadatas))

    def upsert(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
    ) -> None:
        self._collection.upsert(
            ids=list(ids),
            documents=list(documents),
            metadatas=list(metadatas),
        )

    def delete(self, ids: Sequence[str]) -> None:
        self._collection.delete(ids=list(ids))


class ChromaBackend:
    def __init__(self, palace_path: str):
        self.palace_path = palace_path

    def get_collection(self, collection_name: str, create: bool = False):
        try:
            Path(self.palace_path).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CollectionNotFoundError(
                f"Cannot access palace path {self.palace_path}: {exc}"
            ) from exc
        try:
            import chromadb
        except Exception as exc:
            raise StorageError(
                "Chroma backend requested but chromadb is not installed."
            ) from exc

        client = chromadb.PersistentClient(path=self.palace_path)
        try:
            if create:
                collection = client.get_or_create_collection(collection_name)
            else:
                collection = client.get_collection(collection_name)
        except Exception as exc:
            raise CollectionNotFoundError(str(exc)) from exc
        return ChromaCollectionAdapter(collection)


class PostgresCollectionAdapter:
    """Postgres + pgvector implementation of the collection interface."""

    def __init__(
        self,
        dsn: str,
        collection_name: str,
        embedding_dimension: int = 384,
        embedding_model: Optional[str] = None,
    ):
        self.dsn = dsn
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.embedder = EmbeddingProvider(model_name=embedding_model)

        try:
            import psycopg
        except Exception as exc:
            raise StorageError(
                "Postgres backend requested but psycopg is not installed."
            ) from exc

        self._psycopg = psycopg
        self._ensure_schema()

    def _connect(self):
        return self._psycopg.connect(self.dsn)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS mempalace_documents (
                        collection_name TEXT NOT NULL,
                        id TEXT NOT NULL,
                        document TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({self.embedding_dimension}) NOT NULL,
                        PRIMARY KEY (collection_name, id)
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_mempalace_documents_collection
                    ON mempalace_documents (collection_name)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_mempalace_documents_metadata
                    ON mempalace_documents USING GIN (metadata)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_mempalace_documents_embedding
                    ON mempalace_documents USING hnsw (embedding vector_cosine_ops)
                    """
                )
            conn.commit()

    def _build_where_sql(self, where: Optional[Dict[str, Any]], params: List[Any]) -> str:
        clauses = ["collection_name = %s"]
        params.append(self.collection_name)

        for key, value in _normalize_where(where):
            safe_key = _validate_metadata_key(key)
            clauses.append(f"metadata ->> '{safe_key}' = %s")
            params.append(_metadata_value(value))

        return " WHERE " + " AND ".join(clauses)

    def count(self) -> int:
        params: List[Any] = []
        sql = "SELECT COUNT(*) FROM mempalace_documents" + self._build_where_sql(None, params)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                return int(row[0] if row else 0)

    def get(
        self,
        ids: Optional[Sequence[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        include = _normalize_include(include)
        want_docs = "documents" in include
        want_meta = "metadatas" in include

        params: List[Any] = []
        sql = "SELECT id, document, metadata FROM mempalace_documents"
        sql += self._build_where_sql(where, params)

        if ids is not None:
            sql += " AND id = ANY(%s)"
            params.append(list(ids))

        sql += " ORDER BY id"
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)
        if offset is not None:
            sql += " OFFSET %s"
            params.append(offset)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        result: Dict[str, Any] = {"ids": [row[0] for row in rows]}
        if want_docs:
            result["documents"] = [row[1] for row in rows]
        if want_meta:
            result["metadatas"] = [row[2] if isinstance(row[2], dict) else json.loads(row[2]) for row in rows]
        return result

    def query(
        self,
        query_texts: Sequence[str],
        n_results: int,
        include: Optional[Sequence[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        include = _normalize_include(include)
        want_docs = "documents" in include
        want_meta = "metadatas" in include
        want_dist = "distances" in include

        all_ids: List[List[str]] = []
        all_docs: List[List[str]] = []
        all_meta: List[List[Dict[str, Any]]] = []
        all_dist: List[List[float]] = []

        with self._connect() as conn:
            with conn.cursor() as cur:
                for query_text in query_texts:
                    params: List[Any] = []
                    embedding = _vector_literal(self.embedder.embed_query(query_text))
                    sql = (
                        "SELECT id, document, metadata, embedding <=> %s::vector AS distance "
                        "FROM mempalace_documents"
                    )
                    params.append(embedding)
                    sql += self._build_where_sql(where, params)
                    sql += " ORDER BY distance ASC LIMIT %s"
                    params.append(n_results)

                    cur.execute(sql, params)
                    rows = cur.fetchall()

                    all_ids.append([row[0] for row in rows])
                    if want_docs:
                        all_docs.append([row[1] for row in rows])
                    if want_meta:
                        all_meta.append(
                            [
                                row[2] if isinstance(row[2], dict) else json.loads(row[2])
                                for row in rows
                            ]
                        )
                    if want_dist:
                        all_dist.append([float(row[3]) for row in rows])

        result: Dict[str, Any] = {"ids": all_ids}
        if want_docs:
            result["documents"] = all_docs
        if want_meta:
            result["metadatas"] = all_meta
        if want_dist:
            result["distances"] = all_dist
        return result

    def add(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
    ) -> None:
        rows = zip(ids, documents, metadatas, self.embedder.embed_documents(documents))
        with self._connect() as conn:
            with conn.cursor() as cur:
                for doc_id, document, metadata, embedding in rows:
                    cur.execute(
                        """
                        INSERT INTO mempalace_documents
                            (collection_name, id, document, metadata, embedding)
                        VALUES (%s, %s, %s, %s::jsonb, %s::vector)
                        """,
                        (
                            self.collection_name,
                            doc_id,
                            document,
                            json.dumps(metadata),
                            _vector_literal(embedding),
                        ),
                    )
            conn.commit()

    def upsert(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
    ) -> None:
        rows = zip(ids, documents, metadatas, self.embedder.embed_documents(documents))
        with self._connect() as conn:
            with conn.cursor() as cur:
                for doc_id, document, metadata, embedding in rows:
                    cur.execute(
                        """
                        INSERT INTO mempalace_documents
                            (collection_name, id, document, metadata, embedding)
                        VALUES (%s, %s, %s, %s::jsonb, %s::vector)
                        ON CONFLICT (collection_name, id)
                        DO UPDATE SET
                            document = EXCLUDED.document,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding
                        """,
                        (
                            self.collection_name,
                            doc_id,
                            document,
                            json.dumps(metadata),
                            _vector_literal(embedding),
                        ),
                    )
            conn.commit()

    def delete(self, ids: Sequence[str]) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM mempalace_documents
                    WHERE collection_name = %s AND id = ANY(%s)
                    """,
                    (self.collection_name, list(ids)),
                )
            conn.commit()


class PostgresBackend:
    def __init__(
        self,
        dsn: str,
        embedding_dimension: int = 384,
        embedding_model: Optional[str] = None,
    ):
        self.dsn = dsn
        self.embedding_dimension = embedding_dimension
        self.embedding_model = embedding_model

    def get_collection(self, collection_name: str, create: bool = False):
        del create
        return PostgresCollectionAdapter(
            dsn=self.dsn,
            collection_name=collection_name,
            embedding_dimension=self.embedding_dimension,
            embedding_model=self.embedding_model,
        )


def _resolve_backend(
    palace_path: Optional[str] = None,
    backend: Optional[str] = None,
    dsn: Optional[str] = None,
):
    cfg = MempalaceConfig()
    resolved_backend = backend or cfg.storage_backend
    resolved_dsn = dsn or cfg.postgres_dsn

    if not backend and (_is_postgres_dsn(palace_path) or _is_postgres_dsn(resolved_dsn)):
        resolved_backend = "postgres"

    if resolved_backend == "postgres":
        postgres_dsn = resolved_dsn or palace_path
        if not postgres_dsn:
            raise StorageError(
                "Postgres backend selected but no DSN configured. Set MEMPALACE_POSTGRES_DSN."
            )
        return PostgresBackend(
            dsn=postgres_dsn,
            embedding_dimension=cfg.embedding_dimension,
            embedding_model=cfg.embedding_model,
        )

    resolved_path = palace_path or cfg.palace_path
    return ChromaBackend(resolved_path)


def open_collection(
    palace_path: Optional[str] = None,
    collection_name: Optional[str] = None,
    create: bool = False,
    backend: Optional[str] = None,
    dsn: Optional[str] = None,
):
    cfg = MempalaceConfig()
    backend_impl = _resolve_backend(palace_path=palace_path, backend=backend, dsn=dsn)
    return backend_impl.get_collection(collection_name or cfg.collection_name, create=create)
