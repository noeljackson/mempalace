"""
test_postgres.py — Integration tests for the Postgres + pgvector backend.

Requires a running PostgreSQL instance with pgvector.  Skipped automatically
when the MEMPALACE_TEST_POSTGRES_DSN environment variable is not set.
"""

import os

import pytest

from mempalace.storage import (
    PostgresCollectionAdapter,
    PostgresBackend,
    open_collection,
    migrate_collection,
)

POSTGRES_DSN = os.environ.get("MEMPALACE_TEST_POSTGRES_DSN")

pytestmark = pytest.mark.skipif(
    not POSTGRES_DSN,
    reason="MEMPALACE_TEST_POSTGRES_DSN not set — skipping Postgres tests",
)


# ── helpers ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def _ensure_schema():
    """Create the table once at session start so cleanup fixtures work."""
    PostgresCollectionAdapter(dsn=POSTGRES_DSN, collection_name="_setup")


def _clean_table(dsn: str):
    """Truncate the shared table so each test starts fresh."""
    import psycopg

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM mempalace_documents")
        conn.commit()


@pytest.fixture(autouse=True)
def _fresh_table():
    """Ensure the table is empty before every test."""
    _clean_table(POSTGRES_DSN)
    yield
    _clean_table(POSTGRES_DSN)


@pytest.fixture
def col():
    """A PostgresCollectionAdapter pointed at the test database."""
    return PostgresCollectionAdapter(
        dsn=POSTGRES_DSN,
        collection_name="test_col",
    )


@pytest.fixture
def seeded_col(col):
    """Collection pre-loaded with four representative documents."""
    col.add(
        ids=["d1", "d2", "d3", "d4"],
        documents=[
            "The authentication module uses JWT tokens for session management.",
            "Database migrations are handled by Alembic with PostgreSQL 15.",
            "The React frontend uses TanStack Query for state management.",
            "Sprint planning: migrate auth to passkeys by Q3.",
        ],
        metadatas=[
            {"wing": "project", "room": "backend", "source_file": "auth.py"},
            {"wing": "project", "room": "backend", "source_file": "db.py"},
            {"wing": "project", "room": "frontend", "source_file": "App.tsx"},
            {"wing": "notes", "room": "planning", "source_file": "sprint.md"},
        ],
    )
    return col


# ── schema ───────────────────────────────────────────────────────────

class TestSchema:
    def test_ensure_schema_creates_table_and_indexes(self):
        adapter = PostgresCollectionAdapter(dsn=POSTGRES_DSN, collection_name="schema_test")
        assert adapter.count() == 0

    def test_ensure_schema_is_idempotent(self):
        PostgresCollectionAdapter(dsn=POSTGRES_DSN, collection_name="idem_a")
        PostgresCollectionAdapter(dsn=POSTGRES_DSN, collection_name="idem_a")


# ── count ────────────────────────────────────────────────────────────

class TestCount:
    def test_empty_collection(self, col):
        assert col.count() == 0

    def test_after_add(self, seeded_col):
        assert seeded_col.count() == 4


# ── add / get ────────────────────────────────────────────────────────

class TestAddAndGet:
    def test_get_all(self, seeded_col):
        result = seeded_col.get(include=["documents", "metadatas"])
        assert sorted(result["ids"]) == ["d1", "d2", "d3", "d4"]
        assert len(result["documents"]) == 4
        assert all(isinstance(m, dict) for m in result["metadatas"])

    def test_get_by_ids(self, seeded_col):
        result = seeded_col.get(ids=["d2", "d4"], include=["documents"])
        assert sorted(result["ids"]) == ["d2", "d4"]
        assert len(result["documents"]) == 2

    def test_get_with_limit_and_offset(self, seeded_col):
        page1 = seeded_col.get(limit=2, offset=0, include=["documents"])
        page2 = seeded_col.get(limit=2, offset=2, include=["documents"])
        all_ids = page1["ids"] + page2["ids"]
        assert sorted(all_ids) == ["d1", "d2", "d3", "d4"]

    def test_get_with_where(self, seeded_col):
        result = seeded_col.get(
            where={"wing": "project", "room": "backend"},
            include=["documents"],
        )
        assert sorted(result["ids"]) == ["d1", "d2"]

    def test_get_with_where_and_dollar_and(self, seeded_col):
        result = seeded_col.get(
            where={"$and": [{"wing": "project"}, {"room": "frontend"}]},
            include=["documents"],
        )
        assert result["ids"] == ["d3"]

    def test_get_without_include_omits_docs_and_meta(self, seeded_col):
        result = seeded_col.get()
        assert "documents" not in result
        assert "metadatas" not in result
        assert len(result["ids"]) == 4


# ── upsert ───────────────────────────────────────────────────────────

class TestUpsert:
    def test_upsert_inserts_new(self, col):
        col.upsert(
            ids=["u1"],
            documents=["brand new document"],
            metadatas=[{"wing": "notes"}],
        )
        assert col.count() == 1
        result = col.get(ids=["u1"], include=["documents", "metadatas"])
        assert result["documents"] == ["brand new document"]
        assert result["metadatas"][0]["wing"] == "notes"

    def test_upsert_updates_existing(self, seeded_col):
        seeded_col.upsert(
            ids=["d1"],
            documents=["updated auth doc"],
            metadatas=[{"wing": "project", "room": "backend", "version": "2"}],
        )
        assert seeded_col.count() == 4  # no new row
        result = seeded_col.get(ids=["d1"], include=["documents", "metadatas"])
        assert result["documents"] == ["updated auth doc"]
        assert result["metadatas"][0]["version"] == "2"


# ── delete ───────────────────────────────────────────────────────────

class TestDelete:
    def test_delete_specific_ids(self, seeded_col):
        seeded_col.delete(ids=["d1", "d3"])
        assert seeded_col.count() == 2
        result = seeded_col.get(include=["documents"])
        assert sorted(result["ids"]) == ["d2", "d4"]

    def test_delete_nonexistent_is_noop(self, col):
        col.delete(ids=["ghost"])  # should not raise
        assert col.count() == 0


# ── query (vector search) ───────────────────────────────────────────

class TestQuery:
    def test_query_returns_relevant_results(self, seeded_col):
        result = seeded_col.query(
            query_texts=["authentication tokens"],
            n_results=2,
            include=["documents", "distances"],
        )
        assert len(result["ids"]) == 1  # one query text
        assert len(result["ids"][0]) == 2
        # The auth document should be the top hit
        assert "d1" in result["ids"][0]
        assert len(result["distances"][0]) == 2
        # Distances should be non-negative floats (cosine)
        assert all(d >= 0 for d in result["distances"][0])

    def test_query_with_where_filter(self, seeded_col):
        result = seeded_col.query(
            query_texts=["database migration"],
            n_results=10,
            where={"room": "backend"},
            include=["documents", "metadatas"],
        )
        # Only backend docs should appear
        for meta in result["metadatas"][0]:
            assert meta["room"] == "backend"

    def test_query_multiple_texts(self, seeded_col):
        result = seeded_col.query(
            query_texts=["auth tokens", "React frontend"],
            n_results=1,
            include=["documents"],
        )
        assert len(result["ids"]) == 2  # two query texts
        assert len(result["ids"][0]) == 1
        assert len(result["ids"][1]) == 1

    def test_query_empty_collection(self, col):
        result = col.query(
            query_texts=["anything"],
            n_results=5,
            include=["documents"],
        )
        assert result["ids"] == [[]]


# ── collection isolation ─────────────────────────────────────────────

class TestCollectionIsolation:
    def test_different_collections_are_isolated(self):
        col_a = PostgresCollectionAdapter(dsn=POSTGRES_DSN, collection_name="iso_a")
        col_b = PostgresCollectionAdapter(dsn=POSTGRES_DSN, collection_name="iso_b")

        col_a.add(
            ids=["x1"],
            documents=["doc for A"],
            metadatas=[{"src": "a"}],
        )
        col_b.add(
            ids=["x2"],
            documents=["doc for B"],
            metadatas=[{"src": "b"}],
        )

        assert col_a.count() == 1
        assert col_b.count() == 1
        assert col_a.get(include=["documents"])["ids"] == ["x1"]
        assert col_b.get(include=["documents"])["ids"] == ["x2"]


# ── PostgresBackend factory ──────────────────────────────────────────

class TestPostgresBackend:
    def test_get_collection_returns_adapter(self):
        backend = PostgresBackend(dsn=POSTGRES_DSN)
        col = backend.get_collection("backend_test")
        assert isinstance(col, PostgresCollectionAdapter)
        assert col.count() == 0


# ── open_collection integration ──────────────────────────────────────

class TestOpenCollection:
    def test_open_collection_with_postgres_dsn(self):
        col = open_collection(
            backend="postgres",
            dsn=POSTGRES_DSN,
            collection_name="open_col_test",
        )
        assert isinstance(col, PostgresCollectionAdapter)
        assert col.count() == 0


# ── migration from Chroma to Postgres ────────────────────────────────

class TestMigration:
    def test_migrate_chroma_to_postgres(self, seeded_collection):
        """Migrate from the Chroma seeded_collection fixture to Postgres."""
        target = PostgresCollectionAdapter(
            dsn=POSTGRES_DSN,
            collection_name="migration_target",
        )
        result = migrate_collection(seeded_collection, target, batch_size=2)
        assert result["scanned"] == 4
        assert result["written"] == 4
        assert target.count() == 4

        # Verify docs survived the round-trip
        rows = target.get(include=["documents", "metadatas"])
        assert len(rows["ids"]) == 4
        assert all(isinstance(m, dict) for m in rows["metadatas"])
