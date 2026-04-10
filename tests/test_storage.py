from mempalace.storage import iter_collection_batches, migrate_collection


class FakeSourceCollection:
    def __init__(self):
        self.rows = [
            ("a", "doc-a", {"wing": "project", "room": "backend"}),
            ("b", "doc-b", {"wing": "project", "room": "frontend"}),
            ("c", "doc-c", {"wing": "notes", "room": "planning"}),
        ]

    def get(self, limit=None, offset=None, where=None, include=None):
        del where, include
        start = offset or 0
        end = start + (limit or len(self.rows))
        rows = self.rows[start:end]
        return {
            "ids": [row[0] for row in rows],
            "documents": [row[1] for row in rows],
            "metadatas": [row[2] for row in rows],
        }


class FakeTargetCollection:
    def __init__(self):
        self.upserts = []

    def upsert(self, ids, documents, metadatas):
        self.upserts.append(
            {
                "ids": list(ids),
                "documents": list(documents),
                "metadatas": list(metadatas),
            }
        )


def test_iter_collection_batches_scans_in_batches():
    source = FakeSourceCollection()

    batches = list(
        iter_collection_batches(
            source,
            batch_size=2,
            include=["documents", "metadatas"],
        )
    )

    assert len(batches) == 2
    assert batches[0]["ids"] == ["a", "b"]
    assert batches[1]["ids"] == ["c"]


def test_migrate_collection_copies_all_rows():
    source = FakeSourceCollection()
    target = FakeTargetCollection()

    result = migrate_collection(source, target, batch_size=2)

    assert result == {"scanned": 3, "written": 3}
    assert len(target.upserts) == 2
    assert target.upserts[0]["ids"] == ["a", "b"]
    assert target.upserts[1]["ids"] == ["c"]
    assert target.upserts[0]["documents"] == ["doc-a", "doc-b"]
