"""Basic usage example: loading config, building a repository, and adding documents.

This script demonstrates the core workflow without LLM or embeddings.
It focuses on configuration and document storage.
"""


from src.config import load_config
from src.models import Document
from src.repository.factory import build_repository


def main():
    """Load config, build repository, add documents, and retrieve."""
    cfg = load_config()
    print(f"Backend: {cfg.backend}")
    print(f"Mode: {cfg.mode}")
    print()

    repo = build_repository(cfg)
    print(f"Repository: {type(repo).__name__}")
    print()

    docs = [
        Document(
            doc_id="sample::pkg1::abc123",
            text="import os; os.system('malware')",
            metadata={"source": "pypi_sample", "type": "MaliciousCode", "package": "pkg1"},
            label="malicious",
        ),
        Document(
            doc_id="sample::pkg2::def456",
            text="def hello():\n    print('hello')",
            metadata={"source": "pypi_sample", "type": "BenignCode", "package": "pkg2"},
            label="benign",
        ),
    ]

    print("Adding documents...")
    for doc in docs:
        repo.add(doc)
        print(f"  + {doc.doc_id}: {doc.preview()}")

    print()
    print(f"Total documents: {repo.count()}")
    print()

    print("Retrieving all documents:")
    for doc in repo.all():
        print(f"  - {doc.doc_id} [{doc.label}]")

    print()
    print("Searching for 'malware':")
    results = repo.search("malware", top_k=5)
    for doc in results:
        print(f"  - {doc.doc_id}: {doc.preview()}")


if __name__ == "__main__":
    main()
