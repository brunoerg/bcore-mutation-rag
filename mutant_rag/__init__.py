"""Historical bug-based mutation testing RAG for Bitcoin Core."""

import os as _os
import sys as _sys

__version__ = "0.1.0"

# faiss-cpu and torch wheels each ship their own OpenMP runtime on macOS; when
# both are loaded with multiple OpenMP threads the process segfaults. A single
# OpenMP thread avoids it (and makes CPU embeddings deterministic). Must be set
# before either library is imported. Export OMP_NUM_THREADS to override.
if _sys.platform == "darwin":
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
