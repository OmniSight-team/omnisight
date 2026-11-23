"""pfm's tests use the query tests' database, never the stack's (K-18).

`POSTGRES_DB` is pointed at `omnisight_test` before `pfm.settings` is imported. The database is
created and migrated by query's own conftest, run in a process of its own: its Alembic imports
query's `app`, which must not meet ingestion's `app` in this process.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "services" / "pfm"))
os.environ["POSTGRES_DB"] = "omnisight_test"

subprocess.run(
    ["uv", "run", "--package", "omnisight-query", "python", "-c",
     "import sys; sys.path.insert(0, 'services/query/tests'); import conftest"],
    cwd=ROOT, check=True, capture_output=True,
)
