"""The PFM service. In slice 9 it runs the rule detector (OMNI-908); slice 10 adds the model.

No host port (like ingestion): nothing outside the Docker network reaches it.
"""

import logging
import threading
from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from sqlalchemy import create_engine

from pfm.detector import Detector
from pfm.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

engine = create_engine(settings.database_url, pool_pre_ping=True)
detector = Detector(
    engine,
    lookback_s=settings.rules_lookback_s,
    max_gap_s=settings.rules_max_gap_s,
    quiet_s=settings.rules_quiet_s,
    max_history_s=settings.rules_max_history_s,
    tz=ZoneInfo(settings.rules_timezone),
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    stop = threading.Event()
    worker = threading.Thread(
        target=detector.run_forever, args=(settings.rules_interval_s, stop), name="rule-detector", daemon=True
    )
    worker.start()
    yield
    stop.set()
    worker.join(timeout=10)


app = FastAPI(title="omnisight-pfm", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if detector.last_error is None else "degraded",
        "rules": {
            "running": True,
            "interval_s": settings.rules_interval_s,
            "last_look_at": detector.last_look_at,
            "alerts_raised": detector.raised,
            "error": detector.last_error,
        },
        # Slice 10 (OMNI-910) loads the trained PFM here.
        "model": {"loaded": False, "reason": "no checkpoint yet: rules only"},
    }
