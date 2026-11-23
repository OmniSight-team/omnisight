FROM python:3.13-slim

RUN pip install --no-cache-dir uv

WORKDIR /workspace

# The whole workspace, so uv can resolve omnisight-pfm and omnisight-common together.
COPY pyproject.toml uv.lock ./
COPY services ./services

RUN uv sync --frozen --package omnisight-pfm

EXPOSE 8000

# The package is `pfm`, not `app`, so its tests can share one pytest run with the other services'.
CMD ["uv", "run", "--package", "omnisight-pfm", "uvicorn", "pfm.main:app", \
     "--app-dir", "services/pfm", "--host", "0.0.0.0", "--port", "8000", "--reload"]
