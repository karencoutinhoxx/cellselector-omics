FROM python:3.11-slim

WORKDIR /app

# All packages in requirements.txt (pandas, pyarrow, numpy, scipy, neo4j,
# fastapi, ...) ship manylinux wheels for 3.11 — no C compiler needed to
# install them. NOT verified by an actual `docker build` in this
# environment (no Docker daemon available here); if a future build hits a
# "Microsoft Visual C++ ... required" / gcc-not-found style error for some
# package, uncomment the block below and rebuild.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the precomputed weights file exists in the image (committed to the
# repo — see api/main.py::_warm_learned_weights). Without it, startup falls
# back to an ~8 min live SLSQP recompute rather than failing outright, but
# that's the wrong thing to discover in production, so fail the build.
RUN test -f outputs/production_weights.json || \
    (echo "ERROR: outputs/production_weights.json missing — regenerate it \
    (see models/classical/weights_learned.py::optimise_weights_by_class \
    and the regeneration command in api/main.py) and commit it before \
    building." && exit 1)

EXPOSE 8001

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
