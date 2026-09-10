# Data Concierge - Production Dockerfile
# Optimized for Google Cloud Run

FROM python:3.14-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# qsv — the data profiler behind portal onboarding (stats, frequency,
# describegpt, count). Without it, admin-triggered onboarding downloads CSVs
# and produces an index with no column metadata.
#
# TARGETARCH is supplied automatically by BuildKit. The release ships one zip
# per architecture holding several build variants; only the full "qsv" binary
# has describegpt, so that is the one extracted (qsvlite omits it).
ARG QSV_VERSION=22.0.1
ARG TARGETARCH=amd64
RUN set -eux; \
    case "${TARGETARCH}" in \
      amd64) QSV_TARGET="x86_64-unknown-linux-musl" ;; \
      arm64) QSV_TARGET="aarch64-unknown-linux-gnu" ;; \
      *) echo "No qsv build for TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    curl -fsSL -o /tmp/qsv.zip \
      "https://github.com/dathere/qsv/releases/download/${QSV_VERSION}/qsv-${QSV_VERSION}-${QSV_TARGET}.zip"; \
    unzip -j /tmp/qsv.zip qsv -d /usr/local/bin; \
    rm /tmp/qsv.zip; \
    chmod +x /usr/local/bin/qsv; \
    /usr/local/bin/qsv --version

# Set working directory
WORKDIR /app

# Copy all necessary files for pip install
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

# Production stage
FROM python:3.14-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# NOTE: qsv arrives with the /usr/local/bin copy above — a second explicit
# COPY would duplicate the 40 MB binary into another layer.

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser configs/ ./configs/

# Onboarding scripts — the admin panel spawns these as child processes, so
# they must exist in the image (gateway/onboarding_jobs.scripts_dir looks for
# /app/scripts). Without them the Data Portals pane can only report that
# onboarding is unavailable.
COPY --chown=appuser:appuser scripts/ ./scripts/

# Copy .env file if it exists (for local builds - Cloud Run uses env vars)
COPY --chown=appuser:appuser .env* ./

# Create directories for runtime data
RUN mkdir -p /app/notebooks /app/data/submissions /app/data/verified_notebooks \
    /app/data/ckan_onboard /app/data/dcat_onboard && \
    chown -R appuser:appuser /app

# Set environment variables
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Cloud Run sets PORT automatically
    PORT=8080 \
    HOST=0.0.0.0

# Switch to non-root user
USER appuser

# Expose port (Cloud Run uses PORT env var)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the web server
CMD ["python", "-m", "data_concierge.ui.web"]
