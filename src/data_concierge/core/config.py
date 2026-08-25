"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = " AI Data Concierge"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # LLM Configuration
    # claude-sonnet-5 is the latest Sonnet. Two API differences from the 4.x
    # family matter here: sampling params (temperature/top_p/top_k) are
    # rejected with a 400 — no call in this codebase may send them with this
    # model — and adaptive thinking is on by default, so thinking tokens
    # count against max_tokens (hence the higher output cap).
    anthropic_api_key: SecretStr = Field(default=SecretStr(""))
    llm_model: str = "claude-sonnet-5"
    llm_fallback_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 8192

    # Typed Standards evidence signing (datHere content profile).
    # OFF by default: packages build unsigned ("dev-unsigned") until a key is
    # provisioned. When enabled, every published package is Ed25519ph-signed
    # (RFC 8032) and the public key is served in the trust registry at
    # /.well-known/typed-publisher.json. ``evidence_signing_key_seed`` is the
    # 64-hex-char Ed25519 private seed; custody (env / Secret Manager / KMS) is
    # the deployer's choice — this only reads the resolved value.
    evidence_signing_enabled: bool = False
    evidence_signing_key_seed: SecretStr = Field(default=SecretStr(""))
    evidence_signing_kid: str = "dathere:data-concierge-2026"
    evidence_host: str = "data-concierge.dathere.com"
    evidence_trust_registry_url: str = ""  # derived from evidence_host when empty
    evidence_signer_display_name: str = "datHere Data Concierge"

    # Best-effort attestation legs (spec §8.3.2). Each is independent and OFF by
    # default; failures are swallowed so they never block publishing.
    evidence_timestamp_enabled: bool = False  # RFC 3161 trusted timestamp
    evidence_freetsa_url: str = "https://freetsa.org/tsr"
    evidence_rekor_enabled: bool = False  # Sigstore Rekor transparency log
    evidence_rekor_url: str = "https://rekor.sigstore.dev"
    # When true, approving a notebook embeds the commitment view into the
    # published .ipynb and commits the canonical evidence package JSON beside
    # it. OFF by default — this mutates the public GitHub artifact, so it is
    # gated separately from building/previewing packages.
    evidence_embed_enabled: bool = False

    # Verified-answer matching (two-stage: keyword candidates + LLM gate)
    # A low-latency model judges whether a verified answer truly answers a new
    # query, catching temporal/geographic/variable mismatches that keyword
    # overlap alone lets through (see issue #64).
    verified_match_llm_enabled: bool = True
    verified_match_model: str = "claude-haiku-4-5"
    # Stage 1 keyword retrieval threshold — kept low to cast a wide net of
    # candidates for the LLM gate to adjudicate.
    verified_match_candidate_threshold: float = 0.30
    # Max keyword candidates passed to the LLM gate per query.
    verified_match_max_candidates: int = 3
    # Minimum LLM-reported confidence required to accept a match.
    verified_match_confidence_threshold: float = 0.80
    # Keyword-only fallback threshold used when the LLM gate is unavailable
    # (API error or circuit breaker open) — stricter than normal to reduce
    # false positives without the semantic check.
    verified_match_fallback_threshold: float = 0.75
    # Consecutive LLM failures before the circuit breaker opens and the system
    # falls back to keyword-only matching.
    verified_match_circuit_breaker_threshold: int = 3

    # Notebook verification (#131) — execute each generated notebook and check
    # its output re-derives the answer's numbers.
    #
    # ON. It executes LLM-generated code, so it is only safe because of the
    # containment around it: a subprocess with a hard timeout, a minimal env
    # allowlist that withholds every credential, shell-escape cells skipped,
    # and a sitecustomize egress guard that blocks loopback / private /
    # link-local / metadata addresses inside the kernel — so a notebook can
    # still reach census.gov but cannot read 169.254.169.254 to steal the
    # instance service-account token.
    #
    # Cost is one notebook execution per query, capped at two concurrent runs.
    # Set false to disable; the confidence factor is then omitted entirely
    # rather than shown as perpetually "pending".
    notebook_verification_enabled: bool = True
    notebook_verification_timeout_seconds: int = 180
    notebook_verification_cell_timeout_seconds: int = 60

    # Notebook review (third signal) — an adversarial, static
    # static review of each generated notebook's method: does the code
    # actually derive the numbers the answer claims, are the datasets and
    # citations sound, is anything hardcoded that should be computed. Runs
    # alongside execution in the async verification pass and merges into the
    # same ``notebook_verification`` confidence factor. When it cannot run it
    # reports unavailable rather than dragging the score (issue #132 rules).
    # ``notebook_review_model`` empty means "use ``llm_model``".
    notebook_review_enabled: bool = True
    notebook_review_model: str = ""
    notebook_review_timeout_seconds: int = 120

    # Follow-up understanding for chat (notebook revisions). A low-latency
    # model classifies whether a message in an ongoing chat is a brand-new
    # question or a request to revise the previous answer's notebook, and
    # rewrites context-dependent follow-ups ("what about Ohio?") into
    # self-contained queries. Fails open: when unavailable, every message is
    # treated as a new question, which is exactly today's behaviour.
    followup_llm_enabled: bool = True
    followup_model: str = "claude-haiku-4-5"

    # MCP containment (#135). MCP server URLs are rejected when they point at
    # loopback, private, link-local or cloud-metadata addresses — otherwise an
    # admin-registered server turns the service into an SSRF proxy against its
    # own VPC. Local development legitimately runs MCP servers on localhost,
    # so this opts out; leave it false anywhere reachable.
    mcp_allow_private_urls: bool = False

    # Cache (Redis)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: SecretStr = Field(default=SecretStr(""))
    cache_ttl_default: int = 86400  # 24 hours

    # Data Commons
    data_commons_api_url: str = "https://api.datacommons.org"
    data_commons_api_key: SecretStr = Field(default=SecretStr(""))

    # CKAN Configuration
    ckan_url: str = "https://data.dathere.com"
    # Provide via env / .env only — never commit a real key (issue #91).
    ckan_api_key: SecretStr = Field(default=SecretStr(""))

    # WPRDC (Western PA Regional Data Center) - City of Pittsburgh open data
    wprdc_ckan_url: str = "https://data.wprdc.org"
    wprdc_organization: str = "city-of-pittsburgh"

    # OpenRouter (for qsv describegpt AI-enhanced data dictionaries)
    openrouter_api_key: SecretStr = Field(default=SecretStr(""))

    # Pinecone Vector Store (for CKAN semantic search)
    # Provide via env / .env only — never commit a real key (issue #91).
    pinecone_api_key: SecretStr = Field(default=SecretStr(""))
    pinecone_index_name: str = "ckan-dathere-index"
    pinecone_namespace: str = "ckan-namespace"

    # External APIs - Federal Data Sources
    bls_api_key: SecretStr = Field(default=SecretStr(""))
    census_api_key: SecretStr = Field(default=SecretStr(""))
    fred_api_key: SecretStr = Field(default=SecretStr(""))
    bea_api_key: SecretStr = Field(default=SecretStr(""))

    # MCP Server Directories
    census_mcp_dir: str = Field(
        default="", description="Path to Census MCP server project directory"
    )

    # Confidence Thresholds
    confidence_threshold_high: float = 0.85
    confidence_threshold_medium: float = 0.70
    confidence_threshold_low: float = 0.50

    # Agent Configuration
    max_retrieval_attempts: int = 2
    escalation_confidence_threshold: float = 0.50

    # Session Management
    session_timeout_minutes: int = 30
    max_sessions_per_user: int = 5

    # GitHub Notebook Publishing — env-var seeds for github_settings.json.
    # The admin panel writes to github_settings.json via storage; these
    # env vars supply the initial defaults on first load (when no saved
    # settings exist). Admin-panel edits override these values until
    # github_settings.json is cleared.
    github_token: SecretStr = Field(default=SecretStr(""))
    github_repo: str = "dathere/data-concierge-notebooks"
    github_branch: str = "main"
    github_drafts_folder: str = "drafts"
    github_verified_folder: str = "verified"
    github_verified_answers_folder: str = "verified-answers"
    # Shared secret for the inbound GitHub webhook. Recommended to set via
    # env (or secrets manager) on Cloud Run rather than the admin panel so
    # the value survives a re-provision of the storage backend.
    github_webhook_secret: SecretStr = Field(default=SecretStr(""))

    # Admin Email Notifications (SMTP)
    admin_notifications_enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: SecretStr = Field(default=SecretStr(""))
    smtp_use_tls: bool = True
    smtp_from_email: str = ""
    admin_notification_recipients: str = Field(
        default="",
        description="Comma-separated list of admin email addresses to notify",
    )
    app_base_url: str = Field(
        default="http://localhost:8080",
        description="Public base URL used to build links in admin emails",
    )

    # GitHub Sync (verified notebooks)
    github_sync_interval_hours: float = Field(
        default=4.0,
        description=(
            "How often to pull verified notebooks from GitHub in the background. "
            "Set to 0 to disable the background sync (lazy on-read sync still applies)."
        ),
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
