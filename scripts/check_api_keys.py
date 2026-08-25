#!/usr/bin/env python3
"""Check if all required API keys are configured."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_concierge.core.config import settings


def check_api_keys() -> None:
    """Check and report on API key configuration."""
    print("=" * 60)
    print("API Key Configuration Check")
    print("=" * 60)
    print()

    issues = []

    # Check Anthropic API key
    anthropic_key = settings.anthropic_api_key.get_secret_value()
    if not anthropic_key or anthropic_key.startswith("your-"):
        print("❌ ANTHROPIC_API_KEY: Not configured")
        issues.append("Anthropic")
    else:
        print(f"✅ ANTHROPIC_API_KEY: Configured ({anthropic_key[:10]}...)")

    # Check Data Commons API key
    dc_key = settings.data_commons_api_key.get_secret_value()
    if not dc_key or dc_key == "your-data-commons-api-key-here":
        print("❌ DATA_COMMONS_API_KEY: Not configured")
        issues.append("Data Commons")
    else:
        print(f"✅ DATA_COMMONS_API_KEY: Configured ({dc_key[:10]}...)")

    # Check BLS API key
    bls_key = settings.bls_api_key.get_secret_value()
    if not bls_key or bls_key.startswith("your-"):
        print("⚠️  BLS_API_KEY: Not configured (optional, limits apply)")
    else:
        print(f"✅ BLS_API_KEY: Configured ({bls_key[:10]}...)")

    # Check Census API key
    census_key = settings.census_api_key.get_secret_value()
    if not census_key or census_key.startswith("your-"):
        print("⚠️  CENSUS_API_KEY: Not configured (optional, limits apply)")
    else:
        print(f"✅ CENSUS_API_KEY: Configured ({census_key[:10]}...)")

    print()
    print("=" * 60)

    if issues:
        print()
        print("⚠️  MISSING REQUIRED API KEYS:")
        print()

        if "Anthropic" in issues:
            print("📝 Anthropic Claude API:")
            print("   1. Visit: https://console.anthropic.com/")
            print("   2. Sign up or log in")
            print("   3. Go to API Keys section")
            print("   4. Create a new API key")
            print("   5. Add to .env: ANTHROPIC_API_KEY=sk-ant-...")
            print()

        if "Data Commons" in issues:
            print("📝 Data Commons API:")
            print("   1. Visit: https://apikeys.datacommons.org/")
            print("   2. Sign in with Google account")
            print("   3. Create a new API key")
            print("   4. Add to .env: DATA_COMMONS_API_KEY=<your-key>")
            print()

        print("After adding keys, restart the application.")
        print()
        sys.exit(1)
    else:
        print("✅ All required API keys are configured!")
        print()
        sys.exit(0)


if __name__ == "__main__":
    check_api_keys()
