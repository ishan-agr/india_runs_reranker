"""
Safe Anthropic API-key loader (Phase 2, offline only).

Resolution order:
  1. env var ANTHROPIC_API_KEY
  2. redrob_ranker/.secrets/anthropic_key.txt   (gitignored)

The key is NEVER printed, logged, or written to any output. `masked()` is the only
thing allowed near logs. This module is named 'keyloader' (NOT 'secrets') to avoid
shadowing Python's stdlib `secrets`.
"""
import os

KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".secrets", "anthropic_key.txt")


def get_api_key():
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k and k.strip():
        return k.strip()
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "r", encoding="utf-8") as f:
            k = f.read().strip()
        if k:
            return k
    raise SystemExit(
        "No Anthropic API key found.\n"
        "  Option A: create redrob_ranker/.secrets/anthropic_key.txt containing only the key.\n"
        "  Option B: set a user env var ANTHROPIC_API_KEY.\n"
        "Do NOT paste the key into chat. The .secrets/ folder is gitignored."
    )


def masked(k):
    return (k[:6] + "..." + k[-4:]) if k and len(k) > 12 else "****"
