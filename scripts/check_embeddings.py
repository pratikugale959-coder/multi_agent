"""Integration check: verify the live NVIDIA NIM embedding API.

Run from the directory containing ``code_helper``::

    python scripts/check_embeddings.py

Requires a valid ``NVIDIA_API_KEY`` / ``NVIDIA_NIM_API_KEY``. This makes a real
network call; it is **not** part of the offline pytest suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_helper.config import ConfigError, describe_configuration  # noqa: E402
from code_helper.embeddings import EmbeddingClient, EmbeddingError  # noqa: E402


def main() -> int:
    """Probe the embedding model and report the verified dimension."""
    info = describe_configuration()
    print("Configured embedding model:", info["litellm_embed_model"])
    print("Configured dimension      :", info["embedding_dimension"])
    print("NVIDIA key present        :", info["nvidia_key_present"])

    # Build the client without dimension validation so we can measure the truth.
    from dataclasses import replace

    from code_helper.config import get_settings

    settings = replace(get_settings(), embedding_dimension=0)
    client = EmbeddingClient(settings)

    try:
        vector = client.embed_query("Where is the upsert function implemented?")
    except (EmbeddingError, ConfigError) as exc:
        print(f"EMBEDDING CHECK FAILED: {exc}")
        return 1

    print(f"Verified dimension        : {len(vector)}")
    print(f"First coordinates         : {[round(v, 4) for v in vector[:4]]}")
    if len(vector) != info["embedding_dimension"]:
        print(
            "WARNING: live dimension does not match EMBEDDING_DIMENSION "
            f"({len(vector)} != {info['embedding_dimension']})."
        )
        return 2
    print("EMBEDDING CHECK PASSED")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
