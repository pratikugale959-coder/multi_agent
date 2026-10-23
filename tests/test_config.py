"""Tests for configuration loading and secret handling."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from code_helper.config import (
    ConfigError,
    describe_configuration,
    get_settings,
    require_nvidia_credentials,
    require_pinecone_credentials,
    reset_settings_cache,
)
from tests.conftest import make_settings


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_settings_cache()
    yield
    reset_settings_cache()


def test_missing_nvidia_key_raises(tmp_path: Path) -> None:
    settings = replace(make_settings(tmp_path), nvidia_api_key="")
    with pytest.raises(ConfigError, match="NVIDIA"):
        require_nvidia_credentials(settings)


def test_present_key_is_returned(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    assert require_nvidia_credentials(settings) == "test-nvidia-key"
    assert require_pinecone_credentials(settings) == "test-pinecone-key"


def test_missing_pinecone_key_raises(tmp_path: Path) -> None:
    settings = replace(make_settings(tmp_path), pinecone_api_key="")
    with pytest.raises(ConfigError, match="Pinecone"):
        require_pinecone_credentials(settings)


def test_settings_read_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "env-key")
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "env-key")
    monkeypatch.setenv("CHUNK_SIZE", "77")
    monkeypatch.setenv("TARGET_REPO_PATH", str(tmp_path))
    reset_settings_cache()
    settings = get_settings()
    assert settings.nvidia_api_key == "env-key"
    assert settings.chunk_size == 77
    assert settings.target_repo_path == tmp_path.resolve()


def test_invalid_integer_environment_variable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOP_K", "not-a-number")
    reset_settings_cache()
    with pytest.raises(ConfigError):
        get_settings()


def test_run_limits_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_LLM_CALLS", "7")
    monkeypatch.setenv("MAX_TOOL_CALLS", "2")
    reset_settings_cache()
    settings = get_settings()
    assert settings.max_llm_calls == 7
    assert settings.max_tool_calls == 2


def test_max_output_tokens_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_MAX_TOKENS", "321")
    reset_settings_cache()
    assert get_settings().max_output_tokens == 321


def test_chat_timeout_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_TIMEOUT_SECONDS", "12")
    reset_settings_cache()
    assert get_settings().chat_timeout_seconds == 12


def test_litellm_model_ids_use_nvidia_nim_prefix(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    assert settings.litellm_chat_model == "nvidia_nim/meta/llama-3.2-11b-vision-instruct"
    assert settings.litellm_embed_model == "nvidia_nim/nvidia/nemotron-3-embed-1b"


def test_describe_configuration_never_leaks_secret_values() -> None:
    info = describe_configuration()
    assert isinstance(info["nvidia_key_present"], bool)
    assert isinstance(info["pinecone_key_present"], bool)

    rendered = json.dumps(info, default=str)
    settings = get_settings()
    if settings.nvidia_api_key:
        assert settings.nvidia_api_key not in rendered
    if settings.pinecone_api_key:
        assert settings.pinecone_api_key not in rendered
