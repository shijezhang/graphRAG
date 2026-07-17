"""Unit tests for config system: env loading, SecretStr, validate=False."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import Settings, _resolve_env_vars


class TestResolveEnvVars:
    def test_string_interpolation(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "hello")
        assert _resolve_env_vars("${MY_VAR}") == "hello"

    def test_missing_env_var_returns_empty_string(self, monkeypatch):
        monkeypatch.delenv("NO_SUCH_VAR", raising=False)
        assert _resolve_env_vars("${NO_SUCH_VAR}") == ""

    def test_nested_dict_interpolation(self, monkeypatch):
        monkeypatch.setenv("KEY_VAR", "secret-value")
        result = _resolve_env_vars({"llm": {"api_key": "${KEY_VAR}"}})
        assert result == {"llm": {"api_key": "secret-value"}}

    def test_non_env_string_unchanged(self):
        assert _resolve_env_vars("just a string") == "just a string"

    def test_list_values_interpolated(self, monkeypatch):
        monkeypatch.setenv("X", "val")
        assert _resolve_env_vars(["${X}", "literal"]) == ["val", "literal"]


class TestSettingsValidation:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(ValidationError):
            Settings(llm={"api_key": ""})

    def test_valid_with_api_key(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
        s = Settings()
        assert s.llm.api_key.get_secret_value() == "sk-test-key"

    def test_from_yaml_validate_false_no_key_required(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        # Should NOT raise even without an API key
        s = Settings.from_yaml("configs/default.yaml", validate=False)
        assert s is not None

    def test_secret_str_not_leaked_in_repr(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-very-secret")
        s = Settings()
        # SecretStr must not appear in repr/str
        assert "sk-very-secret" not in repr(s)
        assert "sk-very-secret" not in str(s)


class TestPathsConfig:
    def test_computed_properties(self, settings_no_validate):
        p = settings_no_validate.paths
        assert p.chunks_file.name == "chunks.json"
        assert p.graph_file.name == "knowledge_graph.json"
        assert p.communities_file.name == "communities.json"
        assert ".llm_cache" in str(p.llm_cache_dir)


class TestServerConfig:
    def test_auth_none_when_no_credentials(self, settings_no_validate):
        s = settings_no_validate.server
        assert s.auth is None

    def test_default_host_is_localhost(self, settings_no_validate):
        assert settings_no_validate.server.host == "127.0.0.1"
