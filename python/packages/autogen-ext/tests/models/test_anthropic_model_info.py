"""Tests for Anthropic model info resolution, including Bedrock model IDs."""

import pytest

from autogen_ext.models.anthropic._model_info import _normalize_model_id, get_info, get_token_limit


class TestNormalizeModelId:
    """Tests for _normalize_model_id helper."""

    def test_standard_anthropic_id_unchanged(self) -> None:
        assert _normalize_model_id("claude-3-5-sonnet-20240620") == "claude-3-5-sonnet-20240620"

    def test_standard_bedrock_id(self) -> None:
        assert _normalize_model_id("anthropic.claude-3-5-sonnet-20240620-v1:0") == "claude-3-5-sonnet-20240620"

    def test_cross_region_us(self) -> None:
        assert _normalize_model_id("us.anthropic.claude-3-haiku-20240307-v1:0") == "claude-3-haiku-20240307"

    def test_cross_region_eu(self) -> None:
        assert _normalize_model_id("eu.anthropic.claude-3-opus-20240229-v1:0") == "claude-3-opus-20240229"

    def test_bedrock_v2_suffix(self) -> None:
        assert _normalize_model_id("anthropic.claude-3-5-sonnet-20240620-v2:0") == "claude-3-5-sonnet-20240620"

    def test_bedrock_no_version_suffix(self) -> None:
        # Some IDs may not have a version suffix
        assert _normalize_model_id("anthropic.claude-3-5-sonnet-20240620") == "claude-3-5-sonnet-20240620"

    def test_non_bedrock_passthrough(self) -> None:
        # A plain alias like "claude-3-7-sonnet-latest" should pass through unchanged
        assert _normalize_model_id("claude-3-7-sonnet-latest") == "claude-3-7-sonnet-latest"


class TestGetInfo:
    """Tests for get_info with Bedrock model IDs."""

    def test_standard_model(self) -> None:
        info = get_info("claude-3-5-sonnet-20240620")
        assert info["vision"] is True
        assert info["function_calling"] is True

    def test_bedrock_model(self) -> None:
        info = get_info("anthropic.claude-3-5-sonnet-20240620-v1:0")
        assert info["vision"] is True
        assert info["function_calling"] is True

    def test_cross_region_model(self) -> None:
        info = get_info("us.anthropic.claude-3-haiku-20240307-v1:0")
        assert info["vision"] is True
        assert info["function_calling"] is True

    def test_cross_region_eu_model(self) -> None:
        info = get_info("eu.anthropic.claude-3-opus-20240229-v1:0")
        assert info["vision"] is True

    def test_bedrock_and_standard_return_same_info(self) -> None:
        standard = get_info("claude-3-5-sonnet-20240620")
        bedrock = get_info("anthropic.claude-3-5-sonnet-20240620-v1:0")
        assert standard == bedrock

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(KeyError, match="not found"):
            get_info("some-unknown-model")


class TestGetTokenLimit:
    """Tests for get_token_limit with Bedrock model IDs."""

    def test_standard_model(self) -> None:
        assert get_token_limit("claude-3-5-sonnet-20240620") == 200000

    def test_bedrock_model(self) -> None:
        assert get_token_limit("anthropic.claude-3-5-sonnet-20240620-v1:0") == 200000

    def test_cross_region_model(self) -> None:
        assert get_token_limit("us.anthropic.claude-3-haiku-20240307-v1:0") == 200000

    def test_bedrock_and_standard_return_same_limit(self) -> None:
        assert get_token_limit("claude-3-5-sonnet-20240620") == get_token_limit(
            "anthropic.claude-3-5-sonnet-20240620-v1:0"
        )

    def test_unknown_model_returns_default(self) -> None:
        assert get_token_limit("some-unknown-model") == 100000
