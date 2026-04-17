from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError, BadRequestError

from app.models.vlm.providers.qwen_vllm.provider import QwenVLLMProvider


def _make_api_connection_error() -> APIConnectionError:
    return APIConnectionError(request=httpx.Request("POST", "http://fake"))


def _make_response(text: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    return response


def test_generate_returns_content_on_success():
    provider = QwenVLLMProvider(
        base_url="http://fake/v1",
        api_key="none",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
    )
    with patch.object(
        provider._client.chat.completions,
        "create",
        return_value=_make_response("hello"),
    ) as mock_create:
        result = provider.generate(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
            temperature=0,
        )
    assert result == "hello"
    assert mock_create.call_count == 1


def test_generate_retries_once_on_connection_error_then_succeeds():
    provider = QwenVLLMProvider(
        base_url="http://fake/v1",
        api_key="none",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
    )
    with patch.object(
        provider._client.chat.completions,
        "create",
        side_effect=[_make_api_connection_error(), _make_response("ok")],
    ) as mock_create, patch(
        "app.models.vlm.providers.qwen_vllm.provider.time.sleep"
    ):
        result = provider.generate(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
            temperature=0,
        )
    assert result == "ok"
    assert mock_create.call_count == 2


def test_generate_raises_connection_error_after_retry_exhaustion():
    provider = QwenVLLMProvider(
        base_url="http://fake/v1",
        api_key="none",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
    )
    with patch.object(
        provider._client.chat.completions,
        "create",
        side_effect=[
            _make_api_connection_error(),
            _make_api_connection_error(),
        ],
    ), patch("app.models.vlm.providers.qwen_vllm.provider.time.sleep"):
        with pytest.raises(ConnectionError) as excinfo:
            provider.generate(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10,
                temperature=0,
            )
    assert "Qwen/Qwen3-VL-8B-Instruct" in str(excinfo.value)
    assert "http://fake/v1" in str(excinfo.value)


def test_generate_applies_transforms_before_call():
    provider = QwenVLLMProvider(
        base_url="http://fake/v1",
        api_key="none",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        min_pixels=111,
        max_pixels=222,
    )
    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _make_response("ok")

    with patch.object(
        provider._client.chat.completions, "create", side_effect=fake_create
    ):
        provider.generate(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "x"}},
                    {"type": "text", "text": "see <image> this"},
                ],
            }],
            max_tokens=10,
            temperature=0,
        )

    sent = captured["messages"]
    text_part = sent[0]["content"][1]
    img_part = sent[0]["content"][0]
    assert text_part["text"] == "see  this"
    assert img_part["image_url"]["min_pixels"] == 111
    assert img_part["image_url"]["max_pixels"] == 222


def test_generate_does_not_catch_bad_request_error():
    provider = QwenVLLMProvider(
        base_url="http://fake/v1",
        api_key="none",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
    )
    err = BadRequestError(
        message="bad",
        response=httpx.Response(400, request=httpx.Request("POST", "http://fake")),
        body=None,
    )
    with patch.object(
        provider._client.chat.completions, "create", side_effect=err
    ):
        with pytest.raises(BadRequestError):
            provider.generate(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10,
                temperature=0,
            )
