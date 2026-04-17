import copy

from app.models.vlm.providers.qwen_vllm.transforms import strip_image_tokens


def test_strip_image_tokens_removes_image_tag():
    msgs = [{"role": "user", "content": "Hello <image> world"}]
    out = strip_image_tokens(msgs)
    assert out[0]["content"] == "Hello  world"


def test_strip_image_tokens_removes_all_patterns():
    msgs = [{
        "role": "user",
        "content": "<image><|image_pad|><|vision_start|><|vision_end|>hi",
    }]
    out = strip_image_tokens(msgs)
    assert out[0]["content"] == "hi"


def test_strip_image_tokens_leaves_image_url_parts_untouched():
    msgs = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            {"type": "text", "text": "look <image>"},
        ],
    }]
    out = strip_image_tokens(msgs)
    assert out[0]["content"][0]["image_url"]["url"] == "data:image/png;base64,AAA"
    assert out[0]["content"][1]["text"] == "look "


def test_strip_image_tokens_leaves_non_user_turns_untouched():
    msgs = [
        {"role": "assistant", "content": "I see <image> there"},
        {"role": "system", "content": "<image>"},
    ]
    out = strip_image_tokens(msgs)
    assert out[0]["content"] == "I see <image> there"
    assert out[1]["content"] == "<image>"


def test_strip_image_tokens_idempotent_on_clean_text():
    msgs = [{"role": "user", "content": "Hello world"}]
    out = strip_image_tokens(msgs)
    assert out[0]["content"] == "Hello world"


def test_strip_image_tokens_does_not_mutate_input():
    msgs = [{"role": "user", "content": "Hello <image>"}]
    snapshot = copy.deepcopy(msgs)
    strip_image_tokens(msgs)
    assert msgs == snapshot
