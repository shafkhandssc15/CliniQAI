from types import SimpleNamespace

import ocr_engine


def test_openai_vlm_sends_image_and_parses_json(monkeypatch):
    captured = {}

    monkeypatch.setattr(ocr_engine, "normalize_image_bytes", lambda image: (b"image", "image/png"))

    def fake_post(url, headers, json, timeout):
        captured.update(url=url, headers=headers, payload=json, timeout=timeout)
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": '{"Hemoglobin": 12.4, "Predicted_Diseases": []}'}}]},
        )

    monkeypatch.setattr(ocr_engine.requests, "post", fake_post)

    result = ocr_engine.run_openai_vlm(b"report", "test-key")

    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["payload"]["model"] == "gpt-4o-mini"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["messages"][0]["content"][1]["type"] == "image_url"
    assert result["Hemoglobin"] == 12.4
    assert result["Predicted_Diseases"] == []