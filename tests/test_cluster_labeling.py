from __future__ import annotations

import json
from urllib import error

import pytest

from mysphinx_forge.cluster_labeling import (
    ClusterLabelContext,
    OpenAICompatibleClusterLabelGenerator,
    RuleBasedClusterLabelGenerator,
)


def test_rule_based_cluster_label_generator_uses_keyword_and_representative_text() -> None:
    generator = RuleBasedClusterLabelGenerator()

    label = generator.generate_label(
        ClusterLabelContext(
            cluster_id=0,
            cluster_size=8,
            representative_text="什么是股票基金？",
            top_keywords=["黄金", "基金"],
            sample_texts=["什么是股票基金？", "黄金ETF是什么？"],
        )
    )

    assert label == "黄金 | 什么是股票基金？"


def test_openai_cluster_label_generator_reads_label_from_response(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "黄金投资入门",
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request_obj, timeout=30.0):
        assert request_obj.full_url == "https://api.openai.com/v1/chat/completions"
        assert timeout == 30.0
        return FakeResponse()

    monkeypatch.setattr("mysphinx_forge.cluster_labeling.request.urlopen", fake_urlopen)
    generator = OpenAICompatibleClusterLabelGenerator()

    label = generator.generate_label(
        ClusterLabelContext(
            cluster_id=0,
            cluster_size=12,
            representative_text="什么是股票基金？",
            top_keywords=["黄金", "基金", "债券"],
            sample_texts=["什么是股票基金？", "黄金ETF是什么？", "债券基金怎么买？"],
        )
    )

    assert label == "黄金投资入门"


def test_openai_cluster_label_generator_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAICompatibleClusterLabelGenerator()


def test_openai_cluster_label_generator_surfaces_http_errors(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeHttpError(error.HTTPError):
        def __init__(self):
            super().__init__(
                url="https://api.openai.com/v1/chat/completions",
                code=401,
                msg="Unauthorized",
                hdrs=None,
                fp=None,
            )

        def read(self):
            return b'{"error":"bad key"}'

    def fake_urlopen(_request_obj, timeout=30.0):
        raise FakeHttpError()

    monkeypatch.setattr("mysphinx_forge.cluster_labeling.request.urlopen", fake_urlopen)
    generator = OpenAICompatibleClusterLabelGenerator()

    with pytest.raises(ValueError, match="HTTP 401"):
        generator.generate_label(
            ClusterLabelContext(
                cluster_id=0,
                cluster_size=3,
                representative_text="退款怎么申请",
                top_keywords=["退款"],
                sample_texts=["退款怎么申请"],
            )
        )
