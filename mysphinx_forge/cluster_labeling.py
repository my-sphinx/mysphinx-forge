from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Protocol
from urllib import error, request


DEFAULT_CLUSTER_LABEL_MODEL = "gpt-4.1-mini"
DEFAULT_CLUSTER_LABEL_API_BASE_URL = "https://api.openai.com/v1"


@dataclass(slots=True)
class ClusterLabelContext:
    cluster_id: int
    cluster_size: int
    representative_text: str
    top_keywords: list[str]
    sample_texts: list[str]


class ClusterLabelGenerator(Protocol):
    def generate_label(self, context: ClusterLabelContext) -> str: ...


class RuleBasedClusterLabelGenerator:
    def generate_label(self, context: ClusterLabelContext) -> str:
        first_keyword = context.top_keywords[0] if context.top_keywords else ""
        representative_text = context.representative_text.strip()
        if first_keyword and first_keyword not in representative_text:
            return f"{first_keyword} | {representative_text}"
        return representative_text or first_keyword


class OpenAICompatibleClusterLabelGenerator:
    def __init__(
        self,
        model: str = DEFAULT_CLUSTER_LABEL_MODEL,
        api_key: str | None = None,
        api_base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "").strip()
        self.api_base_url = (
            api_base_url
            or os.environ.get("OPENAI_BASE_URL", "").strip()
            or DEFAULT_CLUSTER_LABEL_API_BASE_URL
        ).rstrip("/")
        self.timeout_seconds = timeout_seconds

        if not self.api_key:
            raise ValueError("启用 LLM 聚类标签需要设置 OPENAI_API_KEY 环境变量。")

    def generate_label(self, context: ClusterLabelContext) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是中文聚类标签生成器。"
                        "请根据同一簇的一组问题，生成一个4到12个汉字的中文主题标签。"
                        "要求：概括共同主题，不要照抄某一条原问题，不要用问句，"
                        "不要输出“相关问题”“主题”“问题集合”这类空泛表述，"
                        "不要附带解释、引号、编号，只输出标签文本本身。"
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_user_prompt(context),
                },
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        api_request = request.Request(
            url=f"{self.api_base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(api_request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise ValueError(f"LLM 聚类标签生成失败：HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise ValueError(f"LLM 聚类标签生成失败：{exc.reason}") from exc

        label = _extract_chat_completion_text(response_payload)
        normalized_label = _normalize_label_text(label)
        if not normalized_label:
            raise ValueError("LLM 聚类标签生成失败：返回了空标签。")
        return normalized_label

    def _build_user_prompt(self, context: ClusterLabelContext) -> str:
        sample_lines = "\n".join(f"- {text}" for text in context.sample_texts)
        keyword_text = "、".join(context.top_keywords) if context.top_keywords else "无"
        return (
            f"簇编号：{context.cluster_id}\n"
            f"簇大小：{context.cluster_size}\n"
            f"代表问题：{context.representative_text}\n"
            f"高频关键词：{keyword_text}\n"
            f"示例问题：\n{sample_lines}"
        )


def _extract_chat_completion_text(payload: dict[str, object]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM 聚类标签生成失败：响应中缺少 choices。")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise ValueError("LLM 聚类标签生成失败：响应中缺少 message。")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
        if parts:
            return "".join(parts)
    raise ValueError("LLM 聚类标签生成失败：响应中缺少 content。")


def _normalize_label_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    first_line = stripped.splitlines()[0].strip()
    first_line = re.sub(r"^[\s\"'“”‘’]+|[\s\"'“”‘’]+$", "", first_line)
    first_line = re.sub(r"^[0-9]+[.)、:\-]\s*", "", first_line)
    first_line = re.sub(r"^(标签|主题)[:：]\s*", "", first_line)
    return first_line.strip(" .,!?:;，。！？；：")
