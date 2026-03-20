from __future__ import annotations

import io
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


_BENIGN_MODEL_LOAD_OUTPUT_MARKERS = (
    "BertModel LOAD REPORT",
    "embeddings.position_ids",
    "UNEXPECTED",
)


def load_embedding_model(model_path: Path):
    if not model_path.exists():
        raise ValueError(f"未找到语义模型：{model_path}")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ValueError(
            "未安装 sentence-transformers，请先执行 uv sync。"
        ) from exc

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    process_output: _CapturedProcessOutput | None = None

    try:
        with (
            _capture_process_output() as process_output,
            redirect_stdout(stdout_buffer),
            redirect_stderr(stderr_buffer),
        ):
            model = SentenceTransformer(str(model_path), local_files_only=True)
    except Exception:
        _replay_model_load_output(
            stdout_text=stdout_buffer.getvalue(),
            stderr_text=stderr_buffer.getvalue(),
            process_text=process_output.read() if process_output is not None else "",
        )
        raise

    _replay_model_load_output(
        stdout_text=stdout_buffer.getvalue(),
        stderr_text=stderr_buffer.getvalue(),
        process_text=process_output.read() if process_output is not None else "",
    )
    return model


def _replay_model_load_output(stdout_text: str, stderr_text: str, process_text: str) -> None:
    captured_output = "\n".join(
        text.strip()
        for text in (stdout_text, stderr_text, process_text)
        if text.strip()
    )
    if captured_output and _is_benign_model_load_output(captured_output):
        return

    if stdout_text:
        sys.stdout.write(stdout_text)
    if stderr_text:
        sys.stderr.write(stderr_text)
    if process_text:
        try:
            os.write(sys.__stdout__.fileno(), process_text.encode())
        except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
            sys.stdout.write(process_text)


def _is_benign_model_load_output(output: str) -> bool:
    return all(marker in output for marker in _BENIGN_MODEL_LOAD_OUTPUT_MARKERS)


class _CapturedProcessOutput:
    def __init__(self, stdout_file, stderr_file) -> None:
        self._stdout_file = stdout_file
        self._stderr_file = stderr_file

    def read(self) -> str:
        parts: list[str] = []
        for file_obj in (self._stdout_file, self._stderr_file):
            file_obj.flush()
            file_obj.seek(0)
            content = file_obj.read().decode()
            if content:
                parts.append(content)
        return "".join(parts)


class _capture_process_output:
    def __enter__(self) -> _CapturedProcessOutput:
        self._stdout_file = tempfile.TemporaryFile(mode="w+b")
        self._stderr_file = tempfile.TemporaryFile(mode="w+b")
        self._stdout_fd: int | None = None
        self._stderr_fd: int | None = None
        _safe_flush(sys.stdout)
        _safe_flush(sys.stderr)
        try:
            self._stdout_fd = os.dup(sys.__stdout__.fileno())
            self._stderr_fd = os.dup(sys.__stderr__.fileno())
            os.dup2(self._stdout_file.fileno(), sys.__stdout__.fileno())
            os.dup2(self._stderr_file.fileno(), sys.__stderr__.fileno())
        except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
            self._stdout_fd = None
            self._stderr_fd = None
        return _CapturedProcessOutput(self._stdout_file, self._stderr_file)

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        _safe_flush(sys.stdout)
        _safe_flush(sys.stderr)
        if self._stdout_fd is not None and self._stderr_fd is not None:
            os.dup2(self._stdout_fd, sys.__stdout__.fileno())
            os.dup2(self._stderr_fd, sys.__stderr__.fileno())
            os.close(self._stdout_fd)
            os.close(self._stderr_fd)
        self._stdout_file.flush()
        self._stderr_file.flush()


def _safe_flush(stream) -> None:
    try:
        stream.flush()
    except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
        return
