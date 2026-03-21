from __future__ import annotations

import copy
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from mysphinx_forge.cleaning import resolve_target_column
from mysphinx_forge.deduplication import normalize_dedup_text

MODEL_TEST_USER_INPUT = "请问退款怎么申请？"
MODEL_RESULT_COLUMN = "模型结果"
EXPECTED_RESULT_COLUMN = "预期结果"
MATCH_EXPECTED_COLUMN = "匹配预期"
MODEL_CALL_TIME_COLUMN = "模型调用时间"


@dataclass(slots=True)
class ModelTestResult:
    model_path: str
    user_input: str
    generated_text: str
    model_class: str
    tokenizer_class: str
    device: str


@dataclass(slots=True)
class BatchModelTestStats:
    total_rows: int
    target_column: str
    model_result_column: str = MODEL_RESULT_COLUMN
    model_call_time_column: str = MODEL_CALL_TIME_COLUMN
    expected_result_column: str = EXPECTED_RESULT_COLUMN
    match_expected_column: str = MATCH_EXPECTED_COLUMN
    has_expected_result: bool = False
    matched_expected_count: int = 0
    average_call_time_seconds: float = 0.0
    model_path: str = ""
    device: str = ""
    num_workers: int = 1
    batch_size: int = 1


@dataclass(slots=True)
class ModelTestRuntimeConfig:
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.05
    batch_size: int = 8
    num_workers: int | str = "auto"


class LocalModelTester:
    def __init__(
        self,
        model_path: str | Path,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.05,
        device: str | None = None,
    ) -> None:
        resolved_model_path = Path(model_path)
        if not resolved_model_path.exists():
            raise ValueError(f"未找到测试模型：{resolved_model_path}")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ValueError("未安装模型测试所需依赖，请先执行 uv sync。") from exc

        self.model_path = resolved_model_path
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self._torch = torch
        self.device = device or _resolve_inference_device(torch)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(resolved_model_path),
                local_files_only=True,
                trust_remote_code=True,
            )
            self.model, self._is_causal = _load_generation_model(
                model_path=resolved_model_path,
                auto_causal_model=AutoModelForCausalLM,
                auto_seq2seq_model=AutoModelForSeq2SeqLM,
            )
        except Exception as exc:
            raise ValueError(f"加载测试模型失败：{type(exc).__name__}: {exc}") from exc

        self.model = self.model.to(self.device)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self._is_causal:
            self.tokenizer.padding_side = "left"

        self.model_class = type(self.model).__name__
        self.tokenizer_class = type(self.tokenizer).__name__

    def generate_text(self, user_input: object, *, max_new_tokens: int = 64) -> str:
        return self.generate_texts([user_input], max_new_tokens=max_new_tokens)[0]

    def generate_texts(self, user_inputs: list[object], *, max_new_tokens: int = 64) -> list[str]:
        prompts = ["" if pd.isna(value) else str(value).strip() for value in user_inputs]
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        generation_config = _build_generation_config(
            model=self.model,
            max_new_tokens=max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        with self._torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )

        prompt_lengths = (
            attention_mask.sum(dim=1).tolist()
            if attention_mask is not None
            else [input_ids.shape[1]] * len(prompts)
        )
        outputs: list[str] = []
        for index, generated_ids in enumerate(generated):
            if self._is_causal:
                response_ids = generated_ids[int(prompt_lengths[index]):]
            else:
                response_ids = generated_ids

            generated_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            if not generated_text:
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if not generated_text:
                raise ValueError("模型测试失败：模型未生成有效输出。")
            outputs.append(generated_text)
        return outputs


def run_model_test(
    model_path: str | Path,
    user_input: str = MODEL_TEST_USER_INPUT,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.05,
) -> ModelTestResult:
    tester = LocalModelTester(
        model_path=model_path,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    generated_text = tester.generate_text(user_input, max_new_tokens=max_new_tokens)
    return ModelTestResult(
        model_path=str(tester.model_path),
        user_input=user_input,
        generated_text=generated_text,
        model_class=tester.model_class,
        tokenizer_class=tester.tokenizer_class,
        device=tester.device,
    )


def model_test_dataframe(
    dataframe: pd.DataFrame,
    model_path: str | Path,
    *,
    runtime_config: ModelTestRuntimeConfig,
    target_column: str = "text",
    expected_result_column: str = EXPECTED_RESULT_COLUMN,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[pd.DataFrame, BatchModelTestStats]:
    resolved_target_column = resolve_target_column(dataframe, target_column)
    has_expected_result = expected_result_column in dataframe.columns
    prompts = dataframe[resolved_target_column].tolist()
    expected_results = (
        dataframe[expected_result_column].tolist() if has_expected_result else [None] * len(dataframe)
    )
    devices = resolve_worker_devices(runtime_config.num_workers)

    if len(devices) == 1:
        model_results, model_call_times, device_used = _run_single_process_batches(
            prompts=prompts,
            model_path=model_path,
            runtime_config=runtime_config,
            device=devices[0],
            progress_callback=progress_callback,
        )
    else:
        model_results, model_call_times = _run_multi_worker_batches(
            prompts=prompts,
            model_path=model_path,
            runtime_config=runtime_config,
            devices=devices,
            progress_callback=progress_callback,
        )
        device_used = ",".join(devices)

    tested = dataframe.copy()
    tested[MODEL_RESULT_COLUMN] = model_results
    tested[MODEL_CALL_TIME_COLUMN] = model_call_times

    match_expected: list[bool] = []
    if has_expected_result:
        match_expected = [
            _is_expected_match(expected, model_result)
            for expected, model_result in zip(expected_results, model_results, strict=True)
        ]
        tested[MATCH_EXPECTED_COLUMN] = match_expected

    stats = BatchModelTestStats(
        total_rows=len(tested),
        target_column=resolved_target_column,
        has_expected_result=has_expected_result,
        matched_expected_count=sum(match_expected),
        average_call_time_seconds=round(sum(model_call_times) / len(model_call_times), 4)
        if model_call_times
        else 0.0,
        model_path=str(model_path),
        device=device_used,
        num_workers=len(devices),
        batch_size=runtime_config.batch_size,
    )
    return tested, stats


def resolve_worker_devices(num_workers: int | str = "auto") -> list[str]:
    gpu_count = get_visible_gpu_count()
    if gpu_count <= 0:
        return ["cpu"]

    if num_workers == "auto":
        worker_count = gpu_count
    else:
        worker_count = max(1, min(int(num_workers), gpu_count))
    return [f"cuda:{index}" for index in range(worker_count)]


def get_visible_gpu_count() -> int:
    try:
        import torch
    except ImportError:
        return 0
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def _run_single_process_batches(
    prompts: list[object],
    model_path: str | Path,
    runtime_config: ModelTestRuntimeConfig,
    device: str,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[list[str], list[float], str]:
    tester = LocalModelTester(
        model_path=model_path,
        do_sample=runtime_config.do_sample,
        temperature=runtime_config.temperature,
        top_p=runtime_config.top_p,
        top_k=runtime_config.top_k,
        repetition_penalty=runtime_config.repetition_penalty,
        device=device,
    )
    model_results: list[str] = []
    model_call_times: list[float] = []
    for batch_prompts in _chunk_list(prompts, runtime_config.batch_size):
        started_at = time.perf_counter()
        batch_results = tester.generate_texts(batch_prompts, max_new_tokens=runtime_config.max_new_tokens)
        elapsed_seconds = round(time.perf_counter() - started_at, 4)
        per_row_seconds = round(elapsed_seconds / len(batch_prompts), 4)
        model_results.extend(batch_results)
        model_call_times.extend([per_row_seconds] * len(batch_prompts))
        if progress_callback:
            progress_callback(len(batch_prompts))
    return model_results, model_call_times, tester.device


def _run_multi_worker_batches(
    prompts: list[object],
    model_path: str | Path,
    runtime_config: ModelTestRuntimeConfig,
    devices: list[str],
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[list[str], list[float]]:
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_batch_model_test_worker,
            args=(
                task_queue,
                result_queue,
                {
                    "model_path": str(model_path),
                    "device": device,
                    "max_new_tokens": runtime_config.max_new_tokens,
                    "do_sample": runtime_config.do_sample,
                    "temperature": runtime_config.temperature,
                    "top_p": runtime_config.top_p,
                    "top_k": runtime_config.top_k,
                    "repetition_penalty": runtime_config.repetition_penalty,
                },
            ),
        )
        for device in devices
    ]

    batches = list(enumerate(_chunk_list(list(enumerate(prompts)), runtime_config.batch_size)))
    for batch_id, batch_items in batches:
        task_queue.put((batch_id, batch_items))
    for _ in processes:
        task_queue.put(None)

    for process in processes:
        process.start()

    model_results = [""] * len(prompts)
    model_call_times = [0.0] * len(prompts)
    remaining_batches = len(batches)
    try:
        while remaining_batches > 0:
            result = result_queue.get()
            if result["status"] == "error":
                raise ValueError(result["message"])

            for row_index, generated_text, call_time in result["rows"]:
                model_results[row_index] = generated_text
                model_call_times[row_index] = call_time
            if progress_callback:
                progress_callback(len(result["rows"]))
            remaining_batches -= 1
    finally:
        for process in processes:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join()

    return model_results, model_call_times


def _batch_model_test_worker(task_queue, result_queue, worker_config: dict[str, object]) -> None:
    try:
        tester = LocalModelTester(
            model_path=worker_config["model_path"],
            do_sample=bool(worker_config["do_sample"]),
            temperature=float(worker_config["temperature"]),
            top_p=float(worker_config["top_p"]),
            top_k=int(worker_config["top_k"]),
            repetition_penalty=float(worker_config["repetition_penalty"]),
            device=str(worker_config["device"]),
        )
        max_new_tokens = int(worker_config["max_new_tokens"])
        while True:
            try:
                item = task_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                return

            _batch_id, batch_items = item
            row_indices = [row_index for row_index, _ in batch_items]
            prompts = [prompt for _, prompt in batch_items]
            started_at = time.perf_counter()
            batch_results = tester.generate_texts(prompts, max_new_tokens=max_new_tokens)
            elapsed_seconds = round(time.perf_counter() - started_at, 4)
            per_row_seconds = round(elapsed_seconds / len(batch_items), 4)
            result_queue.put(
                {
                    "status": "ok",
                    "rows": [
                        (row_index, generated_text, per_row_seconds)
                        for row_index, generated_text in zip(row_indices, batch_results, strict=True)
                    ],
                }
            )
    except Exception as exc:
        result_queue.put({"status": "error", "message": f"{type(exc).__name__}: {exc}"})


def _chunk_list(items: list[object], chunk_size: int) -> list[list[object]]:
    return [items[index: index + chunk_size] for index in range(0, len(items), chunk_size)]


def _is_expected_match(expected_result: object, model_result: str) -> bool:
    return normalize_dedup_text(expected_result) == normalize_dedup_text(model_result)


def _load_generation_model(model_path: Path, auto_causal_model, auto_seq2seq_model):
    causal_error: Exception | None = None
    try:
        model = auto_causal_model.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
        )
        model.eval()
        return model, True
    except Exception as exc:
        causal_error = exc

    try:
        model = auto_seq2seq_model.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
        )
        model.eval()
        return model, False
    except Exception as exc:
        if causal_error is not None:
            raise ValueError(
                f"无法按生成模型加载：causal={type(causal_error).__name__}: {causal_error}; "
                f"seq2seq={type(exc).__name__}: {exc}"
            ) from exc
        raise


def _resolve_inference_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def _build_generation_config(
    model: object,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    pad_token_id: int | None,
):
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = max_new_tokens
    generation_config.do_sample = do_sample
    generation_config.repetition_penalty = repetition_penalty
    generation_config.pad_token_id = pad_token_id

    if do_sample:
        generation_config.temperature = temperature
        generation_config.top_p = top_p
        generation_config.top_k = top_k
        return generation_config

    generation_config.temperature = 1.0
    generation_config.top_p = 1.0
    generation_config.top_k = 50
    return generation_config
