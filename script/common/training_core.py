from __future__ import annotations

import gc
import inspect
import json
import logging
import os
import pickle
import random
import time
from collections import Counter
from collections.abc import Iterable
from typing import Any, Literal, overload

import torch
from PIL import Image
from common.confusion_matrix_plot import (
    render_confusion_matrix_image,
    render_row_normalized_confusion_matrix_image,
)

SYSTEM_MSG = (
    "당신은 작물 해충 식별 전문가입니다. "
    "사진을 보고 해충의 이름만 한국어로 답하세요. "
    '해충이 없으면 "정상"이라고만 답하세요. '
    "부가 설명 없이 이름만 출력하세요."
)

PROMPTS = [
    "이 사진에 있는 해충의 이름을 알려주세요.",
    "이 벌레는 무엇인가요?",
    "사진 속 해충을 식별해주세요.",
    "이 작물에 있는 해충의 종류가 무엇인가요?",
    "이 사진에서 어떤 해충이 보이나요?",
]

# RAM 기준 데이터 비율 상한
_RAM_RESERVE_GB = 15.0
_BYTES_PER_IMAGE = 3.0 * 1024**2

_line_count_cache: dict[str, int] = {}
_preloaded_samples: dict[tuple[str, str, float, int], list[dict[str, Any]]] = {}


def _get_logger(logger: logging.Logger | None) -> logging.Logger:
    return logger if logger is not None else logging.getLogger(__name__)


def get_line_count(path: str) -> int:
    if path not in _line_count_cache:
        with open(path, "r", encoding="utf-8") as f:
            _line_count_cache[path] = sum(1 for _ in f)
    return _line_count_cache[path]


def recommend_dataloader_num_workers(
    *,
    configured: int | None = None,
    reserve_cores: int = 2,
    max_workers: int = 8,
    logger: logging.Logger | None = None,
) -> int:
    """CPU 코어 수를 기반으로 DataLoader worker 수를 추천한다."""
    log = _get_logger(logger)
    if configured is not None:
        return max(0, int(configured))

    cpu_count = os.cpu_count() or 4
    workers = max(1, cpu_count - reserve_cores)
    workers = min(max_workers, workers)
    log.info(
        "DataLoader workers 자동 설정: %d (cpu=%d, reserve=%d, cap=%d)",
        workers,
        cpu_count,
        reserve_cores,
        max_workers,
    )
    return workers


def build_sft_config_kwargs(
    *,
    base_kwargs: dict[str, Any],
    sft_config_cls: type,
    seq_len: int | None = None,
    save_only_model: bool | None = None,
    dataloader_num_workers: int | None = None,
    dataloader_pin_memory: bool | None = None,
    dataloader_persistent_workers: bool | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """trl 버전 호환성을 고려해 SFTConfig kwargs를 구성한다."""
    log = _get_logger(logger)
    sft_kwargs = dict(base_kwargs)
    sft_init_params = inspect.signature(sft_config_cls.__init__).parameters

    if seq_len is not None:
        if "max_seq_length" in sft_init_params:
            sft_kwargs["max_seq_length"] = seq_len
        elif "max_length" in sft_init_params:
            sft_kwargs["max_length"] = seq_len
        else:
            log.warning(
                "SFTConfig에서 max length 인자(max_seq_length/max_length)를 찾지 못했습니다."
            )

    if save_only_model is not None:
        if "save_only_model" in sft_init_params:
            sft_kwargs["save_only_model"] = save_only_model
        else:
            log.warning(
                "SFTConfig에서 save_only_model 인자를 지원하지 않습니다. "
                "옵티마이저 상태 체크포인트가 저장될 수 있습니다."
            )

    optional_kwargs = {
        "dataloader_num_workers": dataloader_num_workers,
        "dataloader_pin_memory": dataloader_pin_memory,
        "dataloader_persistent_workers": dataloader_persistent_workers,
    }
    for key, value in optional_kwargs.items():
        if value is not None and key in sft_init_params:
            sft_kwargs[key] = value
    return sft_kwargs


def resolve_split_name(data_dir: str, split: str) -> str:
    primary = os.path.join(data_dir, f"{split}.jsonl")
    if os.path.exists(primary):
        return split

    aliases = {"val": "validation", "validation": "val"}
    alt = aliases.get(split)
    if alt and os.path.exists(os.path.join(data_dir, f"{alt}.jsonl")):
        return alt
    return split


def get_max_data_fraction(n_train_samples: int) -> float:
    try:
        total_bytes: int | None = None
        sysconf = getattr(os, "sysconf", None)
        if callable(sysconf):
            sysconf_names = getattr(os, "sysconf_names", {})
            page_key = sysconf_names.get("SC_PAGE_SIZE")
            phys_key = sysconf_names.get("SC_PHYS_PAGES")
            if isinstance(page_key, int) and isinstance(phys_key, int):
                page_size_raw = sysconf(page_key)
                phys_pages_raw = sysconf(phys_key)
                if isinstance(page_size_raw, (int, float, str)) and isinstance(
                    phys_pages_raw, (int, float, str)
                ):
                    page_size = int(page_size_raw)
                    phys_pages = int(phys_pages_raw)
                    total_bytes = page_size * phys_pages
        if total_bytes is None:
            import psutil

            total_bytes = int(psutil.virtual_memory().total)
        total_ram_gb = total_bytes / 1024**3
    except (ImportError, AttributeError, ValueError, OSError):
        return 1.0

    usable_gb = total_ram_gb - _RAM_RESERVE_GB
    if usable_gb <= 0:
        return 0.1
    max_images = int(usable_gb * 1024**3 / _BYTES_PER_IMAGE)
    return min(1.0, max_images / max(1, n_train_samples))


def crop_to_bbox(
    img: Image.Image, bbox: dict[str, int], padding_ratio: float = 0.0
) -> Image.Image:
    xtl, ytl = bbox["xtl"], bbox["ytl"]
    xbr, ybr = bbox["xbr"], bbox["ybr"]
    bw, bh = xbr - xtl, ybr - ytl
    pad_x, pad_y = int(bw * padding_ratio), int(bh * padding_ratio)
    x1 = max(0, xtl - pad_x)
    y1 = max(0, ytl - pad_y)
    x2 = min(img.width, xbr + pad_x)
    y2 = min(img.height, ybr + pad_y)
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


def cap_image_size(img: Image.Image, max_image_dim: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_image_dim:
        return img
    scale = max_image_dim / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    lanczos = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    resized = img.resize((new_w, new_h), lanczos)
    img.close()
    return resized


def find_label_json(
    data_dir: str, split: str, class_name: str, img_filename: str
) -> dict[str, int] | None:
    json_path = os.path.join(data_dir, split, class_name, img_filename + ".json")
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return None
    for obj in data.get("annotations", {}).get("object", []):
        if obj.get("grow") == 33 and obj.get("points"):
            return obj["points"][0]
    return None


def _extract_label_and_image_path(
    record: dict[str, Any],
) -> tuple[str, str] | None:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    try:
        label = messages[-1]["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(label, str):
        return None

    img_rel_path: str | None = None
    for msg in messages:
        for content in msg.get("content", []):
            if content.get("type") == "image" and "image" in content:
                img_rel_path = content["image"]
                break
        if img_rel_path is not None:
            break
    if not img_rel_path:
        return None
    return label, str(img_rel_path).replace("\\", "/")


def _build_subset_indices(
    *,
    total_lines: int,
    fraction: float,
    random_seed: int,
) -> set[int] | None:
    if fraction >= 1.0:
        return None
    sample_count = int(total_lines * fraction)
    rng = random.Random(random_seed)
    return set(rng.sample(range(total_lines), sample_count))


def _iter_jsonl_samples(
    *,
    data_dir: str,
    resolved_split: str,
    keep: set[int] | None,
):
    jsonl_path = os.path.join(data_dir, f"{resolved_split}.jsonl")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if keep is not None and i not in keep:
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            extracted = _extract_label_and_image_path(record)
            if extracted is None:
                continue
            label, img_rel_path = extracted
            parts = img_rel_path.split("/")
            if len(parts) < 3:
                continue
            img_path = os.path.join(data_dir, img_rel_path)
            if not os.path.exists(img_path):
                continue
            class_name, img_filename = parts[1], parts[2]
            yield label, img_path, class_name, img_filename


def make_conversation(
    image: Image.Image,
    label: str,
    *,
    system_msg: str = SYSTEM_MSG,
    prompts: list[str] = PROMPTS,
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_msg}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": random.choice(prompts)},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": label}]},
        ]
    }


def _preload_samples(
    *,
    data_dir: str,
    split: str,
    fraction: float,
    random_seed: int,
    max_image_dim: int,
    preload_cache_dir: str | None,
    logger: logging.Logger | None,
) -> list[dict[str, Any]]:
    log = _get_logger(logger)
    resolved_split = resolve_split_name(data_dir, split)
    if resolved_split != split:
        log.info("split 매핑 적용: %s -> %s", split, resolved_split)
    cache_key = (data_dir, resolved_split, round(fraction, 4), max_image_dim)
    if cache_key in _preloaded_samples:
        return _preloaded_samples[cache_key]

    cache_file: str | None = None
    if preload_cache_dir:
        os.makedirs(preload_cache_dir, exist_ok=True)
        cache_file = os.path.join(
            preload_cache_dir,
            f"{resolved_split}_f{round(fraction, 4)}_d{max_image_dim}.pkl",
        )
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    samples = pickle.load(f)
                _preloaded_samples[cache_key] = samples
                log.info("이미지 캐시 디스크 로드: %s (%d개)", cache_file, len(samples))
                return samples
            except Exception as exc:
                log.warning("캐시 로드 실패 (%s): %s", cache_file, exc)

    jsonl_path = os.path.join(data_dir, f"{resolved_split}.jsonl")
    total_lines = get_line_count(jsonl_path)
    keep = _build_subset_indices(
        total_lines=total_lines,
        fraction=fraction,
        random_seed=random_seed,
    )
    expected = len(keep) if keep is not None else total_lines
    log.info(
        "이미지 preload 시작: split=%s fraction=%.4f 예상=%d", split, fraction, expected
    )

    samples: list[dict[str, Any]] = []
    started = time.time()
    for label, img_path, class_name, img_filename in _iter_jsonl_samples(
        data_dir=data_dir,
        resolved_split=resolved_split,
        keep=keep,
    ):
        with Image.open(img_path) as opened_img:
            base_img = opened_img.convert("RGB")
        full_img = cap_image_size(base_img.copy(), max_image_dim)

        tight_img = None
        if label != "정상":
            bbox = find_label_json(data_dir, resolved_split, class_name, img_filename)
            if bbox:
                tight_img = cap_image_size(
                    crop_to_bbox(base_img.copy(), bbox, padding_ratio=0.0),
                    max_image_dim,
                )
        base_img.close()

        samples.append({"label": label, "full": full_img, "tight": tight_img})
        if len(samples) % 1000 == 0:
            elapsed = time.time() - started
            log.info(
                "  preload 진행: %d/%d (%.0fs, %.1f img/s)",
                len(samples),
                expected,
                elapsed,
                len(samples) / max(elapsed, 1e-3),
            )

    _preloaded_samples[cache_key] = samples
    if cache_file:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
            log.info("이미지 캐시 저장: %s", cache_file)
        except Exception as exc:
            log.warning("캐시 저장 실패 (%s): %s", cache_file, exc)
    return samples


class LazyImageDataset:
    def __init__(
        self,
        *,
        data_dir: str,
        split: str,
        tight_prob: float,
        fraction: float,
        random_seed: int,
        max_image_dim: int,
        logger: logging.Logger | None,
        system_msg: str = SYSTEM_MSG,
        prompts: list[str] = PROMPTS,
    ):
        self.data_dir = data_dir
        self.tight_prob = tight_prob
        self.random_seed = random_seed
        self.max_image_dim = max_image_dim
        self.logger = _get_logger(logger)
        self.system_msg = system_msg
        self.prompts = prompts
        self.samples = self._collect_metadata(split, fraction)
        self.logger.info(
            "LazyImageDataset 생성: split=%s 샘플=%d", split, len(self.samples)
        )

    def _collect_metadata(self, split: str, fraction: float) -> list[dict[str, Any]]:
        resolved_split = resolve_split_name(self.data_dir, split)
        if resolved_split != split:
            self.logger.info("split 매핑 적용: %s -> %s", split, resolved_split)
        jsonl_path = os.path.join(self.data_dir, f"{resolved_split}.jsonl")
        total_lines = get_line_count(jsonl_path)
        keep = _build_subset_indices(
            total_lines=total_lines,
            fraction=fraction,
            random_seed=self.random_seed,
        )

        out: list[dict[str, Any]] = []
        for label, img_path, class_name, img_filename in _iter_jsonl_samples(
            data_dir=self.data_dir,
            resolved_split=resolved_split,
            keep=keep,
        ):
            bbox = None
            if label != "정상":
                bbox = find_label_json(
                    self.data_dir, resolved_split, class_name, img_filename
                )
            out.append({"label": label, "img_path": img_path, "bbox": bbox})
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def _get_single_item(self, idx: int) -> dict[str, Any]:
        meta = self.samples[idx]
        if meta["bbox"] is not None and random.random() < self.tight_prob:
            orig = Image.open(meta["img_path"]).convert("RGB")
            img = cap_image_size(
                crop_to_bbox(orig, meta["bbox"], padding_ratio=0.0),
                self.max_image_dim,
            )
            orig.close()
        else:
            img = cap_image_size(
                Image.open(meta["img_path"]).convert("RGB"), self.max_image_dim
            )
        return make_conversation(
            img,
            meta["label"],
            system_msg=self.system_msg,
            prompts=self.prompts,
        )

    @overload
    def __getitem__(self, idx: int) -> dict[str, Any]: ...

    @overload
    def __getitem__(self, idx: slice) -> list[dict[str, Any]]: ...

    def __getitem__(self, idx: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(idx, slice):
            return [
                self._get_single_item(i) for i in range(*idx.indices(len(self.samples)))
            ]
        return self._get_single_item(idx)


@overload
def load_dataset_from_jsonl(
    *,
    data_dir: str,
    split: str,
    tight_prob: float = 0.5,
    fraction: float = 1.0,
    random_seed: int = 42,
    preload_cache_dir: str | None = None,
    max_image_dim: int = 768,
    lazy_dataset: Literal[True],
    logger: logging.Logger | None = None,
    system_msg: str = SYSTEM_MSG,
    prompts: list[str] = PROMPTS,
) -> LazyImageDataset: ...


@overload
def load_dataset_from_jsonl(
    *,
    data_dir: str,
    split: str,
    tight_prob: float = 0.5,
    fraction: float = 1.0,
    random_seed: int = 42,
    preload_cache_dir: str | None = None,
    max_image_dim: int = 768,
    lazy_dataset: Literal[False] = False,
    logger: logging.Logger | None = None,
    system_msg: str = SYSTEM_MSG,
    prompts: list[str] = PROMPTS,
) -> list[dict[str, Any]]: ...


def load_dataset_from_jsonl(
    *,
    data_dir: str,
    split: str,
    tight_prob: float = 0.5,
    fraction: float = 1.0,
    random_seed: int = 42,
    preload_cache_dir: str | None = None,
    max_image_dim: int = 768,
    lazy_dataset: bool = False,
    logger: logging.Logger | None = None,
    system_msg: str = SYSTEM_MSG,
    prompts: list[str] = PROMPTS,
) -> LazyImageDataset | list[dict[str, Any]]:
    if lazy_dataset:
        return LazyImageDataset(
            data_dir=data_dir,
            split=split,
            tight_prob=tight_prob,
            fraction=fraction,
            random_seed=random_seed,
            max_image_dim=max_image_dim,
            logger=logger,
            system_msg=system_msg,
            prompts=prompts,
        )

    samples = _preload_samples(
        data_dir=data_dir,
        split=split,
        fraction=fraction,
        random_seed=random_seed,
        max_image_dim=max_image_dim,
        preload_cache_dir=preload_cache_dir,
        logger=logger,
    )
    dataset = []
    for sample in samples:
        if sample["tight"] is None:
            img = sample["full"]
        elif random.random() < tight_prob:
            img = sample["tight"]
        else:
            img = sample["full"]
        dataset.append(
            make_conversation(
                img,
                sample["label"],
                system_msg=system_msg,
                prompts=prompts,
            )
        )
    random.shuffle(dataset)
    return dataset


def clear_gpu_memory(*, logger: logging.Logger | None = None) -> None:
    log = _get_logger(logger)
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.synchronize()
        log.info(
            "GPU 메모리 — allocated: %.1fGB, reserved: %.1fGB",
            torch.cuda.memory_allocated() / 1024**3,
            torch.cuda.memory_reserved() / 1024**3,
        )


def _has_meta_tensors(model: Any, *, logger: logging.Logger | None = None) -> bool:
    log = _get_logger(logger)
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            log.warning("Meta tensor 발견: %s", name)
            return True
    return False


def load_model_with_retry(
    *,
    base_model: str,
    max_retries: int = 2,
    logger: logging.Logger | None = None,
):
    from unsloth import FastVisionModel

    log = _get_logger(logger)
    last_err = None
    model = None
    tokenizer = None

    for attempt in range(1, max_retries + 1):
        try:
            model, tokenizer = FastVisionModel.from_pretrained(
                base_model,
                load_in_4bit=False,
                use_gradient_checkpointing="unsloth",
            )
            if _has_meta_tensors(model, logger=log):
                raise RuntimeError("모델은 로드됐지만 meta tensor가 포함되어 있습니다.")
            return model, tokenizer
        except Exception as exc:
            last_err = exc
            log.warning("모델 로딩 시도 %d/%d 실패: %s", attempt, max_retries, exc)
            try:
                del model
            except UnboundLocalError:
                pass
            try:
                del tokenizer
            except UnboundLocalError:
                pass
            model = None
            tokenizer = None
            for _ in range(5):
                gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
            time.sleep(2)

    raise RuntimeError(
        f"모델 로딩 {max_retries}회 시도 후 실패: {last_err}"
    ) from last_err


def evaluate_model(
    *,
    model: Any,
    tokenizer: Any,
    val_dataset: Any,
    max_samples: int = 200,
    save_dir: str | None = None,
    trial_num: str | int | None = None,
    logger: logging.Logger | None = None,
    system_msg: str = SYSTEM_MSG,
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    log = _get_logger(logger)
    samples = val_dataset[:max_samples]
    y_true, y_pred = [], []
    misclassifications: list[dict[str, str]] = []

    for item in samples:
        messages = item["messages"]
        ground_truth = messages[-1]["content"][0]["text"]
        image = messages[1]["content"][0]["image"]
        infer_messages = [
            {"role": "system", "content": [{"type": "text", "text": system_msg}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": messages[1]["content"][1]["text"]},
                ],
            },
        ]
        try:
            input_text = tokenizer.apply_chat_template(
                infer_messages, add_generation_prompt=True
            )
            inputs = tokenizer(
                image, input_text, add_special_tokens=False, return_tensors="pt"
            ).to("cuda")
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    use_cache=True,
                )
            generated = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            del inputs, output_ids
            y_true.append(ground_truth)
            y_pred.append(generated)
            if generated != ground_truth:
                misclassifications.append(
                    {"truth": ground_truth, "predicted": generated}
                )
        except Exception as exc:
            log.warning("추론 오류: %s", exc)
            continue

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not y_true:
        return {
            "accuracy": 0,
            "f1_macro": 0,
            "f1_weighted": 0,
            "precision_macro": 0,
            "recall_macro": 0,
            "total": 0,
        }

    all_labels = sorted(set(y_true + y_pred))
    score_kwargs: dict[str, Any] = {"labels": all_labels, "zero_division": 0}
    per_class_score_kwargs: dict[str, Any] = {
        "labels": all_labels,
        "average": None,
        "zero_division": 0,
    }

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(
        y_true,
        y_pred,
        average="macro",
        **score_kwargs,
    )
    rec_macro = recall_score(
        y_true,
        y_pred,
        average="macro",
        **score_kwargs,
    )
    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        **score_kwargs,
    )
    prec_weighted = precision_score(
        y_true,
        y_pred,
        average="weighted",
        **score_kwargs,
    )
    rec_weighted = recall_score(
        y_true,
        y_pred,
        average="weighted",
        **score_kwargs,
    )
    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        **score_kwargs,
    )

    prec_per = precision_score(y_true, y_pred, **per_class_score_kwargs)
    rec_per = recall_score(y_true, y_pred, **per_class_score_kwargs)
    f1_per = f1_score(y_true, y_pred, **per_class_score_kwargs)

    def _as_float_list(values: Any, expected_len: int) -> list[float]:
        if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
            return [float(v) for v in values]
        return [float(values)] * expected_len

    prec_per_list = _as_float_list(prec_per, len(all_labels))
    rec_per_list = _as_float_list(rec_per, len(all_labels))
    f1_per_list = _as_float_list(f1_per, len(all_labels))
    label_support = Counter(y_true)
    per_class: dict[str, dict[str, float | int]] = {}
    for i, cls in enumerate(all_labels):
        per_class[cls] = {
            "precision": prec_per_list[i],
            "recall": rec_per_list[i],
            "f1": f1_per_list[i],
            "support": int(label_support.get(cls, 0)),
        }

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    cm_path = None
    cm_row_norm_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        try:
            trial_label = f"Trial {trial_num}" if trial_num is not None else ""
            cm_path = os.path.join(
                save_dir, f"confusion_matrix_trial_{trial_num or 'final'}.png"
            )
            cm_row_norm_path = os.path.join(
                save_dir,
                f"confusion_matrix_row_normalized_trial_{trial_num or 'final'}.png",
            )
            render_confusion_matrix_image(
                confusion_matrix_values=cm.tolist(),
                labels=all_labels,
                output_path=cm_path,
                accuracy=float(acc),
                f1_macro=float(f1_macro),
                trial_label=trial_label,
            )
            render_row_normalized_confusion_matrix_image(
                confusion_matrix_values=cm.tolist(),
                labels=all_labels,
                output_path=cm_row_norm_path,
                accuracy=float(acc),
                f1_macro=float(f1_macro),
                f1_weighted=float(f1_weighted),
                sample_count=len(y_true),
            )
            log.info("혼동 행렬 저장: %s", cm_path)
            log.info("정규화 혼동 행렬 저장: %s", cm_row_norm_path)
        except Exception as exc:
            log.warning("혼동 행렬 플롯 실패: %s", exc)

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_path": cm_path,
        "confusion_matrix_row_normalized_path": cm_row_norm_path,
        "total": len(y_true),
        "correct": int(acc * len(y_true)),
        "top_misclassifications": misclassifications[:20],
    }
