#!/usr/bin/env python3
"""Crop hand images from real_data videos into a train/validation layout."""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_DIR = ROOT / "real_data"
DEFAULT_OUTPUT_DIR = ROOT / "data"
DEFAULT_FRAME_STEP = 1
MIN_DIMENSION = 6
MAX_DIMENSION = 750
DEFAULT_DETECTOR_MODEL = ROOT / "deimv2_dinov3_x_wholebody34_680query_n_batch_640x640.onnx"
DETECTOR_INPUT_SIZE = 640
HAND_LABEL = 26
HAND_THRESHOLD = 0.35
LABEL_TO_ID = {
    "no_waving_hand": 0,
    "waving_hand": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop hand regions from real_data videos into data/train and data/validation folders."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_VIDEO_DIR, help="Directory containing real_data videos.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory to store cropped images (train/validation).",
    )
    parser.add_argument("--frame-step", type=int, default=DEFAULT_FRAME_STEP, help="Take every Nth frame (default: 1).")
    parser.add_argument("--min-dimension", type=int, default=MIN_DIMENSION, help="Minimum crop width/height (default: 6).")
    parser.add_argument("--max-dimension", type=int, default=MAX_DIMENSION, help="Maximum crop width/height (default: 750).")
    parser.add_argument(
        "--detector-model",
        type=Path,
        default=DEFAULT_DETECTOR_MODEL,
        help="ONNX detector used to find hand boxes (default: deimv2...640.onnx).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNGs if duplicates occur.")
    parser.add_argument("--dry-run", action="store_true", help="Plan operations without writing files.")
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="Fraction of crops routed to the validation split (default: 0.2).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for train/validation split assignment.")
    args = parser.parse_args()
    if args.frame_step < 1:
        parser.error("--frame-step must be at least 1")
    if args.max_dimension < args.min_dimension:
        parser.error("--max-dimension must be greater than or equal to --min-dimension")
    if not 0.0 <= args.validation_ratio < 1.0:
        parser.error("--validation-ratio must be between 0.0 (inclusive) and 1.0 (exclusive)")
    return args


def load_detector_session(model_path: Path) -> tuple[ort.InferenceSession, str]:
    if not model_path.exists():
        raise FileNotFoundError(f"Detector model not found: {model_path}")
    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": ".",
                "trt_op_types_to_exclude": "NonMaxSuppression,NonZero,RoiAlign",
            },
        ),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name


def _prepare_detector_blob(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        image,
        (DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    blob = resized.transpose(2, 0, 1).astype(np.float32, copy=False)
    blob = np.expand_dims(blob, axis=0)
    return blob


def _run_detector(session: ort.InferenceSession, input_name: str, image: np.ndarray) -> np.ndarray:
    blob = _prepare_detector_blob(image)
    return session.run(None, {input_name: blob})[0][0]


def detect_hand_box(
    session: ort.InferenceSession,
    input_name: str,
    frame: np.ndarray,
) -> Optional[tuple[float, float, float, float]]:
    detections = _run_detector(session, input_name, frame)
    best_det = None
    best_score = HAND_THRESHOLD
    for det in detections:
        label = int(round(det[0]))
        score = float(det[5])
        if label != HAND_LABEL or score < HAND_THRESHOLD:
            continue
        if score >= best_score:
            best_score = score
            best_det = det
    if best_det is None:
        return None
    return float(best_det[1]), float(best_det[2]), float(best_det[3]), float(best_det[4])


def crop_frame_using_box(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
) -> Optional[tuple[np.ndarray, int, int]]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = min(max(x1, 0.0), 1.0)
    y1 = min(max(y1, 0.0), 1.0)
    x2 = min(max(x2, 0.0), 1.0)
    y2 = min(max(y2, 0.0), 1.0)
    if x2 <= x1 or y2 <= y1:
        return None
    x1_px = max(int(round(x1 * width)), 0)
    y1_px = max(int(round(y1 * height)), 0)
    x2_px = min(int(round(x2 * width)), width)
    y2_px = min(int(round(y2 * height)), height)
    if x2_px <= x1_px or y2_px <= y1_px:
        return None
    crop = frame[y1_px:y2_px, x1_px:x2_px].copy()
    return crop, crop.shape[1], crop.shape[0]


def derive_prefix(stem: str) -> str:
    """Return filename prefix before the first digit; fallback to full stem."""
    match = re.match(r"^([a-zA-Z_]+)", stem)
    if match:
        return match.group(1)
    return stem


def infer_label_from_video(path: Path) -> Optional[str]:
    prefix = derive_prefix(path.stem).lower()
    for label in LABEL_TO_ID:
        if prefix.startswith(label):
            return label
    return None


def iter_video_files(input_dir: Path) -> Iterable[tuple[Path, str]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    for path in sorted(input_dir.glob("*.mp*")):
        label = infer_label_from_video(path)
        if label is None:
            print(f"[skip] {path.name} (does not match expected waving-hand prefixes).", file=sys.stderr)
            continue
        yield path, label


def save_frame(frame: np.ndarray, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def decide_split(label: str, split_counts: dict[str, dict[str, int]], validation_ratio: float, rng: random.Random) -> str:
    """Pick train/validation to keep the running ratio close to the target for each label."""
    if validation_ratio <= 0.0:
        return "train"
    train_count = split_counts["train"][label]
    val_count = split_counts["validation"][label]
    total = train_count + val_count
    ratio_if_train = val_count / (total + 1)
    ratio_if_val = (val_count + 1) / (total + 1)
    target = validation_ratio
    diff_train = abs(ratio_if_train - target)
    diff_val = abs(ratio_if_val - target)
    if diff_val < diff_train:
        return "validation"
    if diff_train < diff_val:
        return "train"
    return "validation" if rng.random() < target else "train"


def process_video(
    video_path: Path,
    label: str,
    args: argparse.Namespace,
    rng: random.Random,
    split_counts: dict[str, dict[str, int]],
    class_counts: dict[str, int],
    detector_session: ort.InferenceSession,
    detector_input_name: str,
) -> dict[str, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")

    frame_index = 0
    saved_per_split = {"train": 0, "validation": 0}
    video_prefix = video_path.stem  # keep per-file uniqueness to avoid collisions
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        if frame_index % args.frame_step != 0:
            frame_index += 1
            continue

        box = detect_hand_box(detector_session, detector_input_name, frame)
        if box is None:
            frame_index += 1
            continue

        crop_result = crop_frame_using_box(frame, box)
        if crop_result is None:
            frame_index += 1
            continue
        crop, width_px, height_px = crop_result
        if (
            width_px < args.min_dimension
            or height_px < args.min_dimension
            or width_px > args.max_dimension
            or height_px > args.max_dimension
        ):
            frame_index += 1
            continue

        split = decide_split(label, split_counts, args.validation_ratio, rng)
        filename = f"{video_prefix}_{frame_index:06d}.png"
        output_path = args.output_dir / split / label / filename
        if not args.dry_run:
            try:
                save_frame(crop, output_path, overwrite=args.overwrite)
            except FileExistsError:
                frame_index += 1
                continue
        saved_per_split[split] += 1
        split_counts[split][label] += 1
        class_counts[label] += 1
        frame_index += 1

    capture.release()
    saved_total = sum(saved_per_split.values())
    print(
        f"[info] Processed {video_path.name} -> {label}: "
        f"train {saved_per_split['train']}, validation {saved_per_split['validation']} (total {saved_total})."
    )
    return saved_per_split


def save_class_distribution_pie(counts: dict[str, int], output_dir: Path) -> Optional[Path]:
    total = sum(counts.values())
    if total == 0:
        return None
    labels = []
    sizes = []
    for label in LABEL_TO_ID:
        labels.append(f"{label} (n={counts[label]})")
        sizes.append(counts[label])

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")

    output_path = output_dir / "class_distribution.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    detector_session, detector_input_name = load_detector_session(args.detector_model)
    rng = random.Random(args.seed)

    total_saved = 0
    class_counts = {label: 0 for label in LABEL_TO_ID}
    split_counts = {
        "train": {label: 0 for label in LABEL_TO_ID},
        "validation": {label: 0 for label in LABEL_TO_ID},
    }

    video_entries = list(iter_video_files(args.input_dir))
    if not video_entries:
        print("[info] No matching videos found.")
        return

    for video_path, label in video_entries:
        saved = process_video(
            video_path,
            label,
            args,
            rng,
            split_counts,
            class_counts,
            detector_session,
            detector_input_name,
        )
        total_saved += sum(saved.values())

    if args.dry_run:
        print("[dry-run] Skipped writing files.")
    else:
        chart_path = save_class_distribution_pie(class_counts, args.output_dir)
        if chart_path:
            print(f"[info] Saved class distribution chart: {chart_path}")
        else:
            print("[warn] No crops saved; skipped class distribution chart.")

    total_saved = sum(class_counts.values())
    print(f"[done] Saved {total_saved} crops from {len(video_entries)} videos.")
    for split in ("train", "validation"):
        summary = ", ".join(f"{label}: {split_counts[split][label]}" for label in LABEL_TO_ID)
        print(f"[summary] {split}: {summary}")


if __name__ == "__main__":
    main()
