from __future__ import annotations

import argparse
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "data" / "grad2_pilot" / "videos"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "grad2_pilot" / "processed"
DEFAULT_TARGET_FPS = 20
DEFAULT_TRIM_SECONDS = 2.0


def build_processed_name(
    source: Path,
    target_fps: int,
    trim_start: float,
    trim_end: float,
) -> str:
    if trim_start != trim_end:
        trim_suffix = f"trim{trim_start:g}s_{trim_end:g}s"
    else:
        trim_suffix = f"trim{trim_start:g}s"
    return f"{source.stem}_4k_{target_fps}fps_{trim_suffix}.mp4"


def compute_trim_window(
    frame_count: int,
    source_fps: float,
    trim_start: float,
    trim_end: float,
) -> tuple[int, int]:
    if source_fps <= 0:
        raise ValueError("source_fps must be greater than zero")
    if frame_count <= 0:
        raise ValueError("frame_count must be greater than zero")
    if trim_start < 0 or trim_end < 0:
        raise ValueError("trim values must be non-negative")

    start_frame = int(round(trim_start * source_fps))
    end_frame = frame_count - int(round(trim_end * source_fps))
    if start_frame >= end_frame:
        raise ValueError("trim settings remove the whole video")
    return start_frame, end_frame


def preprocess_video(
    source: Path,
    output_dir: Path,
    target_fps: int = DEFAULT_TARGET_FPS,
    trim_start: float = DEFAULT_TRIM_SECONDS,
    trim_end: float = DEFAULT_TRIM_SECONDS,
) -> Path:
    source = source.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / build_processed_name(source, target_fps, trim_start, trim_end)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame, end_frame = compute_trim_window(
        frame_count=frame_count,
        source_fps=source_fps,
        trim_start=trim_start,
        trim_end=trim_end,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(target_fps), (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    frame_interval = source_fps / target_fps
    next_keep = float(start_frame)
    frame_index = 0
    written = 0

    try:
        while frame_index < end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index >= start_frame and frame_index + 0.5 >= next_keep:
                writer.write(frame)
                written += 1
                next_keep += frame_interval
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    if written == 0:
        raise RuntimeError(f"No frames were written for: {source}")

    return output_path


def iter_input_videos(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*.mp4"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess three-pen pilot videos.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=int, default=DEFAULT_TARGET_FPS)
    parser.add_argument("--trim-start", type=float, default=DEFAULT_TRIM_SECONDS)
    parser.add_argument("--trim-end", type=float, default=DEFAULT_TRIM_SECONDS)
    args = parser.parse_args()

    videos = iter_input_videos(args.input_dir)
    if not videos:
        raise SystemExit(f"No mp4 files found in {args.input_dir}")

    for video in videos:
        output_path = preprocess_video(
            source=video,
            output_dir=args.output_dir,
            target_fps=args.fps,
            trim_start=args.trim_start,
            trim_end=args.trim_end,
        )
        print(f"{video.name} -> {output_path}")


if __name__ == "__main__":
    main()

