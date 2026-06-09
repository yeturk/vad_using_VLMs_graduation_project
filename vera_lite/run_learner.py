import argparse
import json
from pathlib import Path
from typing import Any

from .dashscope_client import call_qwen, extract_json_object
from .prompts import build_learner_prompt
from .video_frames import extract_dense_frames


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "vera_lite" / "guiding_questions.json"


def load_questions(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["questions"]


def run_learner(
    video: str,
    expected: str | None,
    model: str,
    questions_path: Path = DEFAULT_QUESTIONS,
    frames: int = 0,
    start_frac: float = 0.1,
    end_frac: float = 0.9,
    fps: float = 0.0,
    enable_thinking: bool | None = None,
    thinking_budget: int | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    questions = load_questions(questions_path)
    prompt = build_learner_prompt(questions, expected, frames=frames)

    if frames and frames > 0:
        uris = extract_dense_frames(video, frames, start_frac, end_frac)
        content = [{"image": u} for u in uris] + [{"text": prompt}]
        input_mode = f"{len(uris)} frames [{start_frac}-{end_frac}]"
    else:
        # Video mode. fps controls DashScope's internal frame sampling rate
        # (one frame every 1/fps seconds); higher fps captures fast motion.
        video_item: dict[str, Any] = {"video": video}
        if fps and fps > 0:
            video_item["fps"] = fps
        content = [video_item, {"text": prompt}]
        input_mode = f"video fps={fps}" if fps else "video"

    text, raw_response = call_qwen(
        model=model,
        content=content,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
        max_tokens=max_tokens,
    )
    try:
        parsed = extract_json_object(text)
    except Exception as e:  # empty/unparseable output should not kill the whole run
        parsed = {"verdict": "UNCLEAR", "parse_error": str(e)}
    return {
        "video": video,
        "input_mode": input_mode,
        "expected": expected,
        "model": model,
        "questions": questions,
        "raw_text": text,
        "parsed": parsed,
        "usage": raw_response.get("usage", {}),
        "request_id": raw_response.get("request_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--expected", choices=["NORMAL", "ANOMALY"], default=None)
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--frames", type=int, default=0,
                        help="If >0, send this many time-ordered frames instead of the video.")
    parser.add_argument("--start-frac", type=float, default=0.1)
    parser.add_argument("--end-frac", type=float, default=0.9)
    parser.add_argument("--fps", type=float, default=0.0,
                        help="Video mode: DashScope frame sampling rate (frames per second). "
                        "Range 0.1-10. Higher = denser temporal sampling.")
    parser.add_argument("--enable-thinking", dest="enable_thinking", action="store_true", default=None)
    parser.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = run_learner(args.video, args.expected, args.model, args.questions,
                         args.frames, args.start_frac, args.end_frac, args.fps,
                         args.enable_thinking, args.thinking_budget, args.max_tokens)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
