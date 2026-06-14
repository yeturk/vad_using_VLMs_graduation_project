"""Decisive test: VIDEO mode + high fps + the dense-frame relative-spacing prompt.

Hypothesis: does feeding the video at a higher sampling rate (fps) AND asking the
dense-frame style relative-spacing question let video mode catch the intra-object
temporal anomaly it otherwise misses? If yes, video alone could replace the
dense-frame view; if no, it confirms the bottleneck is presentation (discrete
numbered frames + explicit comparison), not frame count.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vera_lite.dashscope_client import call_qwen, extract_json_object

PROMPT = """
You are an industrial quality-control vision system.

A white board slides LEFT to RIGHT carrying three pens in a row. Because the whole
board translates, all three pens move together across the frame - that is normal.

Watch the video carefully over time. Decide whether the three pens stay RIGID as a
group on the board, or whether any pen moves ON ITS OWN relative to the others / the
board.

Focus on RELATIVE cues that are independent of the board's overall left-to-right travel:
- Spacing: does the gap between adjacent pens stay constant over time, or does the
  spacing between any two pens visibly change (one pen drifting closer to or farther
  from a neighbor)?
- Arrangement: do all three pens keep the same relative positions, or does one pen
  shift, slide, rotate, or jump relative to the other two?
- A pen that moves differently from the group (not merely carried along by the board)
  is an ANOMALY of type unexpected_pen_movement.

Do NOT mistake these normal effects for relative movement:
- The whole group translating left to right (the board carrying all pens together).
- Pens leaving the frame at the right edge as the board exits.

Output strictly as valid JSON:
{
  "observations": "how the pen spacing / relative arrangement evolves over time",
  "relative_movement_detected": true,
  "verdict": "NORMAL / ANOMALY",
  "anomaly_score": 0.0,
  "confidence": "LOW / MEDIUM / HIGH",
  "reason": "one or two sentences"
}
Return JSON only. Do not include markdown.
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument("--thinking-budget", type=int, default=1500)
    parser.add_argument("--max-tokens", type=int, default=3500)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    content = [{"video": args.video, "fps": args.fps}, {"text": PROMPT}]
    print(f"Video: {args.video} | fps={args.fps}")
    text, raw = call_qwen(model=args.model, content=content,
                          thinking_budget=args.thinking_budget, max_tokens=args.max_tokens)
    try:
        parsed = extract_json_object(text)
        print(f"verdict={parsed.get('verdict')} | score={parsed.get('anomaly_score')} "
              f"| rel_move={parsed.get('relative_movement_detected')} | conf={parsed.get('confidence')}")
        print(f"reason: {parsed.get('reason')}")
    except Exception as e:
        parsed = {"verdict": "UNCLEAR", "parse_error": str(e)}
        print(f"[parse error] text len={len(text)}")

    u = raw.get("usage", {})
    print(f"tokens: video={u.get('video_tokens')} out={u.get('output_tokens')} "
          f"reasoning={u.get('output_tokens_details',{}).get('reasoning_tokens')}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"video": args.video, "fps": args.fps, "raw_text": text,
             "parsed": parsed, "usage": u}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
