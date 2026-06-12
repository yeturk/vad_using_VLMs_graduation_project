"""Temporal-grid VLM probe.

Tile N time-ordered frames into a single labeled montage image and send it to the
VLM in ONE call. Laying the whole clip out spatially lets the VLM compare moments
side by side (its strength), so it can judge board motion (stall) and relative pen
arrangement (intra-object movement) while still seeing static cues (count/color/
cap/orientation) in the clear cells — all in a single pass, no window cropping.
"""
import argparse
import base64
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vera_lite.dashscope_client import call_qwen, extract_json_object
from vera_lite.prompts import SYSTEM_CONTEXT, format_questions
from vera_lite.run_learner import load_questions

ROOT = PROJECT_ROOT
DEFAULT_QUESTIONS = ROOT / "vera_lite" / "guiding_questions_v3.json"


def build_grid(video: str, rows: int, cols: int, cell_h: int):
    n = rows * cols
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = [int(round(i * (total - 1) / (n - 1))) for i in range(n)]

    cells = []
    for k, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            frame = cells[-1].copy() if cells else None
            if frame is None:
                continue
        h, w = frame.shape[:2]
        cw = int(round(w * cell_h / h))
        cell = cv2.resize(frame, (cw, cell_h), interpolation=cv2.INTER_AREA)
        # number label (white text on black box, top-left)
        cv2.rectangle(cell, (0, 0), (46, 34), (0, 0, 0), -1)
        cv2.putText(cell, str(k + 1), (6, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cells.append(cell)
    cap.release()

    maxw = max(c.shape[1] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, 0, 0, maxw - c.shape[1],
                                cv2.BORDER_CONSTANT, value=(40, 40, 40)) for c in cells]
    while len(cells) < rows * cols:  # pad missing cells
        cells.append((cells[0] * 0))
    row_imgs = [cv2.hconcat(cells[r * cols:(r + 1) * cols]) for r in range(rows)]
    return cv2.vconcat(row_imgs)


def build_grid_prompt(questions: list[str], rows: int, cols: int) -> str:
    n = rows * cols
    grid_note = (
        f"Input: a single image that is a {rows}x{cols} GRID of {n} video frames "
        f"sampled in time order from ONE short clip. Read the grid left-to-right, "
        f"top-to-bottom: cell 1 (top-left) is the earliest moment, cell {n} "
        f"(bottom-right) is the latest. Each cell is numbered. Treat the cells as a "
        f"temporal sequence: track the SAME pens across cells to judge the board's "
        f"motion (does it keep advancing left-to-right, or stall/not advance across "
        f"some cells?) and any change in the pens' relative arrangement or spacing "
        f"across cells. Pens entering at the left or leaving at the right edge as the "
        f"board travels is normal board motion, not an anomaly."
    )
    schema = """Task:
1. Use the grid to answer each guiding question.
2. Decide whether the clip is NORMAL or ANOMALY.
3. Give an anomaly score from 0.0 to 1.0.

Output strictly as valid JSON:
{
  "question_answers": [
    {"question_id": 1, "answer": "YES / NO / UNCLEAR", "evidence": "short, cite cell numbers"}
  ],
  "verdict": "NORMAL / ANOMALY / UNCLEAR",
  "anomaly_type": "none / missing_cap / wrong_orientation / wrong_color / missing_pen / motion_failure / unexpected_pen_movement / multiple / unclear",
  "anomaly_score": 0.0,
  "confidence": "LOW / MEDIUM / HIGH",
  "reason": "one or two sentences"
}

Return JSON only. Do not include markdown."""
    return (f"{SYSTEM_CONTEXT}\n\n{grid_note}\n\nCurrent guiding questions:\n"
            f"{format_questions(questions)}\n\n{schema}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--cell-height", type=int, default=420)
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--thinking-budget", type=int, default=1500)
    parser.add_argument("--max-tokens", type=int, default=3500)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--save-grid", type=Path, default=None)
    args = parser.parse_args()

    grid = build_grid(args.video, args.rows, args.cols, args.cell_height)
    if args.save_grid:
        args.save_grid.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_grid), grid)

    ok, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    uri = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    questions = load_questions(args.questions)
    prompt = build_grid_prompt(questions, args.rows, args.cols)

    print(f"Video: {args.video} | grid {args.rows}x{args.cols} {grid.shape[1]}x{grid.shape[0]}")
    text, raw = call_qwen(model=args.model, content=[{"image": uri}, {"text": prompt}],
                          thinking_budget=args.thinking_budget, max_tokens=args.max_tokens)
    try:
        parsed = extract_json_object(text)
        print(f"verdict={parsed.get('verdict')} | type={parsed.get('anomaly_type')} "
              f"| score={parsed.get('anomaly_score')} | conf={parsed.get('confidence')}")
        print(f"reason: {parsed.get('reason')}")
    except Exception as e:
        parsed = {"verdict": "UNCLEAR", "parse_error": str(e)}
        print(f"[parse error] raw_text empty? len={len(text)}")

    usage = raw.get("usage", {})
    print(f"tokens: out={usage.get('output_tokens')} reasoning={usage.get('output_tokens_details',{}).get('reasoning_tokens')}")

    if args.out:
        import json
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"video": args.video, "grid": f"{args.rows}x{args.cols}",
             "raw_text": text, "parsed": parsed, "usage": usage},
            ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
