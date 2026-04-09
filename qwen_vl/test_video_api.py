import dashscope
from dashscope import MultiModalConversation
import os, time, base64, tempfile
import numpy as np
import cv2
from dotenv import load_dotenv

load_dotenv(override=True)


def _normalize_api_key(raw_key: str | None) -> str:
    if not raw_key:
        return ""
    key = raw_key.strip().strip('"').strip("'")
    if key and not key.startswith("sk-"):
        key = f"sk-{key}"
    return key


api_key = _normalize_api_key(os.getenv("DASHSCOPE_API_KEY"))
if not api_key:
    raise RuntimeError("DASHSCOPE_API_KEY is missing. Check your .env file.")

dashscope.api_key = api_key
dashscope.base_http_api_url = os.getenv(
    "DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/api/v1"
).strip()

# --- CONFIG ---
FRAMES_DIR = "/home/yunus/projects/vad_using_one_shot_learning/dataset/IPAD_dataset/R01/testing/frames/06"
LABEL_PATH = "/home/yunus/projects/vad_using_one_shot_learning/dataset/IPAD_dataset/R01/test_label/006.npy"
SAMPLE_STEP = 5
SEQUENCE_LEN = 10
SKIP_FIRST_N = 0
VIDEO_FPS = 4
MODEL = "qwen-vl-max"

PROMPT = """You are a quality control system monitoring an industrial conveyor belt.

The belt transports small electronic push buttons (components used in circuit boards).

=== NORMAL CONDITION ===
- The button traveling on the belt is RED in color.
- The button's connector pins (legs) point toward the LEFT side of the frame,
  which is the backward direction relative to the belt's movement.
- The button moves continuously from right to left across the belt.

=== ANOMALY TYPES ===
- ANGLE ANOMALY: The button is rotated incorrectly. Its connector pins do NOT point
  to the left — they may point forward, upward, diagonally, or in any direction
  other than left. The button's body orientation looks tilted or rotated compared
  to the normal position.
- COLOR ANOMALY: The button is NOT red. It may appear green, black, white, or
  any other color.

=== YOUR TASK ===
Watch the video carefully. Focus on:
1. The orientation of the button's connector pins — do they point LEFT?
2. The color of the button — is it RED?

Reply with exactly one of:
- NORMAL
- ANGLE ANOMALY
- COLOR ANOMALY"""


def get_sequence_ground_truth(frame_indices: list[int], labels: np.ndarray) -> int:
    return int(any(labels[i] == 1.0 for i in frame_indices if i < len(labels)))


def frames_to_video(frame_paths: list[str], fps: int) -> str:
    sample = cv2.imread(frame_paths[0])
    h, w = sample.shape[:2]
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, fps, (w, h))
    for p in frame_paths:
        frame = cv2.imread(p)
        writer.write(frame)
    writer.release()
    return tmp.name


def _encode_video(path: str) -> str:
    with open(path, "rb") as f:
        return "data:video/mp4;base64," + base64.b64encode(f.read()).decode()


def call_api(frame_paths: list[str]) -> tuple[str, float]:
    video_path = frames_to_video(frame_paths, VIDEO_FPS)
    try:
        content = [
            {"video": _encode_video(video_path)},
            {"text": PROMPT},
        ]
        start = time.perf_counter()
        response = MultiModalConversation.call(
            model=MODEL,
            messages=[{"role": "user", "content": content}]
        )
        elapsed = time.perf_counter() - start
    finally:
        os.unlink(video_path)

    if response.get("status_code") == 200:
        items = response["output"]["choices"][0]["message"]["content"]
        text = next((item["text"] for item in items if "text" in item), "")
    else:
        text = f"ERROR {response.get('code')}: {response.get('message')}"

    return text, elapsed


def main():
    labels = np.load(LABEL_PATH)
    all_frames = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg"))
    sampled_frames = all_frames[::SAMPLE_STEP][SKIP_FIRST_N:]

    total_seqs = len(sampled_frames) - SEQUENCE_LEN + 1
    print(f"Total frames: {len(all_frames)}")
    print(f"Sampled (every {SAMPLE_STEP}th, skipping first {SKIP_FIRST_N}): {len(sampled_frames)}")
    print(f"Sequences (sliding window, len={SEQUENCE_LEN}): {total_seqs}")
    print(f"Model: {MODEL} | Video FPS: {VIDEO_FPS}")
    print("=" * 80)

    correct = 0
    total = 0

    for seq_idx in range(total_seqs):
        seq_frames = sampled_frames[seq_idx:seq_idx + SEQUENCE_LEN]
        frame_indices = [int(f.replace(".jpg", "")) for f in seq_frames]
        frame_paths = [os.path.join(FRAMES_DIR, f) for f in seq_frames]

        gt = get_sequence_ground_truth(frame_indices, labels)
        gt_str = "ANOMALY" if gt else "NORMAL"

        print(f"\nSeq {seq_idx + 1:02d}/{total_seqs} | frames {frame_indices[0]:03d}–{frame_indices[-1]:03d} | GT: {gt_str}")

        response_text, elapsed = call_api(frame_paths)

        prediction = response_text.strip().split("\n")[0].upper().strip("- ").strip()
        predicted_anomaly = "NORMAL" not in prediction
        match = predicted_anomaly == bool(gt)
        correct += match
        total += 1

        print(f"  {elapsed:.2f}s | Prediction: {prediction} | {'✓' if match else '✗'}")

    print("\n" + "=" * 80)
    print(f"RESULTS: {correct}/{total} correct ({100 * correct / total:.1f}%)")


if __name__ == "__main__":
    main()
