import dashscope
from dashscope import MultiModalConversation
import os
import json
import sys
import cv2
import base64
import numpy as np
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
print("Using DashScope endpoint:", dashscope.base_http_api_url)
print("API key suffix:", f"***{api_key[-4:]}")


VIDEO_SOURCE = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241127/5hnfnc/3.mp4"
)

print(f"\nVideo source: {VIDEO_SOURCE}")


# ─────────────────────────────────────────────
# TEMPORAL CONTEXT — 2×2 GRID IMAGE (AnyAnomaly TC)
# ─────────────────────────────────────────────

def build_temporal_grid(video_path: str, debug_save: bool = True) -> str:
    """
    AnyAnomaly TC yaklaşımı.
    Videonun %20-%80 aralığından 4 frame seçer (boş baş/son kareleri atlar).
    Frame'lere numara etiketi ve bant yönü oku ekler.
    2×2 grid image oluşturup base64 döner.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps

    # Videonun tamamından 4 eşit aralıklı frame (giriş ve çıkış anları dahil)
    indices = [int(i * (total_frames - 1) / 3) for i in range(4)]

    print(f"Video süresi  : {duration:.2f} sn  |  FPS: {fps}  |  Toplam: {total_frames} frame")
    print(f"Seçilen indeksler: {indices}")

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError(f"Frame #{idx} alınamadı.")
        frames.append(frame)
    cap.release()

    h, w = frames[0].shape[:2]
    frames = [cv2.resize(f, (w, h)) for f in frames]

    # Her frame'e sıra numarası ve bant yönü oku ekle
    labels = ["Frame 1 (earliest)", "Frame 2", "Frame 3", "Frame 4 (latest)"]
    font   = cv2.FONT_HERSHEY_SIMPLEX
    for f, label in zip(frames, labels):
        # Üst etiket: frame numarası
        cv2.rectangle(f, (0, 0), (w, 28), (0, 0, 0), -1)
        cv2.putText(f, label, (6, 20), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        # Alt etiket: bant yönü
        cv2.rectangle(f, (0, h - 28), (w, h), (0, 0, 0), -1)
        cv2.putText(f, "BELT: LEFT --> RIGHT", (6, h - 8), font, 0.45, (0, 220, 0), 1, cv2.LINE_AA)

    # 2×2 grid: [F1 F2 / F3 F4]
    top    = np.hstack([frames[0], frames[1]])
    bottom = np.hstack([frames[2], frames[3]])
    grid   = np.vstack([top, bottom])

    if debug_save:
        cv2.imwrite("temporal_grid.jpg", grid)
        print("Grid kaydedildi → temporal_grid.jpg")

    _, buffer = cv2.imencode(".jpg", grid)
    return base64.b64encode(buffer).decode("utf-8")


# ─────────────────────────────────────────────
# PROMPT  (AnyAnomaly tarzı: skor + reasoning + consideration)
# ─────────────────────────────────────────────

PROMPT = """
The image is a 2×2 grid of 4 frames from a single video clip, in chronological order:
  top-left = Frame 1 (earliest)  →  top-right = Frame 2
  bottom-left = Frame 3          →  bottom-right = Frame 4 (latest)

SCENE: A push-button component travels on a green conveyor belt that moves strictly LEFT → RIGHT.
The button has two distinct sides:
  • RED CAP — round, red-colored top
  • PIN SIDE — black base with 4 metal pins protruding

─────────────────────────────────────────────
HOW TO READ ENTRY AND EXIT FRAMES (key insight):
─────────────────────────────────────────────
Because the belt moves LEFT → RIGHT:

  AT ENTRY (button appearing from the left edge):
    → The side you see FIRST at the left edge is the TRAILING side.
    → The side still hidden (further right on the button body) is the LEADING side.

  AT EXIT (button disappearing off the right edge):
    → The side that DISAPPEARS FIRST off the right edge is the LEADING side.
    → The side still visible last on the right is... wait, no:
    → The side that leaves the frame first (on the right) = LEADING side.
    → The side that lingers longest in frame = TRAILING side.

  IN THE MIDDLE (button fully visible):
    → The side positioned closer to the RIGHT edge of the frame = LEADING side.
    → The side positioned closer to the LEFT edge of the frame = TRAILING side.

─────────────────────────────────────────────
CORRECT vs. ANOMALY:
─────────────────────────────────────────────
  ✅ CORRECT  → RED CAP is the LEADING side (closer to right / exits first / appears last at entry)
  ❌ ANOMALY  → PIN SIDE is the LEADING side (red cap trails behind)

─────────────────────────────────────────────
YOUR TASK:
─────────────────────────────────────────────
Step 1 — Analyze each frame using the rules above:
  Frame 1: Is the button entering? If so, which side appears at the left edge first?
  Frame 2: Which side is closer to the right edge of the frame?
  Frame 3: Which side is closer to the right edge of the frame?
  Frame 4: Is the button exiting? If so, which side disappears off the right edge first?

Step 2 — Based on all frames, assign an anomaly score:
  0.0 = clearly correct orientation (RED CAP leads)
  1.0 = clearly reversed orientation (PIN SIDE leads) → anomaly
  0.5 = cannot determine

Step 3 — Output:
FRAME ANALYSIS:
  Frame 1: [observation]
  Frame 2: [observation]
  Frame 3: [observation]
  Frame 4: [observation]
LEADING SIDE IDENTIFIED: [RED CAP / PIN SIDE / uncertain]
ANOMALY SCORE: [0.0 – 1.0]
VERDICT: ✅ NO ANOMALY / ❌ ANOMALY DETECTED
REASON: [one sentence]
"""


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

print("\nTemporal grid oluşturuluyor...")
grid_b64 = build_temporal_grid(VIDEO_SOURCE)

print("\nModel çağrılıyor...")
response = MultiModalConversation.call(
    model="qwen3.6-plus",
    messages=[
        {
            "role": "user",
            "content": [
                {"image": f"data:image/jpeg;base64,{grid_b64}"},
                {"text": PROMPT},
            ],
        }
    ],
)

print("\n" + "=" * 80)
print("RESPONSE STATUS")
print("=" * 80)
print(f"Status Code: {response.get('status_code')}")
print(f"Request ID:  {response.get('request_id')}")
print(f"Message:     {response.get('message', 'Success')}")

print("\n" + "=" * 80)
print("MODEL RESPONSE")
print("=" * 80)

if response.get("status_code") == 200:
    output  = response.get("output", {})
    choices = output.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content", [])
        for item in content:
            if isinstance(item, dict) and "text" in item:
                print(f"\n{item['text']}")

    usage = response.get("usage", {})
    print("\n" + "-" * 80)
    print("USAGE STATISTICS")
    print("-" * 80)
    print(f"Input tokens:  {usage.get('input_tokens')}")
    print(f"Output tokens: {usage.get('output_tokens')}")
    print(f"Total tokens:  {usage.get('total_tokens')}")
else:
    print(f"Error code: {response.get('code')}")
    print(json.dumps(dict(response), indent=2, ensure_ascii=False))
