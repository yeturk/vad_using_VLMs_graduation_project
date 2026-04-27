import dashscope
from dashscope import MultiModalConversation
import os
import json
import sys
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


# Video source: pass a URL as CLI arg, or fall back to the default test video.
# Usage:
#   python test_api_2.py                          # uses default test video URL
#   python test_api_2.py https://...              # uses given URL
#   python test_api_2.py /path/to/local/video.mp4 # uses local file (auto-uploaded to OSS)
VIDEO_SOURCE = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241127/5hnfnc/3.mp4"
)

print(f"\nVideo source: {VIDEO_SOURCE}")


TEXT_PROMPT = """
You are a quality control system inspecting push-button components on a conveyor belt.

YOUR TASK:
Inspect the color of the button's TOP CAP in the image.

CORRECT color:
- The top cap must be RED

ANOMALY colors (anything other than red):
- Blue, green, yellow, orange, purple, black, white, gray, or any other non-red color

STEPS:
1. Locate the button's top cap in the image.
2. Identify its color as precisely as possible.
3. Determine if it is red or not.

ANSWER FORMAT (strictly follow this):
CAP COLOR OBSERVED: [exact color you see]
IS IT RED?: YES / NO
VERDICT: ✅ NO ANOMALY / ❌ ANOMALY DETECTED
REASON: [one sentence explanation]
"""

VIDEO_PROMPT = """
You are a quality control vision system analyzing a video of a conveyor belt production line.
You will receive multiple frames from a single video clip. Your task is to track ONE push-button
component across its ENTIRE journey on the belt — from when it first appears to when it leaves —
and deliver a SINGLE final verdict at the end.

═══════════════════════════════════════════════════════
COMPONENT DESCRIPTION
═══════════════════════════════════════════════════════
The component is a push-button with two distinct sides:
  • RED CAP SIDE   → round, red-colored top cap
  • PIN SIDE       → black base with 4 metal pins protruding

═══════════════════════════════════════════════════════
BELT & CAMERA SETUP
═══════════════════════════════════════════════════════
  • The belt moves from RIGHT → LEFT in the camera frame.
  • "RIGHT side" = right edge of the image
  • "LEFT side"  = left edge of the image
  • The camera is fixed and shoots from directly above or slight angle.

═══════════════════════════════════════════════════════
✅ CORRECT ORIENTATION — MEMORIZE THIS
═══════════════════════════════════════════════════════
  RED CAP   → faces RIGHT side of the image  (right side of the button)
  PIN SIDE  → faces LEFT side of the image   (left side of the button)

  In other words: as the button travels left, the PIN SIDE leads and the RED CAP trails.

❌ ANOMALY ORIENTATION
  PIN SIDE  → on the RIGHT (trailing)
  RED CAP   → on the LEFT  (leading)

═══════════════════════════════════════════════════════
PHASE 1 — TRACK (do NOT give verdict yet)
═══════════════════════════════════════════════════════
Watch ALL frames carefully before making any judgment.

For each phase of the journey, note:

  [ENTRY]
  - Is the button fully visible yet? (yes/no)
  - Which side is on the RIGHT? Which side is on the LEFT?

  [MIDDLE — fully visible frames]
  - Confirm orientation: RED CAP on RIGHT? PIN SIDE on LEFT?
  - Is motion smooth and continuous?
  - Does the button stay on the belt surface?

  [EXIT]
  - Did orientation remain consistent throughout?
  - Any last-moment anomaly (tilting, sliding off)?

If the button is NEVER fully visible across all frames:
  → State: "Button not fully visible in any frame — verdict cannot be given." and STOP.

═══════════════════════════════════════════════════════
PHASE 2 — FINAL VERDICT (after observing the full journey)
═══════════════════════════════════════════════════════
Based on the ENTIRE observed journey, answer:

Q1. ORIENTATION (most representative fully-visible frame):
    → ✅ RED CAP on RIGHT, PIN SIDE on LEFT   (correct)
    → ❌ PIN SIDE on RIGHT, RED CAP on LEFT   (reversed — ANOMALY)

Q2. MOTION:
    → ✅ SMOOTH
    → ❌ NOT SMOOTH — describe: [stopped / tilted / fell]

Q3. BELT POSITION:
    → ✅ ON BELT throughout
    → ❌ OFF / NEAR EDGE — describe when and where

═══════════════════════════════════════════════════════
PHASE 3 — VERDICT STATEMENT
═══════════════════════════════════════════════════════
Trigger ANOMALY only if AT LEAST ONE of these is confirmed:
  1. PIN SIDE is clearly on the RIGHT (button is reversed)
  2. Button visibly stopped, tipped over, or fell
  3. Button slid off or hung over the belt edge

State one of:
  ✅ NO ANOMALY DETECTED
  ❌ ANOMALY DETECTED

═══════════════════════════════════════════════════════
PHASE 4 — ANOMALY REPORT (skip if no anomaly; write "None")
═══════════════════════════════════════════════════════
  ANOMALY TYPE     : reversed orientation / stopped / fell off belt
  DESCRIPTION      : What exactly did you observe?
  OBSERVED POSITION: RED CAP was on [LEFT/RIGHT], PIN SIDE was on [LEFT/RIGHT]
  WHEN             : Entry / Middle / Exit phase of the journey
  LIKELY CAUSE     : component placed backwards / belt vibration / operator error
  SEVERITY         : LOW / MEDIUM / HIGH

═══════════════════════════════════════════════════════
RULES — DO NOT FLAG THESE AS ANOMALIES
═══════════════════════════════════════════════════════
  • Partial visibility at entry or exit frames
  • Slight wobble or minor vibration
  • Slightly off-center position on belt width
  Only flag CLEAR and DEFINITIVE violations listed above.
"""


response = MultiModalConversation.call(
    model="qwen3.6-plus",
    messages=[
        {
            "role": "user",
            "content": [
                {"video": VIDEO_SOURCE},
                {"text": VIDEO_PROMPT},
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
    output = response.get("output", {})
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
