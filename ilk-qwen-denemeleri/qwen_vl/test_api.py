import dashscope
from dashscope import MultiModalConversation
import os
import json
import time
from dotenv import load_dotenv

load_dotenv(override=True)


def _normalize_api_key(raw_key: str | None) -> str:
    if not raw_key:
        return ""
    # Handle accidental quotes/spaces in .env and ensure expected prefix.
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
# s12_deneme_path = "/home/yunus/projects/vad_using_VLMs_graduation_project/145.jpg"
r01_test_06_path1 = "/home/yunus/projects/vad_using_VLMs_graduation_project/036.jpg"
r01_test_06_path2 = "/home/yunus/projects/vad_using_VLMs_graduation_project/057.jpg"
# "image": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"

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

PROMPT2 = "What is the color of the button in the image?"

PROMPT3 = """You are a quality control system on an industrial conveyor belt.

A push button component is moving on the belt.

Step 1 - Observe: Look at the image carefully and answer these two questions:
  a) What color is the button?
  b) Which direction do the button's connector pins point? (left / right / up / forward / diagonal)

Step 2 - Classify using these rules:
  - If the button is RED and pins point LEFT → NORMAL
  - If the pins do NOT point left → ANGLE ANOMALY  
  - If the button is NOT red → COLOR ANOMALY

Reply in this exact format:
Color: <color>
Pins: <direction>
Result: <NORMAL or ANGLE ANOMALY or COLOR ANOMALY>"""


# ==========================================================

_start = time.perf_counter()
response = MultiModalConversation.call(
    model='qwen-vl-plus',
    messages=[
        {
            "role": "user",
            "content": [
                {"image": r01_test_06_path2},
                {"text": PROMPT3}
            ]
        }
    ]
)
_elapsed = time.perf_counter() - _start

print(f"Inference time: {_elapsed:.2f}s")
print("Type of response:", type(response))

print("\n" + "=" * 80)
print("RESPONSE STATUS")
print("=" * 80)
print(f"Status Code: {response.get('status_code')}")
print(f"Request ID: {response.get('request_id')}")
print(f"Message: {response.get('message', 'Success')}")

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
    print(f"Error: {response.get('code')}")
    print(json.dumps(response, indent=2, ensure_ascii=False))