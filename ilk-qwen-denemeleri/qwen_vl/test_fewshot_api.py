import dashscope
from dashscope import MultiModalConversation
import os
import time
import base64
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
REFERENCE_NORMAL = "/home/yunus/projects/vad_using_VLMs_graduation_project/078.jpg"
TEST_FRAME       = "/home/yunus/projects/vad_using_VLMs_graduation_project/057.jpg"
MODEL = "qwen-vl-plus"

PROMPT = """The FIRST image is a REFERENCE showing NORMAL operation of the conveyor belt.

The component on the belt is a push button. It has two distinct parts:
- A BLACK plastic housing/body (this is always black — ignore it for color classification)
- A colored CAP on top of the housing (this is the part whose color matters)
- Metal connector PINS extending from the housing

In the REFERENCE (first image), normal condition is:
- The CAP color is RED
- The connector pins point toward the LEFT side of the frame

The SECOND image is the frame you must classify.
Compare the second image carefully to the reference.

Step 1 - Observe the SECOND image:
  a) What color is the CAP (top part) of the button? (ignore the black housing)
  b) Which direction do the connector pins point? (left / right / up / forward / diagonal)

Step 2 - Classify using these rules IN ORDER:
  1. If the CAP is NOT red → COLOR ANOMALY (stop here)
  2. If the CAP IS red but pins do NOT point left → ANGLE ANOMALY
  3. If the CAP IS red AND pins point left → NORMAL

Reply in this exact format:
Cap color: <color>
Pins: <direction>
Result: <NORMAL or ANGLE ANOMALY or COLOR ANOMALY>"""

PROMPT_COMPARE = "What is the difference between the two images? Explain briefly."


def _encode(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def call(prompt: str, label: str) -> None:
    start = time.perf_counter()
    response = MultiModalConversation.call(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": _encode(REFERENCE_NORMAL)},
                    {"image": _encode(TEST_FRAME)},
                    {"text": prompt},
                ]
            }
        ]
    )
    elapsed = time.perf_counter() - start

    print(f"\n{'=' * 60}")
    print(f"TEST: {label}  ({elapsed:.2f}s)")
    print("=" * 60)
    if response.get("status_code") == 200:
        items = response["output"]["choices"][0]["message"]["content"]
        text = next((item["text"] for item in items if "text" in item), "")
        print(text)
        usage = response.get("usage", {})
        print(f"\nTokens: {usage.get('input_tokens')} in / {usage.get('output_tokens')} out")
    else:
        print(f"Error: {response.get('code')}: {response.get('message')}")


print(f"Model: {MODEL}")
print(f"Reference : {os.path.basename(REFERENCE_NORMAL)}")
print(f"Test frame: {os.path.basename(TEST_FRAME)}")

call(PROMPT, "Anomaly Classification")
call(PROMPT_COMPARE, "Image Comparison")
