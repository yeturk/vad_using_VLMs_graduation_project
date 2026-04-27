import dashscope
from dashscope import MultiModalConversation
import os, time, base64
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
    raise RuntimeError("DASHSCOPE_API_KEY is missing.")

dashscope.api_key = api_key
dashscope.base_http_api_url = os.getenv(
    "DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/api/v1"
).strip()

VIDEO_PATH = "/mnt/c/Users/Alper Kaan/Desktop/vad_using_VLMs_graduation_project/data/r01_clip06_angle_anomaly.mp4"
MODEL = "qwen-vl-max"

PROMPT = """Watch this short video carefully.

Describe what you see:
1. What object is on the conveyor belt?
2. What color is it?
3. How is it oriented? Which direction do its pins/legs point?
4. Does anything change across the video or does it stay consistent?

Be specific and concise."""


def _encode_video(path: str) -> str:
    with open(path, "rb") as f:
        return "data:video/mp4;base64," + base64.b64encode(f.read()).decode()


print(f"Model: {MODEL}")
print(f"Video: {os.path.basename(VIDEO_PATH)} ({os.path.getsize(VIDEO_PATH) // 1024} KB)")
print("=" * 60)

start = time.perf_counter()
response = MultiModalConversation.call(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {"video": _encode_video(VIDEO_PATH)},
                {"text": PROMPT},
            ]
        }
    ]
)
elapsed = time.perf_counter() - start

print(f"Inference time: {elapsed:.2f}s")
print()

if response.get("status_code") == 200:
    items = response["output"]["choices"][0]["message"]["content"]
    text = next((item["text"] for item in items if "text" in item), "")
    print(text)
    usage = response.get("usage", {})
    print(f"\nTokens: {usage.get('input_tokens')} in / {usage.get('output_tokens')} out")
    print(f"Video tokens: {usage.get('video_tokens', 'N/A')}")
else:
    print(f"Error {response.get('status_code')}: {response.get('code')}")
    print(response.get("message"))
