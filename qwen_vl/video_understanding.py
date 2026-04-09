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

VIDEO_PATH = "/home/yunus/projects/vad_using_VLMs_graduation_project/data/ornek_video_720p_25fps.mp4"
MODEL = "qwen-vl-max"

PROMPT = "What do you see in this video? Describe it in detail."

size_mb = os.path.getsize(VIDEO_PATH) / (1024 * 1024)
print(f"Model     : {MODEL}")
print(f"Video     : {os.path.basename(VIDEO_PATH)} ({size_mb:.1f} MB)")

if size_mb > 20:
    print(f"WARNING   : Video is {size_mb:.1f} MB — base64 payload will be very large (~{size_mb * 1.33:.0f} MB). API may reject it.")

print("=" * 60)

with open(VIDEO_PATH, "rb") as f:
    video_b64 = "data:video/mp4;base64," + base64.b64encode(f.read()).decode()

start = time.perf_counter()
response = MultiModalConversation.call(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {"video": video_b64},
                {"text": PROMPT},
            ]
        }
    ]
)
elapsed = time.perf_counter() - start

print(f"Inference time: {elapsed:.2f}s\n")

if response.get("status_code") == 200:
    items = response["output"]["choices"][0]["message"]["content"]
    text = next((item["text"] for item in items if "text" in item), "")
    print(text)
    usage = response.get("usage", {})
    print(f"\nTokens     : {usage.get('input_tokens')} in / {usage.get('output_tokens')} out")
    print(f"Video tokens: {usage.get('video_tokens', 'N/A')}")
else:
    print(f"Error {response.get('status_code')}: {response.get('code')}")
    print(response.get("message"))
