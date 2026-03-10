"""
Generate LUCI's 3 face expressions via ComfyUI API.
Requires ComfyUI running at localhost:8188
"""
import json, urllib.request, urllib.parse, time, os
from pathlib import Path

COMFY_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = Path.home() / "beast" / "workspace" / "luci_faces"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_PROMPT = (
    "portrait of an elegant futuristic AI woman, LUCI, "
    "sleek silver-white hair, luminescent golden eyes, "
    "flawless skin with subtle glowing circuit markings, "
    "white and gold high-tech collar, soft warm gold rim lighting, "
    "dark background, ultra detailed, 8k, cinematic, hyperrealistic, "
    "sharp focus, professional photography"
)
NEGATIVE = (
    "cartoon, anime, ugly, blurry, extra limbs, watermark, text, "
    "deformed, bad anatomy, disfigured, low quality, nsfw"
)

EXPRESSIONS = [
    {"name": "neutral",   "seed": 42,   "extra": "calm serene expression, slight smile, eyes forward"},
    {"name": "speaking",  "seed": 42,   "extra": "mouth slightly open, engaged animated expression, speaking"},
    {"name": "listening", "seed": 42,   "extra": "head slightly tilted, attentive focused eyes, listening intently"},
]

def queue_prompt(prompt_workflow):
    data = json.dumps({"prompt": prompt_workflow}).encode("utf-8")
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["prompt_id"]

def get_history(prompt_id):
    with urllib.request.urlopen(f"{COMFY_URL}/history/{prompt_id}") as r:
        return json.loads(r.read())

def download_image(filename, subfolder=""):
    params = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": "output"})
    with urllib.request.urlopen(f"{COMFY_URL}/view?{params}") as r:
        return r.read()

def build_workflow(positive, negative, seed, width=768, height=1024):
    return {
        "4": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": positive, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["4", 1]}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["4", 0], "positive": ["6", 0],
                         "negative": ["7", 0], "latent_image": ["5", 0],
                         "seed": seed, "steps": 30, "cfg": 7.5,
                         "sampler_name": "dpmpp_2m", "scheduler": "karras",
                         "denoise": 1.0}},
        "9": {"class_type": "VAEDecode",
              "inputs": {"samples": ["8", 0], "vae": ["4", 2]}},
        "10": {"class_type": "SaveImage",
               "inputs": {"images": ["9", 0], "filename_prefix": "luci"}},
    }

print("🎨 Generating LUCI's face — 3 expressions")
print(f"Output: {OUTPUT_DIR}\n")

for expr in EXPRESSIONS:
    positive = f"{BASE_PROMPT}, {expr['extra']}"
    workflow = build_workflow(positive, NEGATIVE, expr["seed"])
    
    print(f"⏳ Generating: {expr['name']}...", flush=True)
    prompt_id = queue_prompt(workflow)
    
    # Wait for completion
    for _ in range(120):
        time.sleep(2)
        history = get_history(prompt_id)
        if prompt_id in history:
            outputs = history[prompt_id]["outputs"]
            for node_id, node_output in outputs.items():
                if "images" in node_output:
                    img_info = node_output["images"][0]
                    img_data = download_image(img_info["filename"], img_info.get("subfolder",""))
                    out_path = OUTPUT_DIR / f"luci_{expr['name']}.png"
                    out_path.write_bytes(img_data)
                    print(f"✅ Saved: {out_path}")
            break
    else:
        print(f"❌ Timeout waiting for {expr['name']}")

print(f"\n✅ All done. Images in: {OUTPUT_DIR}")
print("Files:")
for f in sorted(OUTPUT_DIR.glob("*.png")):
    size = f.stat().st_size // 1024
    print(f"  {f.name} ({size}KB)")
