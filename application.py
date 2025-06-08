from flask import Flask, request, jsonify
from inference import get_model
import supervision as sv
import cv2, os, base64
import google.generativeai as genai
from collections import Counter
import numpy as np

# Config
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "WhatsApp Image 2025-06-08 at 22.47.02_244ecbae.jpg")
ROBOFLOW_KEY = "aWlghc6MlyWbG65M1SM1"
GEMINI_KEY   = "AIzaSyARrgNRAyYn2J31TKTAjz1cXN0ZMnYBA88"          # generated in AI Studio / MakerSuite
MODEL_ID     = "yolov5.taco/1"            # hosted TACO detector

app = Flask(__name__)

# Load model only once
model = get_model(MODEL_ID, api_key=ROBOFLOW_KEY)

@app.route("/")
def health():
    return "OK", 200


@app.route('/detect-local', methods=['GET'])
def detect_local():
    image_path = os.path.join(os.path.dirname(__file__), "WhatsApp Image 2025-06-08 at 22.47.02_244ecbae.jpg")
    img = cv2.imread(image_path)

    if img is None:
        return jsonify({"error": "Image not found or cannot be read"}), 500

    # Run detection logic (reuse your existing detection block)
    result = model.infer(img)[0]
    classes = [pred.class_name for pred in result.predictions]

    # ... rest of your logic for counting and Gemini prompt
    mapping = {
        "plastic": "plastic", "metal": "metal", "paper": "paper",
        "glass": "glass", "organic": "organic", "food": "organic",
        "cigarette": "cigarette_buds", "styrofoam": "plastic",
        "foil": "metal", "cardboard": "paper",
    }

    bucketed = Counter()
    for cls in classes:
        cls_l = cls.lower()
        bucket = "other"
        for k, v in mapping.items():
            if k in cls_l:
                bucket = v
                break
        bucketed[bucket] += 1

    total = sum(bucketed.values())
    lines = [f"{mat.title():<15}: {cnt/total*100:6.1f}% ({cnt} objects)"
             for mat, cnt in bucketed.most_common()]
    lines.append("-"*30)
    lines.append(f"{'Total':<15}: 100.0%")
    comp_block = "\n".join(lines)

    # Prompt Gemini
    PROMPT = f"""
You are ECO-GPT, an upbeat sustainability analyst with TikTok-level meme awareness.
Produce an environmental report in the EXACT visual style shown below, but ALSO:

• Sneak in 1-2 fresh Gen-Z meme references or catchphrases that are trending right now
  (e.g. “rizz”, “skibidi”, “NPC energy”, “that’s cap”, etc.).  
• The meme lines must feel natural and relate to the waste stats or impact section.
• Do NOT break the section titles, emojis, or the overall order.
• Keep the humour light; avoid anything offensive or obscure.

<EXAMPLE_OUTPUT>
🔍 DETECTING WASTE IN IMAGE...
----------------------------------------
✅ Detected X objects: [...]

📊 WASTE COMPOSITION:
----------------------------------------
Plastic        :  90.9% (10 objects)
Paper          :   9.1% (1 objects)
----------------------------------------
Total          : 100.0%

🌍 YOUR ENVIRONMENTAL IMPACT:
----------------------------------------
  🚗 Prevented emissions = ... km of car driving!
  🚿 Water saved = ... showers!
  🐠 You potentially saved ... marine animals!
  🌲 You saved ... trees!

💡 PREVENTION TIPS FOR YOUR WASTE:
----------------------------------------
  💡 <one-line actionable tip>

🎮 YOUR ECO SCORE: ... points!
----------------------------------------
🏆 ACHIEVEMENT UNLOCKED: Eco Warrior!
</EXAMPLE_OUTPUT>

Now produce the same style of report—meme sprinkle included—for the MATERIAL COMPOSITION below.

MATERIAL COMPOSITION:
------------------------------
{comp_block}
"""


    genai.configure(api_key=GEMINI_KEY)
    report = genai.GenerativeModel("gemini-1.5-flash").generate_content(PROMPT).text

    return jsonify({"report": report})


@app.route('/detect', methods=['POST'])
def detect():
    # Image from base64 string
    data = request.json
    image_b64 = data.get("image_base64")

    if not image_b64:
        return jsonify({"error": "Missing image_base64"}), 400

    image_data = base64.b64decode(image_b64)
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Inference
    result = model.infer(img)[0]
    classes = [pred.class_name for pred in result.predictions]

    mapping = {
        "plastic": "plastic", "metal": "metal", "paper": "paper",
        "glass": "glass", "organic": "organic", "food": "organic",
        "cigarette": "cigarette_buds", "styrofoam": "plastic",
        "foil": "metal", "cardboard": "paper",
    }

    bucketed = Counter()
    for cls in classes:
        cls_l = cls.lower()
        bucket = "other"
        for k, v in mapping.items():
            if k in cls_l:
                bucket = v
                break
        bucketed[bucket] += 1

    total = sum(bucketed.values())
    lines = [f"{mat.title():<15}: {cnt/total*100:6.1f}% ({cnt} objects)"
             for mat, cnt in bucketed.most_common()]
    lines.append("-"*30)
    lines.append(f"{'Total':<15}: 100.0%")
    comp_block = "\n".join(lines)

    # Prompt Gemini
    PROMPT = f"""
You are ECO-GPT, an upbeat sustainability analyst with TikTok-level meme awareness.
Produce an environmental report in the EXACT visual style shown below, but ALSO:

• Sneak in 1-2 fresh Gen-Z meme references or catchphrases that are trending right now
  (e.g. “rizz”, “skibidi”, “NPC energy”, “that’s cap”, etc.).  
• The meme lines must feel natural and relate to the waste stats or impact section.
• Do NOT break the section titles, emojis, or the overall order.
• Keep the humour light; avoid anything offensive or obscure.

<EXAMPLE_OUTPUT>
🔍 DETECTING WASTE IN IMAGE...
----------------------------------------
✅ Detected X objects: [...]

📊 WASTE COMPOSITION:
----------------------------------------
Plastic        :  90.9% (10 objects)
Paper          :   9.1% (1 objects)
----------------------------------------
Total          : 100.0%

🌍 YOUR ENVIRONMENTAL IMPACT:
----------------------------------------
  🚗 Prevented emissions = ... km of car driving!
  🚿 Water saved = ... showers!
  🐠 You potentially saved ... marine animals!
  🌲 You saved ... trees!

💡 PREVENTION TIPS FOR YOUR WASTE:
----------------------------------------
  💡 <one-line actionable tip>

🎮 YOUR ECO SCORE: ... points!
----------------------------------------
🏆 ACHIEVEMENT UNLOCKED: Eco Warrior!
</EXAMPLE_OUTPUT>

Now produce the same style of report—meme sprinkle included—for the MATERIAL COMPOSITION below.

MATERIAL COMPOSITION:
------------------------------
{comp_block}
"""


    genai.configure(api_key=GEMINI_KEY)
    report = genai.GenerativeModel("gemini-1.5-flash").generate_content(PROMPT).text

    return jsonify({"report": report})