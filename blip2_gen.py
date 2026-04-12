import os
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from PIL import Image
from tqdm import tqdm

# ===== CONFIG =====
model_id = "Salesforce/blip2-opt-2.7b-coco"
image_folder = "images"
output_file = "captions.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== LOAD MODEL =====
processor = AutoProcessor.from_pretrained(model_id)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
model.to(device)

# ===== PROCESS IMAGES =====
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

results = []

for filename in tqdm(os.listdir(image_folder)):
    if not filename.lower().endswith(image_extensions):
        continue

    path = os.path.join(image_folder, filename)

    try:
        image = Image.open(path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=30)

        caption = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        results.append(f"{filename}\t{caption}")
        print(f"{filename}: {caption}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# ===== SAVE RESULTS =====
with open(output_file, "w", encoding="utf-8") as f:
    for line in results:
        f.write(line + "\n")

print(f"\nSaved captions to {output_file}")