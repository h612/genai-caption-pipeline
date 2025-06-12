import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
import torch
import os
import json

model_yolo = YOLO('yolov8n.pt')
model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

captions = {}

for fname in os.listdir("input"):
    if not fname.endswith((".jpg", ".png")):
        continue

    img_path = os.path.join("input", fname)
    img_cv = cv2.imread(img_path)

    # Object detection
    results = model_yolo(img_cv)
    objects = [model_yolo.names[int(cls)] for cls in results[0].boxes.cls]

    # Captioning
    pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        output = model_blip.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    captions[fname] = {
        "caption": caption,
        "objects": objects
    }

# Save output
with open("output/captions.json", "w") as f:
    json.dump(captions, f, indent=4)
