import os
import cv2
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

def test_pipeline_on_sample_image():
    # Load sample image (use a small truck photo if available)
    sample_path = "input/sample.jpg"
    assert os.path.exists(sample_path), "Sample image not found."

    # Load YOLOv8 and BLIP
    model_yolo = YOLO('yolov8n.pt')
    model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    # Load image using OpenCV and PIL
    img_cv = cv2.imread(sample_path)
    assert img_cv is not None, "Failed to load sample image with OpenCV."

    pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    # Run object detection
    results = model_yolo(img_cv)
    detected_objects = [model_yolo.names[int(cls)] for cls in results[0].boxes.cls]
    assert isinstance(detected_objects, list), "YOLO output is not a list."
    print("✅ YOLO detection successful:", detected_objects)

    # Run caption generation
    inputs = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        output = model_blip.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    assert isinstance(caption, str) and len(caption) > 0, "BLIP failed to generate caption."
    print("✅ Caption generated:", caption)

if __name__ == "__main__":
    test_pipeline_on_sample_image()
