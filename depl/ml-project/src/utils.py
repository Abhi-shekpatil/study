import io
from PIL import Image, ImageDraw, ImageFont


def load_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def draw_detections(image: Image.Image, detections: list) -> Image.Image:
    """Draw bounding boxes and class names on the image"""
    draw = ImageDraw.Draw(image)
    
    # Try to use a default font, fallback to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
    ]
    
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        confidence = det["confidence"]
        class_name = det["class_name"]
        
        # Choose color based on detection index
        color = colors[idx % len(colors)]
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label with background
        label = f"{class_name} ({confidence:.2f})"
        bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1), label, fill=(255, 255, 255), font=font)
    
    return image
