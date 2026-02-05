from ultralytics import YOLO

# 'n' is nano (fastest), 'x' is extra-large (most accurate)
model = YOLO('yolo11n-seg.pt') 

# 2. Run inference on an image
results = model("images.jpg")

# 3. Process results
for result in results:
    result.show()      # Display image with masks & boxes
    result.save(filename="output.jpg")
    
    # Access raw data if needed:
    masks = result.masks.data    # Pixel-level mask tensors
    boxes = result.boxes.xyxy    # Bounding box coordinates
    clss = result.boxes.cls      # Class IDs