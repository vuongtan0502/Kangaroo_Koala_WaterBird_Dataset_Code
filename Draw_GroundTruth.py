import os
import json
import cv2
from collections import defaultdict

# === Paths ===
json_path = 'ground_truth.json'
image_folder = 'input_images'
output_folder = 'output_images'
os.makedirs(output_folder, exist_ok=True)

# === Load JSON ===
with open(json_path, 'r') as f:
    data = json.load(f)

# === Map image_id to filename ===
image_id_to_file = {img['id']: img['file_name'] for img in data['images']}

# === Map category_id to name ===
category_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

# === Group annotations by image_id ===
annotations_by_image = defaultdict(list)
for ann in data['annotations']:
    annotations_by_image[ann['image_id']].append(ann)

# === Draw boxes and save ===
for image_id, anns in annotations_by_image.items():
    filename = image_id_to_file[image_id]
    image_path = os.path.join(image_folder, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Warning: Image not found {filename}")
        continue

    for ann in anns:
        x, y, w, h = map(int, ann['bbox'])
        cat_id = ann['category_id']
        label = category_id_to_name.get(cat_id, 'Unknown')

        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save to output folder
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, image)

print("✅ Ground truth images saved.")
