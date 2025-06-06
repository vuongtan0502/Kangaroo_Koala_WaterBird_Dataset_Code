import json
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURATION ===
train_json_path = 'D:/PR_Curve/Kangaroo_train.json'
valid_json_path = 'D:/PR_Curve/Kangaroo_valid.json'

# Define bins (e.g., 0%-0.02%, 0.02%-0.04%, ..., 0.26%-100%)
bins = np.arange(0.0000, 0.0028, 0.0002).tolist() + [1.0]  # 0.0000 to 0.0026 in 0.0002 steps, then 1.0
bin_labels = [f"{b*100:.4f}%-{bins[i+1]*100:.4f}%" for i, b in enumerate(bins[:-1])]

# === HELPER FUNCTIONS ===
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def merge_coco_jsons(json1, json2):
    images = json1['images'] + json2['images']
    annotations = json1['annotations'] + json2['annotations']
    categories = json1['categories']
    return {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

def calculate_area_distribution(coco_data, bins, bin_labels):
    area_counts = {label: 0 for label in bin_labels}
    image_dims = {img['id']: (img['width'], img['height']) for img in coco_data['images']}
    
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_dims:
            continue
        img_w, img_h = image_dims[img_id]
        img_area = img_w * img_h

        bbox = ann['bbox']  # [x, y, width, height]
        box_area = bbox[2] * bbox[3]
        rel_area = box_area / img_area

        for i in range(len(bins) - 1):
            if bins[i] <= rel_area < bins[i + 1]:
                area_counts[bin_labels[i]] += 1
                break
    return area_counts

# === LOAD, COMBINE, AND COMPUTE ===
train_data = load_json(train_json_path)
valid_data = load_json(valid_json_path)
combined_data = merge_coco_jsons(train_data, valid_data)

area_counts = calculate_area_distribution(combined_data, bins, bin_labels)

# === PRINT RESULTS ===
print("Object Count by Relative Area Ranges (in % of image):")
for k, v in area_counts.items():
    print(f"{k}: {v} objects")

# === OPTIONAL: PLOT ===
plt.figure(figsize=(12, 5))
plt.bar(area_counts.keys(), area_counts.values())
plt.xticks(rotation=45)
plt.ylabel('Number of Objects')
plt.title('Object Count by Relative Area (% of image size)')
plt.tight_layout()
plt.show()

