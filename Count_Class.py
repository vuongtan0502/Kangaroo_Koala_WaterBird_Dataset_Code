import json
from collections import defaultdict

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def count_objects_per_class(coco_data):
    # Map category_id to category name
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Count instances per category_id
    class_counts = defaultdict(int)
    for ann in coco_data['annotations']:
        class_counts[ann['category_id']] += 1
    
    # Convert to readable format: {class_name: count}
    readable_counts = {cat_id_to_name[k]: v for k, v in class_counts.items()}
    return readable_counts

def print_counts(title, counts):
    print(f"\n{title}")
    print("-" * len(title))
    for class_name, count in sorted(counts.items()):
        print(f"{class_name}: {count}")

# === Main ===
train_json_path = 'D:/PR_Curve/Koala_train_org.json'
valid_json_path = 'D:/PR_Curve/Koala_valid_org.json'

# Load both datasets
train_data = load_json(train_json_path)
valid_data = load_json(valid_json_path)

# Count per class
train_counts = count_objects_per_class(train_data)
valid_counts = count_objects_per_class(valid_data)

# Print results
print_counts("Train Set Class Counts", train_counts)
print_counts("Validation Set Class Counts", valid_counts)

