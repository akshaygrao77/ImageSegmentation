import json
import copy

def merge_coco_datasets(dataset1_path, dataset2_path, output_path, category_mapping, original_group_key="original_group"):
    """
    Merges two COCO-format datasets, applying category remapping from dataset2 to dataset1 categories.

    Parameters:
        dataset1_path (str): Path to the base (reference) dataset. Its categories remain unchanged.
        dataset2_path (str): Path to the second dataset. Its categories will be mapped and merged.
        output_path (str): Path to write the final merged dataset.
        category_mapping (dict): Mapping from dataset2 category names to dataset1 category names.
        original_group_key (str): Custom field name to track origin. Default is 'original_group'.
    """
    with open(dataset1_path) as f1, open(dataset2_path) as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    # Build name → ID lookup for dataset1 categories
    name_to_id_data1 = {cat["name"]: cat["id"] for cat in data1["categories"]}
    name_to_id_data2 = {cat["id"]: cat["name"] for cat in data2["categories"]}

    # Track max IDs to prevent collision
    max_img_id = max(img["id"] for img in data1["images"])
    max_ann_id = max(ann["id"] for ann in data1["annotations"])

    # Remap image IDs and annotate original_group
    image_id_mapping = {}
    for img in data1["images"]:
        img[original_group_key] = 0

    for img in data2["images"]:
        old_id = img["id"]
        max_img_id += 1
        img["id"] = max_img_id
        img[original_group_key] = 1
        image_id_mapping[old_id] = img["id"]

    # Remap annotations and categories from dataset2
    new_annotations = []
    for ann in data1["annotations"]:
        ann[original_group_key] = 0

    for ann in data2["annotations"]:
        old_cat_name = name_to_id_data2[ann["category_id"]]
        if old_cat_name not in category_mapping:
            raise ValueError(f"Category '{old_cat_name}' in dataset2 is not mapped to dataset1 categories.")
        
        new_cat_name = category_mapping[old_cat_name]
        ann["category_id"] = name_to_id_data1[new_cat_name]
        ann["image_id"] = image_id_mapping[ann["image_id"]]
        max_ann_id += 1
        ann["id"] = max_ann_id
        ann[original_group_key] = 1
        new_annotations.append(ann)

    # Merge all
    merged = {
        "images": data1["images"] + data2["images"],
        "annotations": data1["annotations"] + new_annotations,
        "categories": data1["categories"]
    }

    # Write to output
    with open(output_path, "w") as f_out:
        json.dump(merged, f_out, indent=2)

    print(f"✅ Merged dataset saved to: {output_path}")

if __name__=='__main__':
    category_mapping = {
    "tire flat": "Broken part",
    "lamp broken": "Broken part",
    "glass shatter": "Broken part",
    "crack": "Cracked",
    "scratch": "Scratch",
    "dent": "Dent"
    }
    merge_coco_datasets(
        dataset1_path="/home/akshay/dgx-code/ImageSegmentation/data/car-parts-and-car-damages/Car_damages_dataset/split_annotations/train.json",
        dataset2_path="/home/akshay/dgx-code/ImageSegmentation/data/CarDD_release/CarDD_COCO/annotations/instances_train2017.json",
        output_path="/home/akshay/dgx-code/ImageSegmentation/data/car_damage_merged/merged_annotations_train.json",
        category_mapping=category_mapping
    )