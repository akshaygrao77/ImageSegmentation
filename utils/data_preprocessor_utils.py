import os
from pycocotools.coco import COCO
import json

from config.config import get_transform
from structures.dataset_structure import COCOSegmentationDataset

def get_colormapping(coco_annotations_json,meta_json):
    coco = COCO(coco_annotations_json)
    categories = coco.loadCats(coco.getCatIds())
    name_id_map=dict()
    for cat in categories:
        name_id_map[cat["name"]] = cat["id"]
    with open(meta_json, 'r') as f:
        meta_data = json.load(f)
    id_to_colormap = dict()
    for obj in meta_data["classes"]:
        id_to_colormap[name_id_map[obj["title"]]] = obj["color"]
    
    return id_to_colormap

def get_dataset(img_dir, ann_dir, is_train=False):
    transforms_image,transforms_mask = get_transform(is_train)
    if is_train:
        append_str = "train"
    else:
        append_str = "val"
    
    dataset = COCOSegmentationDataset(
        root=os.path.join(img_dir, append_str),
        ann_file=os.path.join(ann_dir, append_str + ".json"),
        transforms_image=transforms_image,
        transforms_mask=transforms_mask
    )
    return dataset
