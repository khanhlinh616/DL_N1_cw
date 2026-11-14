"""Quick verification of the real COCO split files."""
import json

with open("data/coco/train_subset14k.json") as f:
    train = json.load(f)
with open("data/coco/val_heldout1k.json") as f:
    val = json.load(f)
with open("data/coco/annotations/instances_val2017.json") as f:
    val2017 = json.load(f)

train_ids = {img["id"] for img in train["images"]}
val_ids = {img["id"] for img in val["images"]}
val2017_ids = {img["id"] for img in val2017["images"]}

print(f"Train images: {len(train_ids)}")
print(f"Val-heldout images: {len(val_ids)}")
print(f"Val2017 images: {len(val2017_ids)}")
print(f"Overlap train & heldout: {len(train_ids & val_ids)}")
print(f"Train boxes: {len(train['annotations'])}")
print(f"Val-heldout boxes: {len(val['annotations'])}")
print(f"Categories in train: {len(train['categories'])}")

assert len(train_ids) == 14000
assert len(val_ids) == 1000
assert len(val2017_ids) == 5000
assert len(train_ids & val_ids) == 0
print("ALL REAL-DATA ASSERTIONS PASSED!")
