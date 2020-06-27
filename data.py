#!/usr/bin/env python

import copy
import json
from pathlib import Path
import random
import shutil
import numpy as np
import cv2

data = Path("data")

raw = data / "raw"
raw_img = raw / "TC images"
raw_mark = raw / "imagedrawn_decrypt"
raw_ok = raw / "OK Orgin Images"

dst = data / "phone"
dst_train = dst / "train"
dst_val = dst / "val"
dst_annotations = dst / "annotations"

for i in (dst, dst_train, dst_val, dst_annotations):
    i.mkdir(exist_ok=True)

okfile = data / "oklist.txt"
oklist = okfile.read_text().split('\n')

val_ratio = 0.1


train = {
    "categories": [],
    "images": [],
    "annotations": [],
    "img_id": 1,
    "mark_id": 1
}

categories = ["Lines", "Polygons", "Rubbers"]
categories_id = dict()
for i in range(len(categories)):
    categories_id[categories[i]] = i + 1
    train["categories"].append({
        "id": i + 1,
        "name": categories[i]
    })

val = copy.deepcopy(train)


def polygon_area(x, y):
    pad = np.zeros(shape=(y.max() + 1, x.max() + 1))
    c = np.stack((x, y), axis=1)
    c = np.expand_dims(c, 1)
    cv2.drawContours(pad, c, -1, 1, 1)
    return pad.sum()


def append_mark(target_json, raw_mark, categories):
    x = []
    y = []
    for i in raw_mark["Points"]:
        a, b = tuple(map(int, i.split(",")))
        x.append(a)
        y.append(b)
    x = np.array(x)
    y = np.array(y)
    area = polygon_area(x, y)
    seg = np.stack([x, y], axis=1).flatten().tolist()
    if len(seg) < 6:
        return

    bbox_x = int(np.min(x))
    bbox_y = int(np.min(y))
    bbox_w = int(np.max(x) + 1 - bbox_x)
    bbox_h = int(np.max(y) + 1 - bbox_y)

    target_json["annotations"].append({
        "id": target_json["mark_id"],
        "image_id": target_json["img_id"],
        "bbox": [
            bbox_x,
            bbox_y,
            bbox_w,
            bbox_h
        ],
        "segmentation": [seg],
        "category_id": categories_id[categories],
        "area": area,
        "iscrowd": 0
    })
    target_json["mark_id"] += 1


for jsonfile in raw_mark.glob("*"):
    if random.random() <= val_ratio:
        target_json = val
        target_img_dir = dst_val
    else:
        target_json = train
        target_img_dir = dst_train

    img_name = jsonfile.stem + ".bmp"
    img = raw_img / img_name
    if not img.exists():
        img_name = jsonfile.stem + ".png"
        img = raw_img / img_name
    shutil.copy(img, target_img_dir / img_name)

    j = json.loads(jsonfile.read_text())
    target_json["images"].append({
        "file_name": img_name,
        "id": target_json["img_id"],
        "height": j["Height"],
        "width": j["Width"]
    })

    for index in categories:
        if j[index] is not None:
            for mark in j[index]:
                append_mark(target_json, mark, index)

    target_json["img_id"] += 1


for okimg_name in oklist:
    if random.random() <= val_ratio:
        target_json = val
        target_img_dir = dst_val
    else:
        target_json = train
        target_img_dir = dst_train

    shutil.copy(raw_img / okimg_name, target_img_dir / okimg_name)

    target_json["images"].append({
        "file_name": okimg_name,
        "id": target_json["img_id"],
        "height": 128,
        "width": 128
    })

    target_json["img_id"] += 1


cnt_train = train["img_id"] - 1
cnt_val = val["img_id"] - 1

for i in (train, val):
    del i["img_id"]
    del i["mark_id"]

with open(dst_annotations / "train.json", "w") as f:
    json.dump(train, f, indent=4)
with open(dst_annotations / "val.json", "w") as f:
    json.dump(val, f, indent=4)

print("total train: %i, val: %i" % (cnt_train, cnt_val))
