#!/usr/bin/env python

from maskscoring_rcnn.tools.train_net import launch, main, default_argument_parser
from detectron2.data.datasets import register_coco_instances
import sys

register_coco_instances(
    "phone_train",
    {},
    "data/phone/annotations/train.json",
    "data/phone/train"
)
register_coco_instances(
    "phone_val",
    {},
    "data/phone/annotations/val.json",
    "data/phone/val"
)

sys.argv[1:1] = [
    "--num-gpus", "4",
    "--config-file", "phone.yaml",
    "--resume"
]

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
