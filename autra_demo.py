import fire
import torch
import torchvision.transforms as TS
import os
import sys
import cv2 as cv
import numpy as np

from autra.data_manager import DataManager
from autra.vocabulary import VOCABULARY
from PIL import Image
from collections import defaultdict

import detectron2
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from open_vocab_seg import add_ovseg_config
from open_vocab_seg.utils import VisualizationDemo, SAMVisualizationDemo


def setup_cfg(config_file):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg


def run(root, generator="Segment_Anything", sam_path="./sam_vit_l_0b3195.pth", granularity=0.7,  output_dir="outputs"):
    config_file = './ovseg_swinB_vitL_demo.yaml'
    cfg = setup_cfg(config_file)
    if generator == 'MaskFormer':
        demo = VisualizationDemo(cfg)
    elif generator == 'Segment_Anything':
        demo = SAMVisualizationDemo(cfg, granularity, sam_path, './ovseg_clip_l_9a1909.pth')
    elif generator == 'Segment_Anything_HQ':
        demo = SAMVisualizationDemo(cfg, granularity, sam_path, './ovseg_clip_l_9a1909.pth', use_hq=True)
    class_names = VOCABULARY
    dm = DataManager(root)
    os.makedirs(output_dir, exist_ok=True)
    iter = 0
    for sample in dm():
        # load image
        for camera_name in sample.keys():
            if "camera" not in camera_name:
                continue
            camera_info = sample[camera_name]
            img = read_image(camera_info["file"], format="BGR")
            _, visualized_output = demo.run_on_image(img, class_names)

            image_pil = Image.fromarray(np.uint8(visualized_output.get_image())).convert('RGB')
            import matplotlib.pyplot as plt
            # draw output image
            plt.figure(figsize=(10, 10))
            plt.imshow(np.asarray(image_pil))
            plt.axis('off')
            plt.savefig(
                os.path.join(output_dir, f"{camera_name}_{iter}.jpg"),
                bbox_inches="tight", dpi=300, pad_inches=0.0
            )
        iter += 1
        break
        if iter == 100: break


if __name__ == '__main__':
    fire.Fire(run)