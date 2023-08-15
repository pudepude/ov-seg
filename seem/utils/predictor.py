# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import numpy as np
import torch
from torch.nn import functional as F
import cv2

from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer, VisImage
from detectron2.modeling.postprocessing import sem_seg_postprocess

#seem
import sys
import os
sys.path.append('seem')
from xdecoder.BaseModel import BaseModel
from xdecoder import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from tasks.interactive import transform, register_classes
from PIL import Image


class SEEMVisualizationDemo(object):
    def __init__(self, cfg="seem/configs/seem/seem_focall_lang.yaml",
                 pretrained_pth="seem/seem_focall_v1.pt"):
        if not os.path.exists(pretrained_pth):
            os.system("wget -O {} {}".format(pretrained_pth, "https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v1.pt"))
        opt = load_opt_from_config_files(cfg)
        opt = init_distributed(opt)
        self.model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    def run_on_image(self, ori_image, class_names=[], debug=True):
        self.model.model.metadata = register_classes(class_names)
        with torch.no_grad():
            self.model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                self.model.model.metadata.thing_classes, is_eval=True)
            self.model.model.task_switch['spatial'] = False
            self.model.model.task_switch['visual'] = False
            self.model.model.task_switch['grounding'] = False
            self.model.model.task_switch['audio'] = False
        # image = {"image": Image.fromarray(ori_image), "mask": None}
        # image_ori = transform(image['image'])
        # new_width = image_ori.size[0]
        # new_height = image_ori.size[1]
        # image_ori = np.asarray(image_ori)
        # visual = Visualizer(image_ori, metadata=self.model.model.metadata)
        # images = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).cuda()

        height, width, _ = ori_image.shape
        if width < height:
            new_width = 512
            new_height = int((new_width / width) * height)
        else:
            new_height = 512
            new_width = int((new_height / height) * width)
        image = cv2.resize(ori_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        images = torch.from_numpy(image.copy()).permute(2, 0, 1).cuda()
        data = {"image": images, "height": new_height, "width": new_width}
        batch_inputs = [data]

        results = self.model.model.evaluate(batch_inputs)
        pano_seg = results[-1]['panoptic_seg'][0]
        pano_seg_info = results[-1]['panoptic_seg'][1]
        vis_output = None
        if debug:
            visual = Visualizer(image, metadata=self.model.model.metadata)
            vis_output = visual.draw_panoptic_seg(pano_seg.cpu(), pano_seg_info)  # rgb Image
        pano_seg_origin = F.interpolate(pano_seg.view(1, 1, new_height, new_width).float(), size=(height, width), mode="nearest").squeeze().int()
        predictions = {"sem_seg": pano_seg_origin, "sem_seg_info": pano_seg_info,
                       'class_names': self.model.model.metadata.thing_classes,
                       'colors': self.model.model.metadata.thing_colors}

        return predictions, vis_output

class Mask2FormerVisualizationDemo(object):
    def __init__(self, cfg):
        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, ori_image, class_names=[], debug=True):
        metadata = register_classes([])
        height, width, _ = ori_image.shape
        if width < height:
            new_width = 512
            new_height = int((new_width / width) * height)
        else:
            new_height = 512
            new_width = int((new_height / height) * width)
        image = cv2.resize(ori_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        outputs = self.predictor(image)
        pano_seg = outputs["panoptic_seg"][0]
        pano_seg_info = outputs["panoptic_seg"][1]
        vis_output = None
        if debug:
            visual = Visualizer(image, metadata=metadata)
            vis_output = visual.draw_panoptic_seg(pano_seg.cpu(), pano_seg_info)  # rgb Image
        pano_seg_origin = F.interpolate(pano_seg.view(1, 1, new_height, new_width).float(), size=(height, width),
                                        mode="nearest").squeeze().int()
        predictions = {"sem_seg": pano_seg_origin, "sem_seg_info": pano_seg_info,
                       'class_names': metadata.thing_classes,
                       'colors': metadata.thing_colors}

        return predictions, vis_output


class SEEMMask2FormerVisualizationDemo(object):
    def __init__(self, cfg,
                 seemcfg="seem/configs/seem/seem_focall_lang.yaml",
                 pretrained_pth="seem/seem_focall_v1.pt"):
        if not os.path.exists(pretrained_pth):
            os.system("wget -O {} {}".format(pretrained_pth, "https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v1.pt"))
        opt = load_opt_from_config_files(seemcfg)
        opt = init_distributed(opt)
        self.model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, ori_image, class_names=[], debug=True):
        metadata = register_classes([])
        height, width, _ = ori_image.shape
        if width < height:
            new_width = 512
            new_height = int((new_width / width) * height)
        else:
            new_height = 512
            new_width = int((new_height / height) * width)
        image = cv2.resize(ori_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        outputs = self.predictor(image)
        pano_seg1 = outputs["panoptic_seg"][0]
        pano_seg_info1 = outputs["panoptic_seg"][1]
        clses = metadata.thing_classes
        cate = [0] * (len(clses) + 1)
        for info in pano_seg_info1:
            cate[info['id']] = info['category_id'] + 1
        sem_seg_cate1 = torch.tensor(cate, dtype=torch.long, device=pano_seg1.device)[pano_seg1.view(-1).long()]

        self.model.model.metadata = metadata
        with torch.no_grad():
            self.model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                self.model.model.metadata.thing_classes, is_eval=True)
            self.model.model.task_switch['spatial'] = False
            self.model.model.task_switch['visual'] = False
            self.model.model.task_switch['grounding'] = False
            self.model.model.task_switch['audio'] = False
        images = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).cuda()
        data = {"image": images, "height": new_height, "width": new_width}
        batch_inputs = [data]
        results = self.model.model.evaluate(batch_inputs)
        pano_seg2 = results[-1]['panoptic_seg'][0]
        pano_seg_info2 = results[-1]['panoptic_seg'][1]

        cate = [0] * (len(clses) + 1)
        for info in pano_seg_info2:
            cate[info['id']] = info['category_id'] + 1
        sem_seg_cate2 = torch.tensor(cate, dtype=torch.long, device=pano_seg2.device)[pano_seg2.view(-1).long()]
        min_sem = torch.minimum(sem_seg_cate1, sem_seg_cate2)
        max_sem = torch.maximum(sem_seg_cate1, sem_seg_cate2)
        background = min_sem == 0
        min_sem[background] = max_sem[background]
        min_sem = min_sem.reshape(pano_seg1.shape)
        labels = torch.unique(min_sem, return_counts=False).cpu().numpy()
        seg_infos = []
        for label in labels:
            if label > 0:
                seg_infos.append(dict(id=label, category_id=label - 1, isthing=False))

        vis_output = None
        if debug:
            visual = Visualizer(image, metadata=metadata)
            vis_output = visual.draw_panoptic_seg(min_sem.cpu(), seg_infos)  # rgb Image
        pano_seg_origin = F.interpolate(min_sem.view(1, 1, new_height, new_width).float(), size=(height, width),
                                        mode="nearest").squeeze().int()
        predictions = {"sem_seg": pano_seg_origin, "sem_seg_info": seg_infos,
                       'class_names': metadata.thing_classes,
                       'colors': metadata.thing_colors}
        #print(predictions)
        return predictions, vis_output

from ultralytics import YOLO
class YOLOVisualizationDemo(object):
    def __init__(self):
        # Load a model
        self.model = YOLO('yolov8x-seg.pt')  # load an official model

    def run_on_image(self, ori_image, class_names=[], debug=True):
        results = self.model(ori_image)  # predict on an image
        print(len(results))
        print("class_names", results[0].names)
        metadata = register_classes(results[0].names)
        masks = results[0].masks
        print(masks.data.shape)
        return None, VisImage(results[0].plot())
