# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import multiprocessing as mp

import numpy as np
from PIL import Image


try:
    import detectron2
except:
    import os
    os.system('pip install git+https://github.com/facebookresearch/detectron2.git')

from detectron2.config import get_cfg

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image

import gradio as gr

def setup_cfg(config_file):
    from open_vocab_seg import add_ovseg_config
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg


def inference(class_names, proposal_gen, granularity, input_img):
    mp.set_start_method("spawn", force=True)

    if proposal_gen == 'MaskFormer':
        from open_vocab_seg.utils import VisualizationDemo
        config_file = './ovseg_swinB_vitL_demo.yaml'
        cfg = setup_cfg(config_file)
        demo = VisualizationDemo(cfg)
    elif proposal_gen == 'Segment_Anything':
        from open_vocab_seg.utils import SAMVisualizationDemo
        config_file = './ovseg_swinB_vitL_demo.yaml'
        cfg = setup_cfg(config_file)
        demo = SAMVisualizationDemo(cfg, granularity, "./sam_vit_l_0b3195.pth", './ovseg_clip_l_9a1909.pth')
    elif proposal_gen == 'Segment_Anything_HQ':
        from open_vocab_seg.utils import SAMVisualizationDemo
        config_file = './ovseg_swinB_vitL_demo.yaml'
        cfg = setup_cfg(config_file)
        demo = SAMVisualizationDemo(cfg, granularity, "./sam_hq_vit_l.pth", './ovseg_clip_l_9a1909.pth', use_hq=True)
    elif proposal_gen == 'SEEM':
        # seem
        import sys
        sys.path.append('seem')
        from utils import SEEMVisualizationDemo
        demo = SEEMVisualizationDemo()
    elif proposal_gen == 'Mask2Former':
        # seem
        import sys
        sys.path.append('seem')
        from utils import Mask2FormerVisualizationDemo
        from mask2former import add_maskformer2_config
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file("./maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
        demo = Mask2FormerVisualizationDemo(cfg)
    elif proposal_gen == "seemMask2Former":
        # seem
        import sys
        sys.path.append('seem')
        from utils import SEEMMask2FormerVisualizationDemo
        from mask2former import add_maskformer2_config
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file("./maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
        demo = SEEMMask2FormerVisualizationDemo(cfg)
    elif proposal_gen == 'YOLOV8':
        import sys
        sys.path.append('seem')
        from utils import YOLOVisualizationDemo
        demo = YOLOVisualizationDemo()

    class_names = list(filter(lambda x: len(x) > 0, class_names.split(',')))
    img = read_image(input_img, format="BGR")
    _, visualized_output = demo.run_on_image(img, class_names)

    return Image.fromarray(np.uint8(visualized_output.get_image())).convert('RGB')


examples = [['Saturn V, toys, desk, wall, sunflowers, white roses, chrysanthemums, carnations, green dianthus', 'Segment_Anything', 0.8, './resources/demo_samples/sample_01.jpeg'],
            ['red bench, yellow bench, blue bench, brown bench, green bench, blue chair, yellow chair, green chair, brown chair, yellow square painting, barrel, buddha statue', 'Segment_Anything', 0.8, './resources/demo_samples/sample_04.png'],
            ['pillow, pipe, sweater, shirt, jeans jacket, shoes, cabinet, handbag, photo frame', 'Segment_Anything', 0.7, './resources/demo_samples/sample_05.png'],
            ['Saturn V, toys, blossom', 'MaskFormer', 1.0, './resources/demo_samples/sample_01.jpeg'],
            ['Oculus, Ukulele', 'MaskFormer', 1.0, './resources/demo_samples/sample_03.jpeg'],
            ['Golden gate, yacht', 'MaskFormer', 1.0, './resources/demo_samples/sample_02.jpeg'],]
output_labels = ['segmentation map']

title = 'OVSeg (+ Segment_Anything)'

description = """
[NEW!] We incorperate OVSeg CLIP w/ Segment_Anything, enabling SAM's text prompts.
Gradio Demo for Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP. \n
OVSeg could perform open vocabulary segmentation, you may input more classes (seperate by comma). You may click on of the examples or upload your own image. \n
It might take some time to process. Cheers!
<p>(Colab only supports MaskFormer proposal generator) Don't want to wait in queue? <a href="https://colab.research.google.com/drive/1O4Ain5uFZNcQYUmDTG92DpEGCatga8K5?usp=sharing"><img data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667"></a></p>
"""

article = """
<p style='text-align: center'>
<a href='https://arxiv.org/abs/2210.04150' target='_blank'>
Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP
</a>
|
<a href='https://github.com/facebookresearch/ov-seg' target='_blank'>Github Repo</a></p>
"""

models = ["SEEM", "Mask2Former", 'seemMask2Former', 'YOLOV8']
models2 = [ "Segment_Anything", "Segment_Anything_HQ", "MaskFormer"]

def run(is_seem: bool):
    if is_seem:
        gr.Interface(
            inference,
            inputs=[
                gr.Textbox(
                    lines=1, placeholder=None, label='class names'),
                gr.Radio(models, label="Proposal generator"),
                gr.Slider(0, 1.0, 0.1,
                          label="For Segment_Anything only, granularity of masks from 0 (most coarse) to 1 (most precise)"),
                gr.Image(type='filepath'),
            ],
            outputs=gr.components.Image(type="pil", label='segmentation map'),
            title="Segment COCO").launch(enable_queue=True, server_name="0.0.0.0")
    else:
        gr.Interface(
            inference,
            inputs=[
                gr.Textbox(
                    lines=1, placeholder=None, label='class names'),
                gr.Radio(models2, label="Proposal generator"),
                gr.Slider(0, 1.0, 0.1, label="For Segment_Anything only, granularity of masks from 0 (most coarse) to 1 (most precise)"),
                gr.Image(type='filepath'),
            ],
            outputs=gr.components.Image(type="pil", label='segmentation map'),
            title=title,
            description=description,
            article=article,
            examples=examples).launch(enable_queue=True, server_name="0.0.0.0")

import fire
if __name__ == '__main__':
    fire.Fire(run)