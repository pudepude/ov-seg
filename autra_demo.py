import fire
import torch
import torchvision.transforms as TS
import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from autra.data_manager import DataManager, CAMERA_PRIORITY_KEYS
from autra.vocabulary import VOCABULARY
from autra.matric_util import project
from autra.compensator import Compensator
from PIL import Image
from collections import defaultdict
from pypcd import pypcd

import detectron2
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
import warnings
import multiprocessing
warnings.filterwarnings("ignore", category=UserWarning)

def setup_cfg(config_file):
    from open_vocab_seg import add_ovseg_config
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg

WEATHER = set(['snow', 'water'])

LIDAR_CATEGORY_MAPPING = {
    'person': 2,
    'bicycle': 3,
    'car' : 1,
    'motorcycle': 3,
    'bus' : 1,
    'train': 1,
    'truck': 1,
    'traffic light': 7,
    'stop sign': 7,
    'bridge': 6,
    'railroad': 5,
    'road': 5,
    'wall-brick': 6,
    'wall-stone': 6,
    'wall-tile': 6,
    'wall-wood': 6,
    'tree': 8,
    'fence': 6,
    'floor': 5,
    'pavement': 5,
    'mountain': 8,
    'grass': 8,
    'building': 6,
    'wall': 6,
    'rug': 5,
    'ceiling': 7,
    'roof': 7,
    'banner': 7
}


def get_color(label, color):
    def cal(l):
        if l == -1: # ignore point dark
            return (40 << 16) | (40 << 8) | 40
        if l == 0: # ignore point dark
            return (40 << 16) | (40 << 8) | 40
        if l == 1: # car green
            return 255 << 8
        if l == 2: # person yellow
            return (255 << 16) | (255 << 8)
        if l == 3: # bicycle blue
            return 255
        if l == 4: # red corn
            return (255 << 16)
        if l == 5: # road white
            return  (255 <<16) | (255 << 8) | 255
        if l == 6: # sink purple fence building
            return (191 << 16) | 255
        if l == 7: # orange banner
            return (255 <<16) | (128 << 8)
        if l == 8: # dark green tree mountain grass
            return (80 << 8)
        assert False, f"unknown label {l}"

    for i in range(10):
        color[label == (i - 1)] = cal(i - 1)
    return color

def get_demo(generator,
            sam_path,
            granularity):
    if 'MaskFormer' == generator:
        from open_vocab_seg.utils import VisualizationDemo
        config_file = './ovseg_swinB_vitL_demo.yaml'
        cfg = setup_cfg(config_file)
        demo = VisualizationDemo(cfg)
    elif 'Segment_Anything_HQ' == generator:
        from open_vocab_seg.utils import SAMVisualizationDemo
        config_file = './ovseg_swinB_vitL_demo.yaml'
        cfg = setup_cfg(config_file)
        demo = SAMVisualizationDemo(cfg, granularity, sam_path, './ovseg_clip_l_9a1909.pth', use_hq=True)
    elif 'Segment_Anything' == generator:
        from open_vocab_seg.utils import SAMVisualizationDemo
        config_file = './ovseg_swinB_vitL_demo.yaml'
        cfg = setup_cfg(config_file)
        demo = SAMVisualizationDemo(cfg, granularity, sam_path, './ovseg_clip_l_9a1909.pth')
    elif 'SEEM' == generator:
        # seem
        import sys
        sys.path.append('seem')
        from utils import SEEMVisualizationDemo
        demo = SEEMVisualizationDemo()
        class_names = []
    elif 'Mask2Former' == generator:
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
        class_names = []
    elif generator == "seemMask2Former":
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
        class_names = []
    else:
        assert False, "Unsupportex generator " + generator
    return demo


def subtask(demos,
            sample,
            cur_output_dir,
            output_raw_img,
            output_project_img,
            task_id
            ):
    print(task_id)
    thread_id = int(multiprocessing.current_process().name.split('-')[-1]) - 1
    demo = demos[thread_id]
    class_names = VOCABULARY
    lidar_info = sample['lidar']
    lidar_file = lidar_info["file"]
    save_filename = os.path.basename(lidar_file)[:-len(".pcd")]
    lidar_points = pypcd.PointCloud.from_path(lidar_file)
    imupoints = np.concatenate(
        [lidar_points.pc_data['x'], lidar_points.pc_data['y'], lidar_points.pc_data['z'],
         lidar_points.pc_data['intensity']
         ]).reshape(4, -1).T.astype(np.float32)
    ref_pose = lidar_info['pose_quat']
    lidar_cls = lidar_points.pc_data['label'].astype(np.int32)  # {-1, 0, 1, 2, 3, 4, 500, 500+}
    imu_conf = lidar_points.pc_data['confidence']  # for detection result >0
    camera_cls = np.zeros_like(lidar_cls)
    # load image and calculate camera_cls
    for camera_name in CAMERA_PRIORITY_KEYS:
        camera_info = sample[camera_name]
        img = read_image(camera_info["file"], format="BGR")
        if output_raw_img:
            # draw output image
            plt.figure(figsize=(10, 10))
            plt.imshow(img[:, :, ::-1])
            plt.axis('off')
            plt.savefig(
                os.path.join(cur_output_dir, f"{camera_name}_{save_filename}_raw.jpg"),
                bbox_inches="tight", dpi=300, pad_inches=0.0
            )
            plt.close()

        predictions, visualized_output = demo.run_on_image(img, class_names, debug=output_project_img)
        sem_seg = predictions['sem_seg']
        sem_seg_info = predictions['sem_seg_info']
        clses = predictions['class_names']
        colors = predictions['colors']
        cate = [0] * (len(clses) + 1)
        for info in sem_seg_info:
            cls = clses[info['category_id']]
            lidar_id = LIDAR_CATEGORY_MAPPING.get(cls, -1)
            cate[info['id']] = lidar_id
        cate[0] = 0
        sem_seg_cate = torch.tensor(cate, dtype=torch.long, device=sem_seg.device)[sem_seg.view(-1).long()]

        # deal with lidar project
        all_points = Compensator.compensate_by_pose(imupoints[:, :3], camera_info['pose_quat'], ref_pose)
        x0, y0, x1, y1 = camera_info['roi']
        width = sem_seg.shape[1]
        height = sem_seg.shape[0]
        camera_xyz, project_ps = project(all_points, camera_info['intrinsics'],
                                         camera_info['rotate_vectors'], camera_info['translation'])
        project_ps = project_ps.round().astype(np.int32)
        valid_mask = (camera_xyz[:, 2] > 0) & (project_ps[:, 0] >= x0) & (project_ps[:, 0] < x1) & \
                     (project_ps[:, 1] >= y0) & (project_ps[:, 1] < y1)

        project_ps = project_ps[valid_mask]
        x = project_ps[:, 0]
        y = project_ps[:, 1]
        pos = y * width + x
        camera_cls[valid_mask] = sem_seg_cate[pos].cpu().numpy()

        if output_project_img:
            color_img = np.uint8(visualized_output.get_image())[:, :, ::-1]
            color_img = cv.resize(color_img, (width, height), interpolation=cv.INTER_CUBIC)
            # for u, v in zip(x, y):
            #     cv.circle(color_img, (u, v), 3, (255, 255, 255))
            plt.figure(figsize=(10, 10))
            plt.imshow(color_img)
            plt.axis('off')
            plt.savefig(
                f"{cur_output_dir}/camera/{camera_name}/{save_filename}.jpg",
                bbox_inches="tight", dpi=300, pad_inches=0.0
            )
            plt.close()

    # fusion lidar cls and camera_cls
    # 1. lidar result with (conf > 0.6 or lidar same with camera) keep no change
    idx_keep = (imu_conf > 0.6) | (camera_cls == lidar_cls)
    # 2. lidar result with conf < 0.6 but >0 or ground keep no change when camera_cls is ground 5 or background <=0
    camera_ground_idx = camera_cls == 5
    camera_background_idx = camera_cls <= 0
    lidar_ground_idx = lidar_cls == 5
    idx_keep = (idx_keep) | ((imu_conf > 0.01) & (camera_background_idx | camera_ground_idx))

    # ignore other ground points
    idx_ignore = (~idx_keep) & (lidar_ground_idx | camera_cls <= 5)

    # other set to camera cls
    # idx_set_camera = (~idx_keep) & (~idx_ignore)

    camera_cls[idx_keep] = lidar_cls[idx_keep]
    camera_cls[idx_ignore] = 0
    # points  with same lidar label>=500 should be same cls
    # cluster_idx = (lidar_cls >= 500).nonzero()[0]
    # cluster_dict = defaultdict(lambda: defaultdict(int))
    # for idx in cluster_idx:
    #     label = camera_cls[idx].item()
    #     cnt_dict = cluster_dict[lidar_cls[idx].item()]
    #     cnt_dict[label] += 1
    # for cluster, cnt_dict in cluster_dict.items():
    #     best_label = sorted(cnt_dict.items(), key=lambda x: -x[1])[0][0]
    #     assert best_label >= -1 and best_label <= 8, f"best_label={best_label} is not in range [-1, 8]"
    #     camera_cls[(lidar_cls == cluster)] = best_label
    lidar_points.pc_data['label'] = camera_cls.astype(np.uint32)
    get_color(camera_cls, lidar_points.pc_data['rgb'])
    lidar_points.save_pcd(f"{cur_output_dir}/lidar/{save_filename}.pcd", compression='binary')
    os.system(f"cp {lidar_info['label']} {cur_output_dir}/label")


def throw_error(e):
    raise e


def run(root, lidar_result_path, thread_cnt=2,
        generator="seemMask2Former",
        sam_path="./sam_vit_l_0b3195.pth",
        granularity=0.1,
        output_dir="outputs",
        output_raw_img=False,
        output_project_img=True):
    demos = [get_demo(generator, sam_path, granularity) for _ in range(thread_cnt)]
    pool = multiprocessing.get_context('spawn').Pool(processes=thread_cnt)

    dm = DataManager(root, lidar_result_path)
    os.makedirs(output_dir, exist_ok=True)
    iter = 0
    for sample in dm():
        assert "lidar" in sample.keys()
        record_name = sample['record_name']
        cur_output_dir = os.sep.join([output_dir, record_name])
        os.makedirs(cur_output_dir, exist_ok=True)
        os.makedirs(cur_output_dir + "/lidar", exist_ok=True)
        os.makedirs(cur_output_dir + "/camera", exist_ok=True)
        os.makedirs(cur_output_dir + "/label", exist_ok=True)
        for camera_name in CAMERA_PRIORITY_KEYS:
            os.makedirs(cur_output_dir + "/camera/" + camera_name, exist_ok=True)
        pool.apply_async(subtask, (demos, sample, cur_output_dir, output_raw_img, output_project_img, iter),
                         error_callback=throw_error)
        iter += 1
    pool.close()
    pool.join()


if __name__ == '__main__':
    fire.Fire(run)