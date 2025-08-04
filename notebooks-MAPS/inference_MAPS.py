import os
import cv2
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'notebooks')))
import torch
import gc
import yaml
import numpy as np
import shutil
import matplotlib.pyplot as plt
from utils import video_bucket
from sam2.build_sam import build_sam2_video_predictor
import warnings
from utils_data import video_data
warnings.filterwarnings('ignore')

def pick_prompts_from_bbox_file(bbox_txt_path, num_prompts):
    """
    Read all non-zero bboxes from `bbox_txt_path` (one line per frame: 'xmin ymin xmax ymax'),
    then select `num_prompts` frames uniformly spaced across [1 .. total_frames-2], snapping each
    target to the nearest annotated frame. Returns a sorted list of (frame_idx, [xmin, ymin, xmax, ymax]).
    """
    # 1) load every annotated prompt
    prompts = []
    with open(bbox_txt_path, 'r') as f:
        for idx, line in enumerate(f):
            coords = list(map(int, line.strip().split()))
            if coords != [0, 0, 0, 0]:
                prompts.append((idx, coords))
    assert len(prompts) > 0
    assert prompts[0][0] == 0
    assert sum(prompts[0][1]) > 0, 'First frame must contain prompt.'

    selected = [prompts[0]]
    if num_prompts == 1:
        return selected
    
    # 2) compute uniform targets (exclude frame 0 and last frame)
    num_prompts -= 1
    L = 900
    targets = [
        (i + 1) * (L // (num_prompts + 1))
        for i in range(num_prompts)
    ]

    # 3) snap each target to nearest available annotated frame
    for t in targets:
        nearest = min(prompts, key=lambda pc: abs(pc[0] - t))
        selected.append(nearest)

    # 4) dedupe & sort by frame index
    seen = set()
    unique = []
    for idx, coords in selected:
        if idx not in seen:
            unique.append((idx, coords))
            seen.add(idx)

    return sorted(unique, key=lambda x: x[0])


def predicted_mask_bbox(out_ids, out_logits):
    mask_to_vis = {}
    bbox_to_vis = {}
    for obj_id, mask_logit in zip(out_ids, out_logits):
        mask = mask_logit[0].cpu().numpy() > 0.0
        mask_to_vis[obj_id] = mask
        nz = np.argwhere(mask)
        if nz.size == 0:
            bbox_to_vis[obj_id] = [0, 0, 0, 0]
        else:
            y0, x0 = nz.min(axis=0)
            y1, x1 = nz.max(axis=0)
            bbox_to_vis[obj_id] = [x0, y0, x1, y1]
    return mask_to_vis, bbox_to_vis

def write_text_to_frame(height, img, text):
    position = (10, height - 10)  # Bottom-left corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)  # White text
    thickness = 2
    cv2.putText(img, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)
    return img

def write_box_to_frame(bbox, img):
    cmap = plt.get_cmap("tab10")
    color = np.array([*cmap(0)[:3]]) * 255
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    return img

def filter_largest_component(mask):
    raw_uint8 = (mask[1].astype(np.uint8) * 255)
    # find components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_uint8, connectivity=8)
    if num_labels > 2:
        # stats[:, cv2.CC_STAT_AREA] gives area for each label
        # skip background at index 0
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1 + areas.argmax()
        filtered_mask = (labels == largest)
    else:
        filtered_mask = mask[1]

    return filtered_mask


if __name__ == '__main__':
    # Hard-coded configurations
    DATA_ROOT = '/home/wan/sam2/notebooks/annotations_small_ppe_bbox'
    MODEL_CFG = 'configs/sam2.1/sam2.1_hiera_b+.yaml'
    CHECKPOINT = '../checkpoints/sam2.1_hiera_base_plus.pt'
    DEVICE = torch.device('cuda')
    FPS = 30
    name = 'MAPS_F_InvalidHu_final_k3'
    # Number of bbox prompts to use as initial seeds
    output_video = f'check_{name}.mp4'
    output_txt = f'bboxes_{name}.txt'
    output_mask = f'frames_{name}'

    # Load model and predictor
    with open('../sam2/' + MODEL_CFG, 'r') as f:
        config = yaml.safe_load(f)
    assert config['model']['is_MAPS'], 'Expected MAPS enabled'
    predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT, device=DEVICE)

    # Iterate over each instance directory 
    all_videos = [i for i in os.listdir(DATA_ROOT)]
    for id, inst_id in enumerate(all_videos):
        inst_dir = os.path.join(DATA_ROOT, inst_id)
        if not os.path.isdir(inst_dir):
            continue

        frames_dir = os.path.join(inst_dir, 'frames')
        mask_dir = os.path.join(inst_dir, output_mask)
        if os.path.exists(mask_dir):
            files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(files) == 900:
                print(f'Inference MAPS on {inst_id} [{id+1}/{len(all_videos)}] [SKIPPED]')
                continue
            else:
                shutil.rmtree(mask_dir)
                os.makedirs(mask_dir)
                print(f'Inference MAPS on {inst_id} [{id+1}/{len(all_videos)}]')
        else:
            os.makedirs(mask_dir)
            print(f'Inference MAPS on {inst_id} [{id+1}/{len(all_videos)}]')

        # Paths
        bbox_src = os.path.join(inst_dir, 'bboxes.txt')
        bbox_sam2 = os.path.join(inst_dir, output_txt)
        video_out = os.path.join(inst_dir, output_video)

        picked_prompts = pick_prompts_from_bbox_file(bbox_src, 1)
        assert picked_prompts[0][0] == 0
        # Load frames
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')],
                             key=lambda x: int(os.path.splitext(x)[0]))
        total_frames = len(frame_files)

        # Initialize SAM2 state
        state = predictor.init_state(video_path=frames_dir)

        # Multiple prompts
        for frame_idx, (x0, y0, x1, y1) in picked_prompts:
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=1,
                box=np.array([x0, y0, x1, y1], dtype=np.float32)
            )

        # Prepare video writer and bbox output
        sample_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
        h, w = sample_frame.shape[:2]
        writer = video_bucket(h, w, video_out, frame_rate=FPS)
        fb = open(bbox_sam2, 'w')

        # Propagate through video
        for out_idx, ids, logits, _ in predictor.propagate_in_video(state, run_name=(name, inst_id)):
            mask_vis, bbox_vis = predicted_mask_bbox(ids, logits)

            # mask_vis = filter_largest_component(mask_vis)

            bx = bbox_vis.get(1, [0, 0, 0, 0])
            fb.write(f"{bx[0]} {bx[1]} {bx[2]} {bx[3]}\n")

            # Save mask image
            mask_img = (mask_vis[1].astype('uint8') * 255)
            cv2.imwrite(os.path.join(mask_dir, f"{out_idx:05d}.png"), mask_img)

            # Overlay and write video frame
            frame_img = cv2.imread(os.path.join(frames_dir, frame_files[out_idx]))
            frame_img = write_box_to_frame(bx, frame_img)
            frame_img = write_text_to_frame(h, frame_img, f'Frame {out_idx}')
            writer.write(frame_img)

        fb.close()
        writer.release()

        # Cleanup
        del state
        torch.cuda.empty_cache()
        gc.collect()

    print(f'MAPS inference complete. [{name}]')