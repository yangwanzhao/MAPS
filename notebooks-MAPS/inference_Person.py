import os
import cv2
import torch
import gc
import yaml
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'notebooks'))) 
import shutil
import matplotlib.pyplot as plt
from utils import video_bucket
from sam2.build_sam import build_sam2_video_predictor
from ultralytics import YOLO
from utils_data import video_data
import warnings
warnings.filterwarnings('ignore')

def run_yolo_world_on_frame(frame):
    # Initialize a YOLO-World model
    model = YOLO("yolov8x-world.pt")  # or choose yolov8m/l-world.pt
    model.set_classes(["person"])
    model.verbose = False
    results = model.predict(frame, verbose=False) # was a path to the image
    return results

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

def bbox_overlap(boxA, boxB, thresh=0.0):
    # box: [xmin, ymin, xmax, ymax]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return False
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    overlap_ratio = interArea / float(boxBArea)
    return overlap_ratio > thresh


if __name__ == '__main__':
    DATA_ROOT = '/home/wan/sam2/notebooks/annotations_small_ppe_bbox'
    MODEL_CFG = 'configs/sam2.1/sam2.1_hiera_l.yaml'
    CHECKPOINT = '../checkpoints/sam2.1_hiera_large.pt'
    DEVICE = torch.device('cuda')
    FPS = 30
    output_video = f'check_Person.mp4'
    output_txt = f'bboxes_Person.txt'
    output_mask = f'frames_Person'

    # Load model and predictor
    with open('../sam2/' + MODEL_CFG, 'r') as f:
        config = yaml.safe_load(f)
    assert not config['model']['is_MAPS'], 'Expected SAM2 enabled'
    predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT, device=DEVICE)

    all_videos = [i for i in os.listdir(DATA_ROOT)]
    match_frame_counts = {}
    none_count = 0
    for id, inst_id in enumerate(all_videos):
        inst_dir = os.path.join(DATA_ROOT, inst_id)
        if not os.path.isdir(inst_dir):
            continue

        frames_dir = os.path.join(inst_dir, 'frames')
        mask_dir = os.path.join(inst_dir, output_mask)
        if os.path.exists(mask_dir):
            files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(files) == 900:
                print(f'Inference PersonSAM2 on {inst_id} [{id+1}/{len(all_videos)}] [SKIPPED]')
                continue
            else:
                shutil.rmtree(mask_dir)
                os.makedirs(mask_dir)
                print(f'Inference PersonSAM2 on {inst_id} [{id+1}/{len(all_videos)}]')
        else:
            os.makedirs(mask_dir)
            print(f'Inference PersonSAM2 on {inst_id} [{id+1}/{len(all_videos)}]')

        bbox_src = os.path.join(inst_dir, 'bboxes.txt')
        bbox_sam2 = os.path.join(inst_dir, output_txt)
        video_out = os.path.join(inst_dir, output_video)

        # Load ground truth PPE bboxes
        gt_bboxes = []
        with open(bbox_src, 'r') as f:
            for line in f:
                coords = list(map(int, line.strip().split()))
                gt_bboxes.append(coords)

        # Load frames
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')],
                             key=lambda x: int(os.path.splitext(x)[0]))
        total_frames = len(frame_files)
        
        assert total_frames == len(gt_bboxes)
        # Find the first frame with a matching person bbox
        found = False
        match_frame = None
        for frame_idx, (ppe_bbox, frame_file) in enumerate(zip(gt_bboxes, frame_files)):
            if ppe_bbox == [0, 0, 0, 0]:
                continue
            frame_path = os.path.join(frames_dir, frame_file)
            results = run_yolo_world_on_frame(frame_path)
            
            # Extract person bboxes from YOLO results
            person_bboxes = []
            for r in results:
                for box in r.boxes.xyxy.cpu().numpy():
                    person_bboxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
            
            # Check for overlap
            match = None
            for person_bbox in person_bboxes:
                if bbox_overlap(person_bbox, ppe_bbox, thresh=0.6):
                    match = person_bbox
                    break
            if match is not None:
                found = True
                match_frame = frame_idx
                break
        if not found:
            print(f'No matching person bbox found for {inst_id} [SKIPPED]')
            none_count += 1
            continue
        else:
            match_frame_counts[match_frame] = match_frame_counts.get(match_frame, 0) + 1
            if match_frame != 0:
                print(f'Found match at frame {match_frame} [SKIPPED]')
                continue
            else:
                print(f'Found match at frame {match_frame}')
        
        # Initialize SAM2 state
        state = predictor.init_state(video_path=frames_dir)
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=match_frame,
            obj_id=1,
            box=np.array(match, dtype=np.float32)
        )

        # Prepare video writer and bbox output
        sample_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
        h, w = sample_frame.shape[:2]
        writer = video_bucket(h, w, video_out, frame_rate=FPS)
        fb = open(bbox_sam2, 'w')

        # Propagate through video
        for out_idx, ids, logits in predictor.propagate_in_video(state):
            mask_vis, bbox_vis = predicted_mask_bbox(ids, logits)
            bx = bbox_vis.get(1, [0, 0, 0, 0])
            fb.write(f"{bx[0]} {bx[1]} {bx[2]} {bx[3]}\n")

            # Save mask image
            mask_img = (mask_vis[1].astype('uint8') * 255)
            cv2.imwrite(os.path.join(mask_dir, f"{out_idx:05d}.png"), mask_img)

            # Overlay and write video frame
            frame_img = cv2.imread(os.path.join(frames_dir, frame_files[out_idx]))

            # Overlay predicted person mask (in red, with transparency)
            color_mask = np.zeros_like(frame_img)
            color_mask[mask_vis[1]] = [0, 0, 255]  # Red mask
            frame_img = cv2.addWeighted(frame_img, 1.0, color_mask, 0.4, 0)

            # Draw PPE bbox from gt_bboxes at 1 FPS (every 30th frame)
            if out_idx % 15 == 0 and out_idx < len(gt_bboxes):
                ppe_bbox = gt_bboxes[out_idx]
                if ppe_bbox != [0, 0, 0, 0]:
                    frame_img = write_box_to_frame(ppe_bbox, frame_img)

            # Draw predicted person bbox
            frame_img = write_box_to_frame(bx, frame_img)
            frame_img = write_text_to_frame(h, frame_img, f'Frame {out_idx}')
            writer.write(frame_img)

        fb.close()
        writer.release()

        # Cleanup
        del state
        torch.cuda.empty_cache()
        gc.collect()
    
    print('PersonSAM2 inference complete.')
    print('Match frame summary:')
    for frame_idx in sorted(match_frame_counts.keys()):
        print(f"Videos matched at frame {frame_idx}: {match_frame_counts[frame_idx]}")
    print(f"Videos with no match: {none_count}")
