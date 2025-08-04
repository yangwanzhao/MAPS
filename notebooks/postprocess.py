import os
import cv2
import numpy as np
from notebooks.video_data import video_clips


def get_largest_connected_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        # Only background, no mask
        return np.zeros_like(mask), None
    elif num_labels == 2:
        # Only one connected mask (foreground + background)
        # The mask itself is the largest component (label 1)
        lcc_mask = (labels == 1).astype(np.uint8) * 255
        x, y, w, h, area = stats[1]
        return lcc_mask, (x, y, w, h)
    else:
        # Ignore background (label 0), find largest component
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        lcc_mask = (labels == largest_label).astype(np.uint8) * 255
        x, y, w, h, area = stats[largest_label]
        return lcc_mask, (x, y, w, h)


def process_video_output(video_name, run_name, father_path, total_frames=900, frame_interval=15):
    frames_folder = os.path.join(father_path, video_name, f"frames_{run_name}")
    out_frames_folder = os.path.join(father_path, video_name, f"frames_{run_name}_P_LCC")
    out_bboxes_file = os.path.join(father_path, video_name, f"bboxes_{run_name}_P_LCC.txt")

    os.makedirs(out_frames_folder, exist_ok=True)

    # Read all mask filenames and sort by frame index
    mask_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.png') or f.endswith('.jpg')])
    mask_files_dict = {int(os.path.splitext(f)[0]): f for f in mask_files}

    bboxes_out = []
    for idx in range(total_frames):
        if idx % frame_interval == 0 and idx in mask_files_dict:
            mask_path = os.path.join(frames_folder, mask_files_dict[idx])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                bboxes_out.append('0 0 0 0')
                continue
            lcc_mask, bbox = get_largest_connected_component(mask)
            
            # You actually do not need new masks. => Now I need for person_aware filtering.
            out_mask_path = os.path.join(out_frames_folder, f'{idx:05d}.png') # only save at 1 FPS.
            cv2.imwrite(out_mask_path, lcc_mask)
            
            # You need bboxes.
            if bbox is not None:
                x, y, w, h = bbox
                xmin = x
                ymin = y
                xmax = x + w - 1
                ymax = y + h - 1
                bboxes_out.append(f'{xmin} {ymin} {xmax} {ymax}')
            else:
                bboxes_out.append('0 0 0 0')
        else:
            bboxes_out.append('0 0 0 0')
    
    assert len(bboxes_out) == total_frames
    with open(out_bboxes_file, 'w') as f:
        for line in bboxes_out:
            f.write(line + '\n')


def main():
    dataset_folder = 'video_clips'
    run_names = [
        'MAPS_F_InvalidHu_final_k1',
    ]
    video_list = [v for v in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, v)) and v in video_clips]
    for run_name in run_names:
        for ind, video_name in enumerate(video_list):
            print(f'Processing run {run_name} [{ind+1}/{len(video_list)}]', end='\r')
            process_video_output(video_name, run_name, dataset_folder)
        print('')

if __name__ == '__main__':
    main()
