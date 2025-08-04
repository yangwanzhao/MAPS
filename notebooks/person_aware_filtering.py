import os
import numpy as np
from notebooks.video_data import video_clips
from evaluate_run import read_bboxes
from PIL import Image

def is_empty_bbox(bbox):
    return np.allclose(bbox, [0, 0, 0, 0])

def check_false_positive(run_name, root_path='./'):
    indices = np.arange(0, 900, 15)
    fp_frames = 0
    fn_frames = 0
    tp_frames = 0
    tn_frames = 0
    total_frames = 0
    for video in sorted(os.listdir(root_path)):
        if video not in video_clips:
            continue
        vid_dir = os.path.join(root_path, video)
        if not os.path.isdir(vid_dir):
            continue
        gt_txt = os.path.join(vid_dir, 'bboxes.txt')
        pr_txt = os.path.join(vid_dir, f'bboxes_{run_name}.txt')
        if not (os.path.exists(gt_txt) and os.path.exists(pr_txt)):
            continue
        gt_boxes = read_bboxes(gt_txt)
        pr_boxes = read_bboxes(pr_txt)
        if len(pr_boxes) < 900:
            continue
        gt_boxes_eval = np.array(gt_boxes)[indices]
        pr_boxes_eval = np.array(pr_boxes)[indices]
        for gt_bbox, pr_bbox in zip(gt_boxes_eval, pr_boxes_eval):
            total_frames += 1
            if not is_empty_bbox(pr_bbox):
                if is_empty_bbox(gt_bbox):
                    fp_frames += 1
                else:
                    tp_frames += 1
            else:
                if is_empty_bbox(gt_bbox):
                    tn_frames += 1
                else:
                    fn_frames += 1
                
    print(f'[FP: {fp_frames} | TP: {tp_frames} | FN: {fn_frames} | TN: {tn_frames} | Total: {total_frames}]')
    return fp_frames, tp_frames, fn_frames, tn_frames, total_frames

def mask_overlap(ppe_mask_path, person_mask_path):
    ppe_mask = np.array(Image.open(ppe_mask_path).convert('1'), dtype=np.uint8)
    person_mask = np.array(Image.open(person_mask_path).convert('1'), dtype=np.uint8)
    intersection = np.logical_and(ppe_mask, person_mask).sum()
    ppe_area = ppe_mask.sum()
    if ppe_area == 0:
        return 0.0
    return intersection / ppe_area

def filter_bboxes_with_person_mask(run_name, root_path, overlap_thr):
    indices = np.arange(0, 900)
    skipped_videos = 0
    all_videos = [i for i in os.listdir(root_path) if i in video_clips]
    for vid, video in enumerate(sorted(all_videos)):
        print(f'Processing videos ... [{vid+1}/{len(all_videos)}]', end='\r')
        vid_dir = os.path.join(root_path, video)
        pr_txt = os.path.join(vid_dir, f'bboxes_{run_name}.txt')
        pr_boxes = read_bboxes(pr_txt)
        assert len(pr_boxes) == 900

        person_mask_dir = os.path.join(vid_dir, 'frames_Person')
        out_txt = os.path.join(vid_dir, f'bboxes_{run_name}_P_Person.txt')
        ########################################
        # frames_Person: Check for exactly 900 frames named 00000.jpg to 00899.jpg
        valid_person_mask = False
        if os.path.isdir(person_mask_dir):
            mask_files = [f for f in os.listdir(person_mask_dir) if f.endswith('.png')]
            if len(mask_files) == 900:
                valid_person_mask = True
        if not valid_person_mask:
            with open(out_txt, 'w') as f:
                for bbox in pr_boxes:
                    f.write(' '.join(map(str, [int(i) for i in bbox])) + '\n')
            skipped_videos += 1
            continue
        ########################################
        # frames_{run_name}_P_LCC: Check for exactly 900/15 frames (should be after postprocessing.py)
        # frames_{run_name}: This is for SOTA folders where 900 frames exist
        valid_lcc_mask = False
        lcc_mask_dir = os.path.join(vid_dir, f'frames_{run_name}')
        if os.path.isdir(lcc_mask_dir):
            mask_files = [f for f in os.listdir(lcc_mask_dir) if f.endswith('.png')]
            if run_name.endswith('P_LCC') and len(mask_files) == 900//15:
                valid_lcc_mask = True
            elif run_name in ['SAMURAI', 'DAM4SAM', 'SAM2Long', 'SAM2'] and len(mask_files) == 900:
                valid_lcc_mask = True
            else:
                raise Exception('Wrong')
        assert valid_lcc_mask, 'Run postprocess.py first to get processed masks!'
        ########################################

        # Per-video cache
        import pickle
        cache_path = os.path.join(vid_dir, f'overlap_cache_{run_name}.pkl')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                overlap_cache = pickle.load(f)
        else:
            overlap_cache = {}
        cache_updated = False

        filtered_bboxes = []
        for idx in indices:
            if idx % 15 == 0:
                frame_name = f'{idx:05d}.png'
                ppe_mask_path = os.path.join(vid_dir, f'frames_{run_name}', frame_name)
                person_mask_path = os.path.join(vid_dir, 'frames_Person', frame_name)
                assert os.path.exists(ppe_mask_path) , ppe_mask_path
                assert os.path.exists(person_mask_path), person_mask_path

                cache_key = (ppe_mask_path, person_mask_path)
                if cache_key in overlap_cache:
                    overlap = overlap_cache[cache_key]
                else:
                    overlap = mask_overlap(ppe_mask_path, person_mask_path)
                    overlap_cache[cache_key] = overlap
                    cache_updated = True
                    
                if overlap <= overlap_thr:
                    filtered_bboxes.append([0,0,0,0])
                else:
                    filtered_bboxes.append(pr_boxes[idx])
            else:
                filtered_bboxes.append([0,0,0,0])

        with open(out_txt, 'w') as f:
            for bbox in filtered_bboxes:
                f.write(' '.join(map(str, [int(i) for i in bbox])) + '\n')

        # Save cache if updated (per video)
        if cache_updated:
            with open(cache_path, 'wb') as f:
                pickle.dump(overlap_cache, f)

    print('Filtering complete. New bbox files written.')
    print(f'Skipped {skipped_videos} videos due to missing or incomplete person masks.')

if __name__ == '__main__':
    # must be after P_LCC
    run_name = ['MAPS_F_InvalidHu_final_tiny_P_LCC']

    root_path = './' 
    thr = 0.000
    
    for name in run_name:
        print(f'#### {name} ####')
        check_false_positive(name, root_path)
        filter_bboxes_with_person_mask(name, root_path, overlap_thr=thr)
        check_false_positive(f'{name}_P_Person', root_path)