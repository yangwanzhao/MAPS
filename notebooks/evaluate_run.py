import numpy as np
import os
import warnings
from notebooks.video_data import video_clips
warnings.filterwarnings('ignore')

def is_valid_bbox(bbox):
    return not (bbox[0] == 0 and bbox[1] == 0 and bbox[2] == 0 and bbox[3] == 0)
            
def _to_array(bboxes):
    """Helper: (N,4) array from list of length‐4 lists/tuples."""
    return np.asarray(bboxes, dtype=float).reshape(-1, 4)

def is_empty_bbox(bbox):
    return np.allclose(bbox, [0, 0, 0, 0])

def compute_ious(gt_bboxes, pred_bboxes):
    """
    Returns: np.array of shape (N,) giving IoU for each frame.
    IoU = 1 if both GT and pred are empty, 0 if only one is empty, else standard IoU.
    """
    
    gt = _to_array(gt_bboxes)
    pr = _to_array(pred_bboxes)
    # Masks for empty boxes
    gt_empty = np.all(np.isclose(gt, 0, atol=1e-8), axis=1)
    pr_empty = np.all(np.isclose(pr, 0, atol=1e-8), axis=1)
    both_empty = gt_empty & pr_empty
    one_empty = gt_empty ^ pr_empty

    # Standard IoU for non-empty pairs
    valid = ~(both_empty | one_empty)
    ious = np.zeros(len(gt))
    if np.any(valid):
        xA = np.maximum(gt[valid,0], pr[valid,0])
        yA = np.maximum(gt[valid,1], pr[valid,1])
        xB = np.minimum(gt[valid,2], pr[valid,2])
        yB = np.minimum(gt[valid,3], pr[valid,3])
        interW = np.clip(xB - xA, a_min=0, a_max=None)
        interH = np.clip(yB - yA, a_min=0, a_max=None)
        inter  = interW * interH
        area_gt = (gt[valid,2] - gt[valid,0]) * (gt[valid,3] - gt[valid,1])
        area_pr = (pr[valid,2] - pr[valid,0]) * (pr[valid,3] - pr[valid,1])
        union = area_gt + area_pr - inter
        ious[valid] = np.where(union > 0, inter / union, 0.0)
    ious[both_empty] = 1.0
    ious[one_empty] = 0.0
    return ious

def compute_center_errors(gt_bboxes, pred_bboxes):
    """
    Returns: np.array of shape (N,) giving Euclidean distance between bbox centers.
    Ignores frames where GT is empty or pred is empty (returns np.nan for those frames).
    """
    gt = _to_array(gt_bboxes)
    pr = _to_array(pred_bboxes)
    gt_empty = np.all(np.isclose(gt, 0, atol=1e-8), axis=1)
    pr_empty = np.all(np.isclose(pr, 0, atol=1e-8), axis=1)
    valid = ~(gt_empty | pr_empty)
    errs = np.full(len(gt), np.nan)
    if np.any(valid):
        gt_centers = np.column_stack(((gt[valid,0]+gt[valid,2])/2, (gt[valid,1]+gt[valid,3])/2))
        pr_centers = np.column_stack(((pr[valid,0]+pr[valid,2])/2, (pr[valid,1]+pr[valid,3])/2))
        errs[valid] = np.linalg.norm(gt_centers - pr_centers, axis=1)
    return errs

# ──────────────── VOS‐style metrics ───────────────────── #

def AUC(gt_bboxes, pred_bboxes, num_thresh=101):
    """
    Area‐Under‐Curve of the success‐plot over IoU thresholds [0..1].
    """
    ious = compute_ious(gt_bboxes, pred_bboxes)
    ths  = np.linspace(0.0, 1.0, num_thresh)
    success = [(ious >= t).mean() for t in ths]
    return np.trapezoid(success, ths)

def P(gt_bboxes, pred_bboxes, thresh=20.0):
    """
    Precision: fraction of frames with center‐error <= `thresh` pixels @ pixel‐threshold.

    d_i = center‐error_i
    P = (1/N) * |{ i : d_i ≤ thresh }|, ignoring frames with nan error.
    """
    errs = compute_center_errors(gt_bboxes, pred_bboxes)
    valid = ~np.isnan(errs)
    if not valid.any():
        return 0.0
    return float((errs[valid] <= thresh).mean())

def Pnorm(gt_bboxes, pred_bboxes, num_thresh=101):
    """
    Normalized‐Precision AUC: 
    z = center‐error / diagonal_length_of_GT_bbox;
    area under the success‐plot z in [0..1].
    Ignores frames with nan error or zero diagonal.
    """
    gt = _to_array(gt_bboxes)
    errs = compute_center_errors(gt_bboxes, pred_bboxes)
    diag = np.linalg.norm(gt[:,2:] - gt[:,:2], axis=1)
    norm_err = np.where(diag>0, errs/diag, np.nan)
    valid = ~np.isnan(norm_err)
    ths = np.linspace(0.0, 1.0, num_thresh)
    success = [(norm_err[valid] <= t).mean() if valid.any() else 0.0 for t in ths]
    return np.trapezoid(success, ths)

# ──────────────── GOT‐10k‐style metrics ───────────────── #

def AO(gt_bboxes, pred_bboxes):
    """
    Average Overlap: mean IoU over all frames.
    """
    return float(compute_ious(gt_bboxes, pred_bboxes).mean())

def OP(gt_bboxes, pred_bboxes, thresh=0.5):
    """
    Overlap Precision @ thresh: fraction of frames with IoU >= thresh.
    """
    return float((compute_ious(gt_bboxes, pred_bboxes) >= thresh).mean())

def OP50(gt_bboxes, pred_bboxes):
    """Shorthand for OP@0.5"""
    return OP(gt_bboxes, pred_bboxes, thresh=0.5)

def OP75(gt_bboxes, pred_bboxes):
    """Shorthand for OP@0.75"""
    return OP(gt_bboxes, pred_bboxes, thresh=0.75)

def read_bboxes(path):
    boxes = []
    for line in open(path, 'r'):
        parts = line.split()
        if len(parts) == 4:
            boxes.append(list(map(float, parts)))

    # assert len(boxes) == 900 
    # This is used when SAM2 is running.
    if len(boxes) < 900:
        return []

    return boxes

def evaluate_run(run_name, root_path='./', verbose=False, viewmode=True):
    res = []
    if verbose:
        print("ID\tAUC\tP@20\tP_norm\tAO\tOP@0.5\tOP@0.75\tVideo")
    for vid, video in enumerate(sorted(os.listdir(root_path))):
        
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
        indices = np.arange(0, 900, 15)
        gt_boxes_eval = np.array(gt_boxes)[indices]
        pr_boxes_eval = np.array(pr_boxes)[indices]
        auc = AUC(gt_boxes_eval, pr_boxes_eval)
        p20 = P(gt_boxes_eval, pr_boxes_eval, thresh=20.0)
        pnorm = Pnorm(gt_boxes_eval, pr_boxes_eval)
        ao = AO(gt_boxes_eval, pr_boxes_eval)
        op50 = OP50(gt_boxes_eval, pr_boxes_eval)
        op75 = OP75(gt_boxes_eval, pr_boxes_eval)
        res.append((auc, p20, pnorm, ao, op50, op75, video))
        if verbose:
            print(f"{vid}\t{auc:.4f}\t{p20:.4f}\t{pnorm:.4f}\t{ao:.4f}\t{op50:.4f}\t{op75:.4f}\t{video}")
    # Overall
    auc_all = float(np.mean([r[0] for r in res]))
    p20_all = float(np.mean([r[1] for r in res]))
    pnorm_all = float(np.mean([r[2] for r in res]))
    ao_all = float(np.mean([r[3] for r in res]))
    op50_all = float(np.mean([r[4] for r in res]))
    op75_all = float(np.mean([r[5] for r in res]))
    overall = {
        'AUC': auc_all,
        'P@20': p20_all,
        'P_norm': pnorm_all,
        'AO': ao_all,
        'OP@0.5': op50_all,
        'OP@0.75': op75_all,
        '#Video': len([r[0] for r in res])
    }
    # Gloves
    auc_gloves = float(np.mean([r[0] for r in res if '_c5_' in r[6]]))
    p20_gloves = float(np.mean([r[1] for r in res if '_c5_' in r[6]]))
    pnorm_gloves = float(np.mean([r[2] for r in res if '_c5_' in r[6]]))
    ao_gloves = float(np.mean([r[3] for r in res if '_c5_' in r[6]]))
    op50_gloves = float(np.mean([r[4] for r in res if '_c5_' in r[6]]))
    op75_gloves = float(np.mean([r[5] for r in res if '_c5_' in r[6]]))
    gloves = {
        'AUC': auc_gloves,
        'P@20': p20_gloves,
        'P_norm': pnorm_gloves,
        'AO': ao_gloves,
        'OP@0.5': op50_gloves,
        'OP@0.75': op75_gloves,
        '#Video': len([r[0] for r in res if '_c5_' in r[6]])
    }
    # Masks
    auc_masks = float(np.mean([r[0] for r in res if '_c6_' in r[6] or '_c9_' in r[6]]))
    p20_masks = float(np.mean([r[1] for r in res if '_c6_' in r[6] or '_c9_' in r[6]]))
    pnorm_masks = float(np.mean([r[2] for r in res if '_c6_' in r[6] or '_c9_' in r[6]]))
    ao_masks = float(np.mean([r[3] for r in res if '_c6_' in r[6] or '_c9_' in r[6]]))
    op50_masks = float(np.mean([r[4] for r in res if '_c6_' in r[6] or '_c9_' in r[6]]))
    op75_masks = float(np.mean([r[5] for r in res if '_c6_' in r[6] or '_c9_' in r[6]]))
    masks = {
        'AUC': auc_masks,
        'P@20': p20_masks,
        'P_norm': pnorm_masks,
        'AO': ao_masks,
        'OP@0.5': op50_masks,
        'OP@0.75': op75_masks,
        '#Video': len([r[0] for r in res if '_c6_' in r[6] or '_c9_' in r[6]])
    }
    
    if viewmode:
        print('*'*25 + f'  {run_name}  ' + '*'*25)
        print(" \tAUC\tP@20\tP_norm\tAO\tOP@0.5\tOP@0.75\t#Video")
        print(f'Overall\t{auc_all:.4f}\t{p20_all:.4f}\t{pnorm_all:.4f}\t{ao_all:.4f}\t{op50_all:.4f}\t{op75_all:.4f}\t{overall["#Video"]}')
        print(f"Gloves\t{auc_gloves:.4f}\t{p20_gloves:.4f}\t{pnorm_gloves:.4f}\t{ao_gloves:.4f}\t{op50_gloves:.4f}\t{op75_gloves:.4f}\t{gloves['#Video']}")
        print(f"Masks\t{auc_masks:.4f}\t{p20_masks:.4f}\t{pnorm_masks:.4f}\t{ao_masks:.4f}\t{op50_masks:.4f}\t{op75_masks:.4f}\t{masks['#Video']}")
        print('')
    # Per-video AUCs for gloves and masks
    per_video_gloves = [(r[6], r[0]) for r in res if '_c5_' in r[6]]
    per_video_masks = [(r[6], r[0]) for r in res if '_c6_' in r[6] or '_c9_' in r[6]]
    return {
        'overall': overall,
        'gloves': gloves,
        'masks': masks,
        'per_video': {
            'gloves': per_video_gloves,
            'masks': per_video_masks
        }
    }

if __name__ == '__main__':
    root_path = './video_clips'
    
    # final
    name_list = ['MAPS_F_InvalidHu_final']
    
    for name in name_list:
        evaluate_run(name, root_path, viewmode=True)