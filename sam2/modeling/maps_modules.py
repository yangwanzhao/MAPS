import numpy as np
import math
import cv2

class AdaptiveHuFilterEMA:
    def __init__(self, 
                 log_path='/home/wan/sam2/notebooks-MAPS/hu_log.txt', 
                 huOrder=0, k=3, alpha=0.1, beta=0.1, use_option=2, total_frames=900):
        """EMA-based Hu moment filter with option switch"""
        self.alpha = alpha  # EMA smoothing factor
        self.beta = beta    # First-frame influence factor
        self.use_option = use_option  # 1 for output correction, 2 for anchored update
        self.k = k

        self.initialized = False
        self.target_size = (128, 128)
        self.huOrder = huOrder
        self.total_frames = total_frames
        self.cur_frame = 0
        self.first_hu = None

        self.log_file = log_path
        with open(self.log_file, 'w') as f:
            f.write("frameID\talpha\tbeta\tk\tmean\tstd\tcur_hu\tis_valid\n")

    def initialize(self, first_mask):
        if self.initialized:
            raise RuntimeError("Filter already initialized")

        if isinstance(first_mask, (float, int, np.floating, np.integer)):
            first_hu = float(first_mask)
        else:
            first_hu = self.compute_hu(first_mask)

        self.mean = first_hu
        self.var = 1e-4
        self.first_hu = first_hu
        self.cur_frame += 1
        self.initialized = True

    def evaluate(self, new_mask, frame_id=None):
        if not self.initialized:
            raise RuntimeError("Filter not initialized with first frame")
        if frame_id == None:
            frame_id = self.cur_frame

        self.alpha = self._update_alpha(frame_id)
        self.beta = self._update_beta(frame_id)
        self.k = self._update_k(frame_id)
        # self.k = 4

        if isinstance(new_mask, (float, int, np.floating, np.integer)):
            cur_hu = float(new_mask)
        else:
            cur_hu = self.compute_hu(new_mask)

        filtered_mean = self.get_mean()
        
        is_valid = abs(cur_hu - filtered_mean) <= self.k * math.sqrt(self.var)
        if frame_id < 90: 
            is_valid = True

        self.log_status(frame_id, filtered_mean, cur_hu, is_valid, self.alpha, self.beta, self.k)
        self.cur_frame += 1

        return is_valid

    def update(self, new_mask):
        if isinstance(new_mask, (float, int, np.floating, np.integer)):
            cur_hu = float(new_mask)
        else:
            cur_hu = self.compute_hu(new_mask)

        if self.use_option == 1:
            self.standard_update(cur_hu)
        else:
            self.anchored_update(cur_hu)

    def _update_alpha(self, frame_id, alpha_start=0.2, alpha_end=0.8, delta=4.0):
        # Exponentially increase alpha over time.
        r_t = frame_id / self.total_frames
        return alpha_start + (alpha_end - alpha_start) * (1 - math.exp(-delta * r_t))

    def _update_beta(self, frame_id, beta_start=1.0, beta_end=0.1, gamma=6.0):
        # Exponentially decrease beta over time.
        r_t = frame_id / self.total_frames
        return beta_end + (beta_start - beta_end) * math.exp(-gamma * r_t)

    def _update_k(self, frame_id, k_start=6, k_end=3, lamb=10.0):
        # Use a sigmoid function to smoothly decrease k.
        r_t = frame_id / self.total_frames
        return k_end + (k_start - k_end) / (1 + math.exp(lamb * (r_t - 0.5)))

    def output_corrected_mean(self, current_mean):
        return (1 - self.beta) * current_mean + self.beta * self.first_hu

    def get_mean(self):
        if self.use_option == 1:
            filtered_mean = self.output_corrected_mean(self.mean)
        else:
            filtered_mean = self.mean
        return filtered_mean
    
    def standard_update(self, cur_hu):
        prev_mean = self.mean
        self.mean = self.alpha * cur_hu + (1 - self.alpha) * self.mean
        self.var = self.alpha * (cur_hu - prev_mean)**2 + (1 - self.alpha) * self.var

    def anchored_update(self, cur_hu):
        prev_mean = self.mean
        standard_mean = self.alpha * cur_hu + (1 - self.alpha) * self.mean
        self.mean = self.beta * self.first_hu + (1 - self.beta) * standard_mean
        self.var = self.alpha * (cur_hu - prev_mean)**2 + (1 - self.alpha) * self.var

    def crop_and_resize(self, mask):
        mask_cpu = mask[0][0].cpu().numpy()
        ys, xs = np.where(mask_cpu > 0)
        if len(xs) == 0 or len(ys) == 0:
            return np.zeros(self.target_size, dtype=np.uint8)
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        cropped_mask = mask_cpu[ymin:ymax+1, xmin:xmax+1]
        return cv2.resize(cropped_mask, self.target_size, interpolation=cv2.INTER_NEAREST)

    def calculate_all_hu_momen(self, mask):
        M = cv2.moments(mask.astype(np.uint8))
        hu = cv2.HuMoments(M).flatten()
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        return hu_log[self.huOrder]

    def compute_hu(self, mask):
        assert mask.size(0) == 1 and mask.size(1) == 1, f'Wrong mask size: {mask.size()}'
        mask_resized = self.crop_and_resize(mask)
        mask_hu = self.calculate_all_hu_momen(mask_resized)

        return mask_hu


    def log_status(self, frame_id, mean, cur_hu, is_valid, alpha, beta, k):
        std = math.sqrt(self.var)
        with open(self.log_file, 'a') as f:
            f.write(f"{frame_id}\t{alpha:.4f}\t{beta:.4f}\t{k:.4f}\t{mean:.4f}\t{std:.4f}\t{cur_hu:.4f}\t{is_valid}\n")
