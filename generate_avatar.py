import argparse
import copy
import math
import os
import sys
import time
import warnings

import cv2
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image

os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO
from torchalign import FacialLandmarkDetector
from utils import get_video_fps, decompose_tfm, img_warp, metrix_M, laplacianSmooth

# Suppress warnings and set device
warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} for inference.')

class Runner:
    def __init__(self, args, img_size=(256, 256)):
        self.device = device
        self.img_size = img_size
        self.pads = args.pads
        self.avatar_name = args.avatar_name
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Load face detection model
        self.face_det = YOLO(os.path.join(args.pretrained_model_dir, '******')).to(self.device)

        # Load landmark detection model
        self.lmk_net = FacialLandmarkDetector(os.path.join(args.pretrained_model_dir, '******'))
        self.lmk_net = self.lmk_net.to(self.device).eval()

        # Determine input type and load frames
        self.face = args.face
        self.fps = args.fps if args.face.lower().endswith(('jpg', 'jpeg', 'png')) else get_video_fps(args.face)
        self.full_frames = self._load_frames()

        # Initialize smoothers
        self.kpts_smoother = laplacianSmooth()
        self.abox_smoother = laplacianSmooth()
        self.frame_info = {}

    def _load_frames(self):
        """Load frames from video or image."""
        if self.face.lower().endswith(('jpg', 'jpeg', 'png')):
            print(f'Reading image as face: {self.face}')
            return [cv2.imread(self.face)]
        else:
            print(f'Reading video as face: {self.face}')
            frames = []
            video_stream = cv2.VideoCapture(self.face)
            while True:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                frames.append(frame)
            return frames

    @staticmethod
    def landmark_to_keypoints(landmark):
        """Convert facial landmarks to keypoints."""
        lefteye = np.mean(landmark[60:68, :], axis=0)
        righteye = np.mean(landmark[68:76, :], axis=0)
        nose = landmark[54, :]
        leftmouth = np.mean([landmark[76, :], landmark[88, :]], axis=0)
        rightmouth = np.mean([landmark[82, :], landmark[92, :]], axis=0)
        return lefteye, righteye, nose, leftmouth, rightmouth

    @torch.no_grad()
    def detect_face(self, face_img):
        """Detect faces in an image."""
        boxes = self.face_det(face_img, imgsz=640, conf=0.01, iou=0.5, half=True, augment=False, device=self.device)[0].boxes
        return boxes.xyxy.cpu().numpy()

    @torch.no_grad()
    def detect_lmk(self, image, bbox=None):
        """Detect facial landmarks."""
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        bbox_tensor = torch.from_numpy(np.array(bbox)[:, :4])
        return self.lmk_net(img_pil, bbox=bbox_tensor, device=self.device).cpu().numpy()

    @torch.no_grad()
    def get_input_imginfo_by_index(self, idx):
        """Get image information for a specific frame index."""
        frame = self.full_frames[idx].copy()
        bbox = self.detect_face(frame)[0][:5]
        landmark = self.detect_lmk(frame, [bbox])[0]
        keypoints = self.kpts_smoother.smooth(self.landmark_to_keypoints(landmark))

        # Generate transformation matrix and align frame
        m = metrix_M(face_size=200, expand_size=256, keypoints=keypoints)
        align_frame = img_warp(frame, m, 256, adjust=0)
        align_bbox = self.abox_smoother.smooth(np.reshape(self.detect_face(align_frame)[0][:4], (-1, 2))).reshape(-1)

        # Rewarp image while maintaining scale
        w, h = 256, 256
        rt, s = decompose_tfm(m)
        align_frame = cv2.warpAffine(frame, rt, (math.ceil(w / s[0][0]), math.ceil(h / s[1][1])), flags=cv2.INTER_CUBIC)
        inv_m = cv2.invertAffineTransform(rt)

        # Crop and resize face
        h, w, _ = align_frame.shape
        bbox = align_bbox * np.array([(w - 1) / 255, (h - 1) / 255, (w - 1) / 255, (h - 1) / 255])
        rect = [round(f) for f in bbox[:4]]
        pady1, pady2, padx1, padx2 = self.pads
        face = align_frame[max(0, rect[1] - pady1):min(h, rect[3] + pady2), max(0, rect[0] - padx1):min(w, rect[2] + padx2)]
        face = cv2.resize(face, self.img_size)

        return {
            'img': face,
            'frame': frame,
            'coords': (max(0, rect[1] - pady1), min(h, rect[3] + pady2), max(0, rect[0] - padx1), min(w, rect[2] + padx2)),
            'align_frame': align_frame,
            'm': m,
            'inv_m': inv_m,
        }

    def run(self):
        """Run the processing pipeline."""
        frame_info_list = []
        for frame_id in tqdm(range(len(self.full_frames)), desc="Processing frames"):
            imginfo = self.get_input_imginfo_by_index(frame_id)
            frame_info_list.append(imginfo)

        frame_h, frame_w = self.full_frames[0].shape[:2]
        data = {
            'fps': self.fps,
            'frame_num': len(self.full_frames),
            'frame_h': frame_h,
            'frame_w': frame_w,
            'frame_info_list': frame_info_list
        }

        avatar_info_file = os.path.join(self.save_dir, f'{self.avatar_name}.pkl')
        with open(avatar_info_file, 'wb') as f:
            pickle.dump(data, f)

def main(args):
    runner = Runner(args)
    runner.run()

if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
    parser.add_argument('--face', type=str, help='Filepath of video/image that contains faces to use', default='******')
    parser.add_argument('--avatar_name', type=str, help='Avatar name', default='******')
    parser.add_argument('--save_dir', type=str, default='******')
    parser.add_argument('--fps', type=float, help='FPS, specified only if input is a static image (default: 25)', default=25., required=False)
    parser.add_argument('--pads', nargs='+', type=int, default=[0, 0, 0, 0], help='Padding (top, bottom, left, right). Please adjust to include chin at least')
    parser.add_argument('--pretrained_model_dir', type=str, default='******', help='Directory containing pretrained models')

    args = parser.parse_args()

    main(args)
    print(f'Total time: {time.time() - start_time:.2f} seconds')
