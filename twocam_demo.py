import argparse

import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width
from modules.center_mass import compute_com


class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def run_demo(net, image_provider1, image_provider2, height_size, cpu, track, smooth, com):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts # +1 for Hidden COM
    previous_poses = []
    # original delay 33
    # 0 = pause / wait for input indefinetly
    delay = 33
    total_provider = zip(image_provider1, image_provider2)
    for img1, img2 in total_provider:
        orig_img1 = img1.copy()
        orig_img2 = img2.copy()
        heatmaps1, pafs1, scale1, pad1 = infer_fast(net, img1, height_size, stride, upsample_ratio, cpu)
        heatmaps2, pafs2, scale2, pad2 = infer_fast(net, img2, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num1 = 0
        total_keypoints_num2 = 0
        all_keypoints_by_type1 = []
        all_keypoints_by_type2 = []
        
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num1 += extract_keypoints(heatmaps1[:, :, kpt_idx], all_keypoints_by_type1, total_keypoints_num1)
            total_keypoints_num2 += extract_keypoints(heatmaps2[:, :, kpt_idx], all_keypoints_by_type2, total_keypoints_num2)

        pose_entries1, all_keypoints1 = group_keypoints(all_keypoints_by_type1, pafs1, demo=True)
        pose_entries2, all_keypoints2 = group_keypoints(all_keypoints_by_type2, pafs2, demo=True)
        for kpt_id in range(all_keypoints1.shape[0]):
            all_keypoints1[kpt_id, 0] = (all_keypoints1[kpt_id, 0] * stride / upsample_ratio - pad1[1]) / scale1
            all_keypoints1[kpt_id, 1] = (all_keypoints1[kpt_id, 1] * stride / upsample_ratio - pad1[0]) / scale1
        for kpt_id in range(all_keypoints2.shape[0]):
            all_keypoints2[kpt_id, 0] = (all_keypoints2[kpt_id, 0] * stride / upsample_ratio - pad2[1]) / scale2
            all_keypoints2[kpt_id, 1] = (all_keypoints2[kpt_id, 1] * stride / upsample_ratio - pad2[0]) / scale2
        current_poses1 = []
        current_poses2 = []
        for n in range(len(pose_entries1)):
            if len(pose_entries1[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints+1, 2), dtype=np.int32) * -1 # +1 here for COM
            found_kpts = []
            C_pts = []
            BOS = [[-1, -1], [-1, -1]]
            for kpt_id in range(num_keypoints):
                if pose_entries1[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints1[int(pose_entries1[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints1[int(pose_entries1[n][kpt_id]), 1])
                    found_kpts.append(kpt_id)
            if com:
                COM, C_pts, BOS = compute_com(found_kpts, pose_keypoints)
                pose_keypoints[-1] = COM
            pose = Pose(pose_keypoints, pose_entries1[n][18], C_pts, BOS)
            current_poses1.append(pose)
        
        for n in range(len(pose_entries2)):
            if len(pose_entries2[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints+1, 2), dtype=np.int32) * -1 # +1 here for COM
            found_kpts = []
            C_pts = []
            BOS = [[-1, -1], [-1, -1]]
            for kpt_id in range(num_keypoints):
                if pose_entries2[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints2[int(pose_entries2[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints2[int(pose_entries2[n][kpt_id]), 1])
                    found_kpts.append(kpt_id)
            if com:
                COM, C_pts, BOS = compute_com(found_kpts, pose_keypoints)
                pose_keypoints[-1] = COM
            pose = Pose(pose_keypoints, pose_entries2[n][18], C_pts, BOS)
            current_poses2.append(pose)
        
        #if track:
            #track_poses(previous_poses, current_poses, smooth=smooth)
            #previous_poses = current_poses
        for pose in current_poses1:
            pose.draw(img1)
        for pose in current_poses2:
            pose.draw(img2)
	
        img1 = cv2.addWeighted(orig_img1, 0.6, img1, 0.4, 0)
        img2 = cv2.addWeighted(orig_img2, 0.6, img2, 0.4, 0)
        #for pose in current_poses:
            #cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          #(pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            #if track:
                #cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            #cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        cv2.imshow('Demo, Feed 1', img1)
        cv2.imshow('Demo, Feed 2', img2)
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 33:
                delay = 0
            else:
                delay = 33


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--vid1', type=str, default='', help='path to video file or camera id - first feed')
    parser.add_argument('--vid2', type=str, default='', help='path to video file or camera id - second feed')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    parser.add_argument('--com', action='store_true', help='find center of mass if possible')
    args = parser.parse_args()

    if args.vid1 == '' or args.vid2 == '':
        raise ValueError('Both --vid1 and --vid2 have to be provided')

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    #frame_provider = ImageReader(args.images)
    frame_provider1 = VideoReader(args.vid1)
    frame_provider2 = VideoReader(args.vid2)
    run_demo(net, frame_provider1, frame_provider2, args.height_size, args.cpu, args.track, args.smooth, args.cpu)
