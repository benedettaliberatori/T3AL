import numpy as np
import pandas as pd
import json
import torch.utils.data as data
import torch
from torch.functional import F
import os
import yaml
import cv2
from PIL import Image
import importlib
from config.dataset_class import activity_dict, thumos_dict
from data.components.utils import map_reduce, transform, load_json


class T3ALDataset(data.Dataset):
    def __init__(self, subset, nsplit, config):
        self.subset = subset
        self.nsplit = nsplit
        self.config = config
        self.dataset = config["dataset"]["name"]
        self.feature_path = config["training"]["feature_path"]
        self.video_path = config["training"]["video_path"]
        self.video_info_path = config["dataset"]["training"]["video_info_path"]
        self.video_anno_path = config["dataset"]["training"]["video_anno_path"]
        self.split = config["dataset"]["split"]
        self.class_to_idx = activity_dict if self.dataset == "anet" else thumos_dict
        video_infos = self.get_video_info()
        self.video_annos = self.get_video_anno(video_infos)
        self.get_video_list()

    def baselinegetVideoData(self, index):
        video_idx = self.subset_mask_list[index]
        video, fps = self.loadVideo(video_idx)
        return video_idx, video, fps

    def baselinegetFeatureData(self, index):
        video_idx = self.subset_mask_list[index]
        video = self.loadFeature(video_idx)
        return video_idx, video

    def extractgetVideoData(self, index):
        video_idx = self.subset_mask_list[index]
        video, fps = self.loadVideo(video_idx)
        return video, video_idx

    def oraclegetVideoData(self, index):
        video_idx = self.subset_mask_list[index]
        video, fps = self.loadVideo(video_idx)
        annotations = self.video_annos[video_idx]["annotations"]
        start_id = [
            float(seg["segment"][0]) * fps
            for seg in annotations
            if seg["label"] in self.lbl_dict
        ]
        end_id = [
            min(float(seg["segment"][1]) * fps, video.shape[0] - 1)
            for seg in annotations
            if seg["label"] in self.lbl_dict
        ]
        label_id = [
            self.class_to_idx[seg["label"]]
            for seg in annotations
            if seg["label"] in self.lbl_dict
        ]
        sliced_video = [
            video[int(start_id[i]) : int(end_id[i])] for i in range(len(start_id))
        ]
        return sliced_video, label_id

    def __getitem__(self, index):
        if self.config["training"]["baseline"]:
            if self.config["training"]["from_features"]:
                video_idx, video = self.baselinegetFeatureData(index)
                return index, video_idx, video
            else:
                video_idx, video, fps = self.baselinegetVideoData(index)
                return index, video_idx, video, fps
        elif self.config["training"]["oracle"]:
            video_proposal, label_id = self.oraclegetVideoData(index)
            return index, video_proposal, label_id
        elif self.config["training"]["extract"]:
            video, video_idx = self.extractgetVideoData(index)
            return index, video, video_idx

    def process_frames(self, allframes):
        allframes = [
            transform()(Image.fromarray((frame).astype("uint8")).convert("RGB"))
            for frame in allframes
        ]
        return allframes

    def loadFeature(self, idx):
        video_data = np.load(os.path.join(self.feature_path, idx + ".npy"))
        video_data = torch.Tensor(video_data)
        return video_data

    def loadVideo(self, idx):
        video_extensions = [".mp4", ".mkv", ".webm"]
        video = None
        for ext in video_extensions:
            video_path = os.path.join(self.video_path, f"{idx}{ext}")
            if os.path.exists(video_path):
                video = cv2.VideoCapture(video_path)
                break
        if video is None or not video.isOpened():
            raise Exception(
                f"Video is not opened! {os.path.join(self.video_path, idx)}"
            )
        fps = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        allframes = [
            frame
            for rval, frame in (video.read() for _ in range(num_frames))
            if frame is not None
        ]
        allframes = map_reduce(self.process_frames, num_workers=8, reduce="sum")(
            allframes
        )
        video_data = torch.stack(allframes, dim=0)
        return video_data, fps

    def get_video_anno(self, video_infos):
        anno_database = load_json(self.video_anno_path)
        video_dict = {}
        for video_name in video_infos.keys():
            video_info = anno_database[video_name]
            video_subset = video_infos[video_name]["subset"]
            video_info.update({"subset": video_subset})
            video_dict[video_name] = video_info
        return video_dict

    def get_video_info(self):
        video_infos = {}
        if self.dataset == "anet":
            dataset_info = pd.DataFrame(pd.read_csv(self.video_info_path)).values[:]
            for info in dataset_info:
                video_infos[info[0]] = {"duration": info[2], "subset": info[5]}
        elif self.dataset == "thumos":
            dataset_info = json.load(open(self.video_anno_path))
            for info in dataset_info.keys():
                video_infos[info] = {
                    "duration": dataset_info[info]["duration_second"],
                    "subset": info.split("_")[1],
                }
        else:
            raise NotImplementedError("Dataset not implemented")
        return video_infos

    def get_video_splits(self):

        attributes = [
            "t2_dict_test_thumos",
            "t2_dict_train_thumos",
            "split_t2_test_thumos",
            "split_t2_train_thumos",
            "split_t1_test_thumos",
            "split_t1_train_thumos",
            "t1_dict_test_thumos",
            "t1_dict_train_thumos",
            "t1_dict_test",
            "t1_dict_train",
            "t2_dict_test",
            "t2_dict_train",
            "split_t2_test",
            "split_t2_train",
            "split_t1_test",
            "split_t1_train",
        ]
        split_dict = {}

        for attr in attributes:
            split_dict[attr] = getattr(
                importlib.import_module("config.zero_shot"),
                f"{attr}_{self.nsplit}",
                None,
            )

        temporal_dict = {}
        if self.split == 50 and self.subset == "train":
            self.lbl_dict = (
                split_dict["split_t2_train"]
                if self.dataset == "anet"
                else split_dict["split_t2_train_thumos"]
            )
            self.class_to_idx = (
                split_dict["t2_dict_train"]
                if self.dataset == "anet"
                else split_dict["t2_dict_train_thumos"]
            )
            self.num_classes = 100 if self.dataset == "anet" else 10
        elif self.split == 50 and self.subset == "validation":
            self.lbl_dict = (
                split_dict["split_t2_test"]
                if self.dataset == "anet"
                else split_dict["split_t2_test_thumos"]
            )
            self.class_to_idx = (
                split_dict["t2_dict_test"]
                if self.dataset == "anet"
                else split_dict["t2_dict_test_thumos"]
            )
            self.num_classes = 100 if self.dataset == "anet" else 10
        elif self.split == 75 and self.subset == "train":
            self.lbl_dict = (
                split_dict["split_t1_train"]
                if self.dataset == "anet"
                else split_dict["split_t1_train_thumos"]
            )
            self.class_to_idx = (
                split_dict["t1_dict_train"]
                if self.dataset == "anet"
                else split_dict["t1_dict_train_thumos"]
            )
            self.num_classes = 150 if self.dataset == "anet" else 15
        elif self.split == 75 and self.subset == "validation":
            self.lbl_dict = (
                split_dict["split_t1_test"]
                if self.dataset == "anet"
                else split_dict["split_t1_test_thumos"]
            )
            self.class_to_idx = (
                split_dict["t1_dict_test"]
                if self.dataset == "anet"
                else split_dict["t1_dict_test_thumos"]
            )
            self.num_classes = 50 if self.dataset == "anet" else 5

        for idx in self.video_annos.keys():
            labels = self.video_annos[idx]["annotations"]
            label_list = []
            if labels != []:
                for j in range(len(labels)):
                    tmp_info = labels[j]
                    gt_label = tmp_info["label"]
                    if gt_label in self.lbl_dict:
                        label_list.append([gt_label])
            label_list = list(set([item for sublist in label_list for item in sublist]))
            if not all(elem in self.lbl_dict for elem in label_list):
                continue

            if len(label_list) > 0:
                temporal_dict[idx] = {
                    "labels": label_list,
                    "video_duration": self.video_annos[idx]["duration_second"],
                }

        return temporal_dict

    def get_video_list(self):
        self.video_mask = {}
        idx_list = self.get_video_splits()
        print("No of videos in " + self.subset + " is " + str(len(idx_list.keys())))
        self.anno_final_idx = list(idx_list.keys())
        print("Loading " + self.subset + " Video Information ...")
        print("No of class", len(self.class_to_idx.keys()))
        self.subset_mask_list = list(self.anno_final_idx)

    def __len__(self):
        return len(self.subset_mask_list)
