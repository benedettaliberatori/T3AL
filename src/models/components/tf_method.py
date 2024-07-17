import torch
from torch import nn
from torch.functional import F
import open_clip
import matplotlib.pyplot as plt
import json
import os
import cv2
import re
import importlib
tokenize = open_clip.get_tokenizer("coca_ViT-L-14")

class T3AL0Net(nn.Module):
    def __init__(
        self,
        stride: int,
        kernel_size: int,
        normalize: bool,
        dataset: str,
        visualize: bool,
        remove_background: bool,
        split: int,
        setting: int, 
        video_path: str,
    ):
        super(T3AL0Net, self).__init__()

        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14", pretrained="mscoco_finetuned_laion2B-s13B-b90k"
        )
        self.model = self.model.float()
        print(f"Loaded COCA model")

        self.stride = stride
        self.kernel_size = kernel_size
        self.normalize = normalize
        self.dataset = dataset
        self.visualize = visualize
        self.remove_background = remove_background
        self.topk = 3
        self.m = 0.7
        self.split = split
        self.setting = setting
        self.video_path = video_path

        if self.dataset == "thumos":
            dict_test_name = (
                f"t2_dict_test_thumos_{split}"
                if self.setting == 50
                else f"t1_dict_test_thumos_{split}" if self.setting == 75 else None
            )
            self.annotations_path = "./data/thumos_annotations/thumos_anno_action.json"
            self.video_dir = os.path.join(self.video_path, "Thumos14/videos/")
        elif self.dataset == "anet":
            dict_test_name = (
                f"t2_dict_test_{split}"
                if self.setting == 50
                else f"t1_dict_test_{split}" if self.setting == 75 else None
            )
            self.annotations_path = (
                "./data/activitynet_annotations/anet_anno_action.json"
            )
            self.video_dir = os.path.join(self.video_path, "ActivityNetVideos/videos/")
        else:
            raise ValueError(f"Not implemented dataset: {self.dataset}")

        self.dict_test = getattr(
            importlib.import_module("config.zero_shot"), dict_test_name, None
        )

        self.cls_names = self.dict_test
        self.num_classes = len(self.cls_names)
        self.inverted_cls = {v: k for k, v in self.cls_names.items()}
        self.text_features = self.get_text_features()

        with open(self.annotations_path, "r") as f:
            self.annotations = json.load(f)

    def get_text_features(self):
        prompts = []
        for c in self.cls_names:
            c = re.sub(r"([a-z])([A-Z])", r"\1 \2", c)
            prompts.append("a video of action" + " " + c)

        text = [tokenize(p) for p in prompts]
        text = torch.stack(text)
        text = text.squeeze()
        text = text.to(next(self.model.parameters()).device)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def compute_score(self, model, x, y):
        x = x / x.norm(dim=-1, keepdim=True)
        scores = (model.logit_scale.exp() * x @ y.T).softmax(dim=-1)
        pred = scores.argmax(dim=-1)
        return pred, scores

    def infer_pseudo_labels(self, image_features):
        image_features_avg = image_features.mean(dim=0)
        self.background_embedding = image_features_avg.unsqueeze(0)
        self.text_features = self.text_features.to(image_features.device)
        _, scores_avg = self.compute_score(
            self.model,
            image_features_avg.unsqueeze(0),
            self.text_features,
        )
        _, indexes = torch.topk(scores_avg, self.topk)
        return indexes[0][0]

    def moving_average(self, data, window_size):
        padding_size = window_size
        padded_data = torch.cat(
            [
                torch.ones(padding_size).to(data.device) * data[0],
                data,
                torch.ones(padding_size).to(data.device) * data[-1],
            ]
        )
        kernel = (torch.ones(window_size) / window_size).to(data.device)
        smoothed_data = F.conv1d(padded_data.view(1, 1, -1), kernel.view(1, 1, -1))
        smoothed_data = smoothed_data.view(-1)[
            padding_size // 2 + 1 : -padding_size // 2
        ]
        return smoothed_data

    def plot_visualize(
        self, video_name, similarity, indexes, segments_gt, segment, unique_labels
    ):

        fig = plt.figure(figsize=(25, 20))
        plt.scatter(
            torch.arange(similarity.shape[0]),
            similarity.detach().cpu().numpy(),
            c="darkblue",
            s=1,
            alpha=0.5,
        )
        plt.title(video_name)
        plt.text(
            0.7,
            0.9,
            f"{self.inverted_cls.get(indexes.item(), None)}",
            fontsize=20,
            transform=plt.gcf().transFigure,
            c="red",
        )
        for i, label in enumerate(unique_labels):
            plt.text(
                0.05,
                0.9 - i * 0.05,
                label,
                fontsize=20,
                transform=plt.gcf().transFigure,
                c="green",
            )
        for i, seg in enumerate(segments_gt):
            plt.axvspan(
                seg[0],
                seg[1],
                color="green",
                alpha=0.2,
            )
        for i, seg in enumerate(segment):
            plt.axvspan(
                seg[0],
                seg[1],
                color="red",
                alpha=0.1,
            )

    def select_segments(self, similarity):
        if self.dataset == 'thumos':
            mask = similarity > similarity.mean()
        elif self.dataset == 'anet': 
            mask = similarity > self.m
        else:
            raise ValueError(f"Not implemented dataset: {self.dataset}")
        selected = torch.nonzero(mask).squeeze()
        segments = []
        if selected.numel() and selected.dim() > 0:
            interval_start = selected[0]
            for i in range(1, len(selected)):
                if selected[i] <= selected[i - 1] + self.stride:
                    continue
                else:
                    interval_end = selected[i - 1]
                    if interval_start != interval_end:
                        segments.append([interval_start.item(), interval_end.item()])
                    interval_start = selected[i]

            if interval_start != selected[-1]:
                segments.append([interval_start.item(), selected[-1].item()])

        return segments
    
    def get_video_fps(self, video_name):
        video_extensions = [".mp4", ".mkv", ".webm"]
        for ext in video_extensions:
            video_path = os.path.join(self.video_dir, video_name + ext)
            if os.path.exists(video_path):
                fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
                break
        return fps
    


    def forward(self, x):
        idx, video_name, image_features = x
        video_name = video_name[0]
        with torch.no_grad():
            image_features = image_features @ self.model.visual.proj
        image_features = image_features.squeeze(0)
        indexes = self.infer_pseudo_labels(image_features)
        class_label = self.inverted_cls[indexes.item()]
        pseudolabel_feature = self.text_features[indexes].squeeze()
        pseudolabel_feature = pseudolabel_feature / pseudolabel_feature.norm(
            dim=-1, keepdim=True
        )

        if self.remove_background:
            image_features = image_features - self.background_embedding

        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (
            self.model.logit_scale.exp() * pseudolabel_feature @ image_features_norm.T
        )

        if self.dataset == "thumos":
            similarity = self.moving_average(similarity, self.kernel_size).squeeze()
        if self.normalize:
            similarity = (similarity - similarity.min()) / (
                similarity.max() - similarity.min()
            )
            
        fps = self.get_video_fps(video_name)
    
        segments_gt = [
            anno["segment"] for anno in self.annotations[video_name]["annotations"]
        ]
        segments_gt = [
            [int(float(seg[0]) * fps), int(float(seg[1]) * fps)] for seg in segments_gt
        ]
        label_gt = [
            anno["label"] for anno in self.annotations[video_name]["annotations"]
        ]
        unique_labels = set(label_gt)

        segments = self.select_segments(similarity)

        pred_mask = torch.zeros(image_features.shape[0]).to(image_features.device)
        gt_mask = torch.zeros(image_features.shape[0]).to(image_features.device)

        if segments:
            image_features = [
                torch.mean(image_features[seg[0] : seg[1]], dim=0) for seg in segments
            ]
            image_features = torch.stack(image_features)
            pred, scores = self.compute_score(
                self.model,
                image_features,
                self.text_features.to(image_features.device),
            )
            for seg in segments:
                pred_mask[seg[0] : seg[1]] = 1
            for anno in segments_gt:
                gt_mask[anno[0] : anno[1]] = 1
            output = [
                {
                    "label": indexes.item(),
                    "score": scores[i],
                    "segment": segments[i],
                }
                for i in range(pred.shape[0])
            ]
        else:
            output = [
                {
                    "label": -1,
                    "score": 0,
                    "segment": [],
                }
            ]

        if self.visualize:
            self.plot_visualize(
                video_name, similarity, indexes, segments_gt, segments, unique_labels
            )
            return (
                video_name,
                output,
                pred_mask,
                gt_mask,
                unique_labels,
                plt,
            )
        else:
            return (
                video_name,
                output,
                pred_mask,
                gt_mask,
                unique_labels,
                None,
            )
