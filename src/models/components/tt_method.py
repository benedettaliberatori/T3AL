import torch
from torch import nn
from torch.functional import F
from src.models.components.loss import ByolLoss
import open_clip
import matplotlib.pyplot as plt
import json
import os
import cv2
import re
import copy
import importlib

tokenize = open_clip.get_tokenizer("coca_ViT-L-14")

class T3ALNet(nn.Module):
    def __init__(
        self,
        p: float,
        stride: int,
        randper: int,
        kernel_size: int,
        n: int,
        normalize: bool,
        dataset: str,
        visualize: bool,
        text_projection: bool,
        text_encoder: bool,
        image_projection: bool,
        logit_scale: bool,
        remove_background: bool,
        ltype: str,
        steps: int,
        refine_with_captions: bool,
        split: int,
        setting: int,
        video_path: str,
    ):
        super(T3ALNet, self).__init__()

        self.stride = stride
        self.randper = randper
        self.p = p
        self.n = n
        self.normalize = normalize
        self.text_projection = text_projection
        self.text_encoder = text_encoder
        self.image_projection = image_projection
        self.logit_scale = logit_scale
        self.remove_background = remove_background
        self.ltype = ltype
        self.steps = steps
        self.refine_with_captions = refine_with_captions
        self.split = split
        self.setting = setting
        self.dataset = dataset
        self.visualize = visualize
        self.kernel_size = kernel_size
        self.video_path = video_path
        self.topk = 3
        self.m = 0.7

        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14", pretrained="mscoco_finetuned_laion2B-s13B-b90k"
        )
        self.model = self.model.float()
        print(f"Loaded COCA model")

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
        self.text_features = self.get_text_features(self.model)

        with open(self.annotations_path, "r") as f:
            self.annotations = json.load(f)

        if self.ltype == "BCE":
            self.tta_loss = torch.nn.BCEWithLogitsLoss()
        elif "BYOL" in self.ltype:
            self.tta_loss = ByolLoss()
        else:
            raise ValueError(f"Not implemented loss type: {self.ltype}")

    def get_text_features(self, model):
        prompts = []
        for c in self.cls_names:
            c = re.sub(r"([a-z])([A-Z])", r"\1 \2", c)
            prompts.append("a video of action" + " " + c)

        text = [tokenize(p) for p in prompts]
        text = torch.stack(text)
        text = text.squeeze()
        text = text.to(next(model.parameters()).device)
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def compute_score(self, x, y):
        x = x / x.norm(dim=-1, keepdim=True)
        scores = (self.model.logit_scale.exp() * x @ y.T).softmax(dim=-1)
        pred = scores.argmax(dim=-1)
        return pred, scores

    def select_segments(self, similarity):
        
        if self.dataset == "thumos":
            mask = similarity > similarity.mean()
        elif self.dataset == "anet":
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

    def infer_pseudo_labels(self, image_features):
        image_features_avg = image_features.mean(dim=0)
        self.background_embedding = image_features_avg.unsqueeze(0)
        self.text_features = self.text_features.to(image_features.device)
        _, scores_avg = self.compute_score(
            image_features_avg.unsqueeze(0),
            self.text_features,
        )
        _, indexes = torch.topk(scores_avg, self.topk)
        return indexes[0][0], scores_avg

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

    def get_segments_gt(self, video_name, fps):
        segments_gt = [
            anno["segment"]
            for anno in self.annotations[video_name]["annotations"]
            if anno["label"] in self.cls_names
        ]
        segments_gt = [
            [int(float(seg[0]) * fps), int(float(seg[1]) * fps)] for seg in segments_gt
        ]
        label_gt = [
            anno["label"]
            for anno in self.annotations[video_name]["annotations"]
            if anno["label"] in self.cls_names
        ]
        unique_labels = set(label_gt)
        return segments_gt, unique_labels

    def get_indices(self, signal):

        if (100 * self.n) >= signal.shape[1]:
            pindices = torch.arange(signal.shape[1]).to("cuda")
            nindices = torch.arange(signal.shape[1]).to("cuda")
        else:
            pindices = torch.topk(signal, (100 * self.n) % signal.shape[1])[1]
            nindices = torch.topk(-signal, (100 * self.n) % signal.shape[1])[1]
        pindices = pindices.squeeze().sort()[0]
        nindices = nindices.squeeze().sort()[0]
        if pindices.shape[0] < self.n:
            pindices = pindices.repeat_interleave(self.n // pindices.shape[0] + 1)
            pindices = pindices[: self.n]
        if nindices.shape[0] < self.n:
            nindices = nindices.repeat_interleave(self.n // nindices.shape[0] + 1)
            nindices = nindices[: self.n]
        pindices = pindices[:: (len(pindices) - 1) // (self.n - 1)][: self.n]
        nindices = nindices[:: (len(nindices) - 1) // (self.n - 1)][: self.n]
        pindices = torch.clamp(
            pindices
            + torch.randint(-self.randper, self.randper, (self.n,)).to(signal.device),
            0,
            signal.shape[1] - 1,
        )
        nindices = torch.clamp(
            nindices
            + torch.randint(-self.randper, self.randper, (self.n,)).to(signal.device),
            0,
            signal.shape[1] - 1,
        )

        return pindices, nindices

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
        return plt

    def compute_tta_embedding(self, class_label, device):
        class_label = re.sub(r"([a-z])([A-Z])", r"\1 \2", class_label)
        class_label = "a video of action" + " " + class_label
        text = tokenize(class_label).to(device)
        tta_emb = self.model.encode_text(text)
        tta_emb = tta_emb / tta_emb.norm(dim=-1, keepdim=True)
        return tta_emb

    def forward(self, x, optimizer):
        idx, video_name, image_features_pre = x
        image_features_pre = copy.deepcopy(image_features_pre)
        video_name = video_name[0]
        fps = self.get_video_fps(video_name)

        if not self.image_projection:
            image_features = image_features_pre
            image_features = image_features.squeeze(0)
        else:
            image_features_pre.requires_grad = True
            with torch.no_grad():
                image_features = image_features_pre @ self.model.visual.proj
                image_features = image_features.squeeze(0)
                
        indexes, _ = self.infer_pseudo_labels(image_features)
        class_label = self.inverted_cls[indexes.item()]

        segments_gt, unique_labels = self.get_segments_gt(video_name, fps)

        for _ in range(self.steps):
            if self.image_projection:
                image_features = (image_features_pre @ self.model.visual.proj).squeeze(
                    0
                )
                before_optimization_parameters_image_encoder = copy.deepcopy(
                    self.model.visual.state_dict()
                )
                before_optimization_image_projection = copy.deepcopy(
                    self.model.visual.proj
                )

            if self.text_projection:
                before_optimization_text_projection = copy.deepcopy(
                    self.model.text.text_projection
                )
                before_optimization_parameters_text_encoder = copy.deepcopy(
                    self.model.text.state_dict()
                )
            else:
                before_optimization_parameters_text_encoder = copy.deepcopy(
                    self.model.text.state_dict()
                )
            before_optimization_logit_scale = copy.deepcopy(self.model.logit_scale)

            tta_emb = self.compute_tta_embedding(class_label, image_features.device)
            
            features = image_features - self.background_embedding if self.remove_background else image_features
            similarity = self.model.logit_scale.exp() * tta_emb @ features.T
            
            if self.dataset == "thumos":
                similarity = self.moving_average(
                    similarity.squeeze(), self.kernel_size
                ).unsqueeze(0)
            
            pindices, nindices = self.get_indices(similarity)
            image_features_p, image_features_n = image_features[pindices], image_features[nindices]
            image_features_p = image_features_p / image_features_p.norm(
                dim=-1, keepdim=True
            )
            image_features_n = image_features_n / image_features_n.norm(
                dim=-1, keepdim=True
            )
            similarity_p = (
                self.model.logit_scale.exp() * tta_emb @ image_features_p.T
            )
            similarity_n = (
                self.model.logit_scale.exp() * tta_emb @ image_features_n.T
            )
            similarity = torch.cat(
                [similarity_p.squeeze(), similarity_n.squeeze()], dim=0
            )
            gt = torch.cat(
                [
                    torch.ones(similarity_p.shape[1]),
                    torch.zeros(similarity_n.shape[1]),
                ],
                dim=0,
            ).to(similarity.device)
            
            
            if self.ltype in ["BYOL", "BCE"]:
                tta_loss = self.tta_loss(similarity, gt)
            elif self.ltype == "BYOLfeat":
                tta_loss = self.tta_loss(similarity, gt) + self.tta_loss(
                    image_features_p,
                    tta_emb.repeat_interleave(image_features_p.shape[0], dim=0),
                )
            else:
                raise ValueError(f"Not implemented loss type: {self.ltype}")

            tta_loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            
        if self.text_projection:
            assert not torch.equal(
                before_optimization_text_projection,
                copy.deepcopy(self.model.text.text_projection),
            ), f"Parameter text_projection has not been updated."

        if self.image_projection:
            assert not torch.equal(
                before_optimization_image_projection,
                copy.deepcopy(self.model.visual.proj),
            ), f"Parameter has not been updated."

        with torch.no_grad():
            tta_emb = self.compute_tta_embedding(class_label, image_features.device)
            
            if self.remove_background:
                image_features = image_features - self.background_embedding
            
            image_features_norm = image_features / image_features.norm(
                dim=-1, keepdim=True
            )
            similarity = self.model.logit_scale.exp() * tta_emb @ image_features_norm.T
            
            if self.dataset == "thumos":
                similarity = self.moving_average(similarity.squeeze(), self.kernel_size)
            if self.normalize:
                similarity = (similarity - similarity.min()) / (
                    similarity.max() - similarity.min()
                )
            similarity = similarity.squeeze()
            segments = self.select_segments(similarity)
            pred_mask = torch.zeros(image_features.shape[0]).to(image_features.device)
            gt_mask = torch.zeros(image_features.shape[0]).to(image_features.device)
            after_optimization_text_encoder = copy.deepcopy(
                self.model.text.state_dict()
            )
            after_optimization_logit_scale = copy.deepcopy(self.model.logit_scale)
            
            
            if self.refine_with_captions and len(segments) > 1:
                self.model.locit_scale = before_optimization_logit_scale
                self.model.text.load_state_dict(
                    before_optimization_parameters_text_encoder
                )
                with open(f"./captions/{video_name}.txt", "r") as f:
                    captions = f.readlines()
                captions = [
                    (int(c.split("-")[0].split(".")[0]) * 3, c.split("-")[1])
                    for c in captions
                ]
                captions_per_segment = [[] for _ in range(len(segments))]
                image_features_per_segment = [[] for _ in range(len(segments))]
                for i, seg in enumerate(segments):
                    image_features_per_segment[i] = image_features[seg[0] : seg[1]]
                    for cap in captions:
                        if cap[0] >= seg[0] and cap[0] <= seg[1]:
                            captions_per_segment[i].append((cap[1]))
                captions_per_segment = [
                    [tokenize(p) for p in cap] for cap in captions_per_segment
                ]
                segments = [
                    seg
                    for seg, cap in zip(segments, captions_per_segment)
                    if len(cap) > 0
                ]
                captions_per_segment = [
                    cap for cap in captions_per_segment if len(cap) > 0
                ]
                captions_per_segment = [
                    torch.stack(cap) for cap in captions_per_segment
                ]
                captions_per_segment = [cap.squeeze() for cap in captions_per_segment]
                captions_per_segment = [
                    cap.to(image_features.device) for cap in captions_per_segment
                ]
                captions_per_segment = [
                    cap.unsqueeze(0) if len(cap.shape) == 1 else cap
                    for cap in captions_per_segment
                ]
                captions_per_segment = [
                    self.model.encode_text(cap) for cap in captions_per_segment
                ]
                captions_per_segment = [cap.mean(dim=0) for cap in captions_per_segment]
                captions_per_segment = [
                    cap / cap.norm(dim=-1, keepdim=True) for cap in captions_per_segment
                ]
                similarity_with_other_captions = []
                for cap in captions_per_segment:
                    similarity_with_other_captions.append(
                        cap @ torch.stack(captions_per_segment).T
                    )
                segments = [
                    seg
                    for seg, sim in zip(segments, similarity_with_other_captions)
                    if torch.sum(sim > self.p) > len(segments) // 2
                ]
                self.model.logit_scale = after_optimization_logit_scale
                self.model.text.load_state_dict(after_optimization_text_encoder)

            if segments:
                image_features = [
                    torch.mean(image_features[seg[0] : seg[1]], dim=0)
                    for seg in segments
                ]
                text_features = self.get_text_features(self.model)
                image_features = torch.stack(image_features)
                pred, scores = self.compute_score(
                    image_features,
                    text_features.to(image_features.device),
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
                    for i in range((len(segments)))
                ]
            else:
                output = [
                    {
                        "label": -1,
                        "score": 0,
                        "segment": [],
                    }
                ]
        self.model.text.load_state_dict(before_optimization_parameters_text_encoder)
        self.model.logit_scale = before_optimization_logit_scale
        if self.image_projection:
            self.model.visual.load_state_dict(
                before_optimization_parameters_image_encoder
            )
            
        if self.visualize:
            sim_plot = self.plot_visualize(
                video_name, similarity, indexes, segments_gt, segments, unique_labels
            )
        else:
            sim_plot = None
        return (
            video_name,
            output,
            pred_mask,
            gt_mask,
            unique_labels,
            sim_plot,
        )