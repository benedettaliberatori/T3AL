import torch
from torch import nn
from transformers import CLIPTokenizer, CLIPModel
import yaml
from open_clip.tokenizer import tokenize as tokenize
import open_clip
import json
import re
import os
import importlib
tokenize = open_clip.get_tokenizer("coca_ViT-L-14")

class BaselineNet(nn.Module):
    def __init__(
        self,
        model_name: str,
        dataset: str,
        p: float,
        split: int,
        setting: int,
        video_path: str,
    ):
        super(BaselineNet, self).__init__()

        self.model_name = model_name
        self.dataset = dataset
        self.p = p
        self.split = split
        self.setting = setting
        self.video_path = video_path

        if self.model_name == "clip-16":
            self.model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch16"
            ).float()
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-base-patch16"
            )
            print(f"Loaded CLIP ViT-B/16 model")
            self.encode_function = self.model.get_image_features
        elif self.model_name == "clip-32":
            self.model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).float()
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            print(f"Loaded CLIP ViT-B/32 model")
            self.encode_function = self.model.get_image_features
        elif self.model_name == "coca":
            self.model, _, transform = open_clip.create_model_and_transforms(
                model_name="coca_ViT-L-14",
                pretrained="mscoco_finetuned_laion2B-s13B-b90k",
            )
            print(f"Loaded COCA model")
            self.model = self.model.float()
            self.encode_function = self.model.encode_image
        else:
            raise ValueError(f"Requested model is not available {self.model_name}")

        if self.dataset == "thumos":
            dict_test_name = (
                f"t2_dict_test_thumos_{split}"
                if self.setting == 50
                else f"t1_dict_test_thumos_{split}" if self.setting == 75 else None
            )
            self.annotations_path = (
                "./data/thumos_annotations/thumos_anno_action.json"
            )
            self.video_dir = os.path.join(self.video_path, "/Thumos14/videos/")
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
            raise ValueError("Dataset not implemented")
        
        self.dict_test = getattr(
            importlib.import_module("config.zero_shot"), dict_test_name, None
        )
        self.cls_names = self.dict_test
        self.num_classes = len(self.cls_names)
        self.inverted_cls = {v: k for k, v in self.cls_names.items()}
        self.text_features = self.get_text_features(self.model)

        with open(self.annotations_path, "r") as f:
            self.annotations = json.load(f)
            
        with open(f"./config/{self.dataset}.yaml", "r", encoding="utf-8") as f:
            tmp = f.read()
            self.config = yaml.load(tmp, Loader=yaml.FullLoader)

    def get_text_features(self, model):
        prompts = []
        for c in self.cls_names:
            c = re.sub(r"([a-z])([A-Z])", r"\1 \2", c)
            prompts.append("a video of action" + " " + c)
        if "clip" in self.model_name:
            tokens = self.tokenizer(prompts, padding=True, return_tensors="pt")
            text_features = model.get_text_features(**tokens)
        elif self.model_name == "coca":
            text = [tokenize(p) for p in prompts]
            text = torch.stack(text)
            text = text.squeeze()
            text_features = model.encode_text(text)
        else:
            raise ValueError(f"Requested model is not available {self.model_name}")
        text_features = text_features.to("cuda")
        return text_features

    def compute_score(self, x, y):
        x = x / x.norm(dim=-1, keepdim=True)
        y = y / y.norm(dim=-1, keepdim=True)
        scores = (self.model.logit_scale.exp() * x @ y.T).softmax(dim=-1)
        pred = scores
        return pred, scores

    def get_image_features(self, images):
        chunk_size = 100
        t = images.shape[0]
        image_features = [
            self.encode_function(images[i * chunk_size : (i + 1) * chunk_size])
            for i in range(t // chunk_size)
        ]
        if t % chunk_size != 0:
            img = self.encode_function(images[t - t % chunk_size :])
            image_features.append(img)
        return image_features

    def find_segments_of_ones(self, binary_mask):
        segments = []
        current_segment = None
        for idx, value in enumerate(binary_mask):
            if value == 1:
                if current_segment is None:
                    current_segment = [idx]
            elif current_segment is not None:
                current_segment.append(idx - 1)
                segments.append(current_segment)
                current_segment = None
        if current_segment is not None:
            current_segment.append(len(binary_mask) - 1)
            segments.append(current_segment)
        return segments

    def forward(self, x):

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        if not self.config["training"]["from_features"]:
            start_event.record()
            idx, video_name, images, fps = x
            video_name = video_name[0]
            images = images.squeeze(0)  
            image_features = self.get_image_features(images)
            image_features = torch.cat(image_features, dim=0)
            end_event.record()
            end_event.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            print(
                f"Elapsed Time for feature extraction per frame: {elapsed_time/image_features.shape[0]} ms\n"
            )
        else:
            idx, video_name, image_features = x
            video_name = video_name[0]
        pred, scores = self.compute_score(
            image_features, self.text_features
        ) 
        pred = pred.squeeze()
        scores = scores.squeeze()
        
        pred = scores.argmax(dim=-1)
        pred_scores = scores.max(dim=-1).values
        pred_mask = torch.where(pred_scores > self.p, 1, 0)
        
        segments = self.find_segments_of_ones(pred_mask)
        output = []
        for segment in segments:
            segment_scores = scores[segment[0] : segment[1]]
            if segment_scores.numel() > 0:
                segment_scores = segment_scores.mean(dim=0)
                pred = segment_scores.argmax()
                output.append(
                    {
                        "label": pred.item(),
                        "score": segment_scores,
                        "segment": segment,
                    }
                )
        return (video_name, output)
