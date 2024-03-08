import os
import json
import numpy as np
import pandas as pd
import cv2
import importlib
import yaml
from tabulate import tabulate


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (
        (candidate_segments[:, 1] - candidate_segments[:, 0])
        + (target_segment[1] - target_segment[0])
        - segments_intersection
    )
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction["score"].values.argsort()[::-1]

    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby("video-id")

    tiou = np.zeros((0,))

    # Assigning true positive to truly ground truth instances.
    for idx, this_pred in prediction.iterrows():
        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred["video-id"])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()

        tiou_arr = segment_iou(
            this_pred[["t-start", "t-end"]].values, this_gt[["t-start", "t-end"]].values
        )

        tiou = np.append(tiou, tiou_arr)

        # Retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]["index"]] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]["index"]] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)

    recall_cumsum = tp_cumsum / npos
    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(
            precision_cumsum[tidx, :], recall_cumsum[tidx, :]
        )

    return ap, tiou


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011."""
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


def evaluate(dataset, predicted_data, split, setting, video_path):

    if dataset == "thumos":
        annotations_path = "./data/thumos_annotations/thumos_anno_action.json"
        video_dir = video_path + "Thumos14/videos/"
        tiou_thresholds = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        dict_test_name = (
            f"t2_dict_test_thumos_{split}"
            if setting == 50
            else f"t1_dict_test_thumos_{split}" if setting == 75 else None
        )
    elif dataset == "anet":
        annotations_path = "./data/activitynet_annotations/anet_anno_action.json"
        video_dir = video_path + "ActivityNetVideos/videos/"
        tiou_thresholds = np.array([0.5, 0.75, 0.95])
        dict_test_name = (
            f"t2_dict_test_{split}"
            if setting == 50
            else f"t1_dict_test_{split}" if setting == 75 else None
        )
    else:
        raise ValueError("Invalid dataset name")

    dict_test = getattr(
        importlib.import_module("config.zero_shot"), dict_test_name, None
    )
    cls_names = list(dict_test.keys())

    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    cls_dict = {v: k for v, k in enumerate(cls_names)}
    ground_truth_data = []

    for key in annotations.keys():
        if key in predicted_data.keys():
            for el in annotations[key]["annotations"]:

                if el["label"] in cls_names:
                    ground_truth_data.append(
                        {
                            "video-id": key,
                            "t-start": float(el["segment"][0]),
                            "t-end": float(el["segment"][1]),
                            "label": el["label"],
                        }
                    )

    ground_truth = pd.DataFrame(ground_truth_data)
    print(ground_truth.head())
    ground_truth.to_csv("ground_truth.csv", index=False)

    video_extensions = [".mp4", ".mkv", ".webm"]
    predicted = pd.DataFrame(columns=["video-id", "t-start", "t-end", "score"])
    predicted_list = []

    for video in predicted_data.keys():
        for ext in video_extensions:
            video_path = os.path.join(video_dir, video + ext)
            if os.path.exists(video_path):
                fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
                for el in predicted_data[video]:
                    if el["label"] != -1:
                        predicted_list.append(
                            {
                                "video-id": video,
                                "t-start": float(el["segment"][0]) / fps,
                                "t-end": float(el["segment"][1]) / fps,
                                "score": el["score"][el["label"]]
                                .detach()
                                .cpu()
                                .numpy(),
                                "label": cls_dict[el["label"]],
                            }
                        )
                break
    predicted = pd.DataFrame(predicted_list)

    print(predicted.head())
    predicted.to_csv("predicted.csv", index=False, float_format="%.1f")

    print("Ground truth labels: ", ground_truth["label"].unique())
    print("Predicted labels: ", predicted["label"].unique())

    ground_truth_by_label = ground_truth.groupby("label")
    prediction_by_label = predicted.groupby("label")
    tiou_all = np.empty((0,))
    ap_all = np.zeros((len(cls_names), len(tiou_thresholds)))
    print("\n")
    for i, class_label in enumerate(cls_names):
        if class_label in predicted["label"].unique():
            ground_truth_class = ground_truth_by_label.get_group(
                class_label
            ).reset_index(drop=True)
            prediction_class = prediction_by_label.get_group(class_label).reset_index(
                drop=True
            )
            ap, tiou = compute_average_precision_detection(
                ground_truth_class, prediction_class, tiou_thresholds
            )
            ap = [round(x * 100, 1) for x in ap]
            ap_all[i] = ap
            print(class_label, ap)
            print("\n")

            tiou_all = np.append(tiou_all, tiou)

    table_data = []
    for i, class_label in enumerate(cls_names):
        table_data.append([class_label] + list(ap_all[i]))
    table_data.append(["IoU"] + list(tiou_all))

    headers = ["Class"] + [f"AP@{int(tiou)}%" for tiou in tiou_thresholds]
    print(tabulate(table_data, headers, tablefmt="latex_raw"))

    ap_all = np.mean(ap_all, axis=0)
    print(ap_all)
    tiou_avg = np.mean(tiou_all)
    print("Average TIOU: ", tiou_avg)

    return ap_all, tiou_avg
