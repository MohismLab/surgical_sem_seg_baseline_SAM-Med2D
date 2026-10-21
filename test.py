from segment_anything import sam_model_registry
import torch.nn as nn
import torch
import argparse
import os
from utils import FocalDiceloss_IoULoss, generate_point, save_masks
from torch.utils.data import DataLoader
from DataLoader import TestingDataset
from metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
import logging
import datetime
import cv2
import random
import csv
import json
from collections import defaultdict


"""
CUDA_VISIBLE_DEVICES=1 nohup python /home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/SAM-Med2D/test.py
 --work_dir /mnt/hdd2/task2/sam-med2d/predict
 --data_path /mnt/hdd2/task2/sam_lora/test
 --sam_checkpoint /mnt/hdd2/task2/sam-med2d/models/sam-med2d/epoch50_sam_best_on_epoch23.pth
 --save_pred True > /mnt/hdd2/task2/sam-med2d/predict/predict.log 2>&1 &

新增平均hd95+单独病人的iou/dice/hd95

不需要predict后的mask则删除上述命令中的 --save_pred True

生成f"/mnt/hdd2/task2/sam-med2d/predict/fold{fold}_predict_per_image_detailed_metrics.csv"的代码在：
/home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/SAM-Med2D/generate_detailed_metrics.py
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="sammed", help="run model name")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--data_path", type=str, default="data_demo", help="train data path") 
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice', 'hd95'], help="metrics")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--boxes_prompt", type=bool, default=True, help="use boxes prompt")
    parser.add_argument("--point_num", type=int, default=1, help="point num")
    parser.add_argument("--iter_point", type=int, default=1, help="iter num") 
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--prompt_path", type=str, default=None, help="fix prompt path")
    parser.add_argument("--save_pred", type=bool, default=False, help="save result")
    parser.add_argument("--skip_metrics", action="store_true", help="skip per-sample metrics computation")
    args = parser.parse_args()
    if args.iter_point > 1:
        args.point_num = 1
    return args


def extract_patient_id(img_name):
    patient_id = img_name.split('_')[0]
    return patient_id


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key=='image' or key=='label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                 device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
        )
    
    if ori_h < image_size and ori_w < image_size:
        top = torch.div((image_size - ori_h), 2, rounding_mode='trunc')  #(image_size - ori_h) // 2
        left = torch.div((image_size - ori_w), 2, rounding_mode='trunc') #(image_size - ori_w) // 2
        masks = masks[..., top : ori_h + top, left : ori_w + left]
        pad = (top, left)
    else:
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        pad = None 
    return masks, pad


def prompt_and_decoder(args, batched_input, ddp_model, image_embeddings):
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = ddp_model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

        low_res_masks, iou_predictions = ddp_model.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = ddp_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=args.multimask,
        )
    
    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)
    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions


def is_not_saved(save_path, mask_name):
    masks_path = os.path.join(save_path, f"{mask_name}")
    if os.path.exists(masks_path):
        return False
    else:
        return True


def main(args):
    print('*'*100)
    for key, value in vars(args).items():
        print(key + ': ' + str(value))
    print('*'*100)

    all_metrics = list(args.metrics)
    if 'hd95' not in all_metrics:
        all_metrics.append('hd95')

    model = sam_model_registry[args.model_type](args).to(args.device) 

    criterion = FocalDiceloss_IoULoss()
    test_dataset = TestingDataset(data_path=args.data_path, image_size=args.image_size, mode='test', requires_name=True, point_num=args.point_num, return_ori_mask=True, prompt_path=args.prompt_path)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)
    print('Test data:', len(test_loader))

    test_pbar = tqdm(test_loader)
    l = len(test_loader)

    model.eval()
    test_loss = []
    test_metrics = {}
    prompt_dict = {}

    patient_metrics = defaultdict(lambda: defaultdict(list))

    for i, batched_input in enumerate(test_pbar):
        batched_input = to_device(batched_input, args.device)
        ori_labels = batched_input["ori_label"]
        original_size = batched_input["original_size"]
        labels = batched_input["label"]
        img_name = batched_input['name'][0]

        patient_id = extract_patient_id(img_name)

        if args.prompt_path is None:
            prompt_dict[img_name] = {
                        "boxes": batched_input["boxes"].squeeze(1).cpu().numpy().tolist(),
                        "point_coords": batched_input["point_coords"].squeeze(1).cpu().numpy().tolist(),
                        "point_labels": batched_input["point_labels"].squeeze(1).cpu().numpy().tolist()
                        }

        with torch.no_grad():
            image_embeddings = model.image_encoder(batched_input["image"])

        if args.boxes_prompt:
            save_path = os.path.join(args.work_dir, args.run_name, "boxes_prompt")
            batched_input["point_coords"], batched_input["point_labels"] = None, None
            masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)
            points_show = None

        else:
            save_path = os.path.join(f"{args.work_dir}", args.run_name, f"iter{args.iter_point if args.iter_point > 1 else args.point_num}_prompt")
            batched_input["boxes"] = None
            point_coords, point_labels = [batched_input["point_coords"]], [batched_input["point_labels"]]
     
            for iter in range(args.iter_point):
                masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)
                if iter != args.iter_point-1:
                    batched_input = generate_point(masks, labels, low_res_masks, batched_input, args.point_num)
                    batched_input = to_device(batched_input, args.device)
                    point_coords.append(batched_input["point_coords"])
                    point_labels.append(batched_input["point_labels"])
                    batched_input["point_coords"] = torch.concat(point_coords,dim=1)
                    batched_input["point_labels"] = torch.concat(point_labels, dim=1)
  
            points_show = (torch.concat(point_coords, dim=1), torch.concat(point_labels, dim=1))

        masks, pad = postprocess_masks(low_res_masks, args.image_size, original_size)
        if args.save_pred:
            save_masks(masks, save_path, img_name, args.image_size, original_size, pad, batched_input.get("boxes", None), points_show)

        loss = criterion(masks, ori_labels, iou_predictions)
        test_loss.append(loss.item())

        if not args.skip_metrics:
            test_batch_metrics = SegMetrics(masks, ori_labels, all_metrics)
            test_batch_metrics = [float('{:.4f}'.format(metric)) for metric in test_batch_metrics]

            for j, metric_name in enumerate(all_metrics):
                patient_metrics[patient_id][metric_name].append(test_batch_metrics[j])
  
    average_loss = np.mean(test_loss)
    if args.prompt_path is None:
        with open(os.path.join(args.work_dir,f'{args.image_size}_prompt.json'), 'w') as f:
            json.dump(prompt_dict, f, indent=2)

    if args.skip_metrics:
        print(f"\nTest loss: {average_loss:.4f} (per-sample metrics skipped; "
              f"recomputed by the external-validation wrapper)")
        return

    test_metrics = {}
    for metric_name in all_metrics:
        all_values = []
        for pid in patient_metrics:
            all_values.extend(patient_metrics[pid][metric_name])
        test_metrics[metric_name] = '{:.4f}'.format(np.nanmean(all_values))

    print("\n" + "="*80)
    print(f"Overall Test Results")
    print("="*80)
    print(f"Test loss: {average_loss:.4f}, metrics: {test_metrics}")

    print("\n" + "="*80)
    print(f"Per-Patient Metrics")
    print("="*80)
    header = f"{'Patient ID':<15}"
    for metric_name in all_metrics:
        header += f"{'mean_' + metric_name:>15}"
    header += f"{'num_samples':>15}"
    print(header)
    print("-"*len(header))


    csv_rows = []

    for patient_id in sorted(patient_metrics.keys()):
        metrics_dict = patient_metrics[patient_id]
        row = f"{patient_id:<15}"
        csv_row = {'patient_id': patient_id}
        
        for metric_name in all_metrics:
            values = metrics_dict[metric_name]

            if metric_name == 'hd95':
                mean_val = float(np.nanmean(values))
            else:
                mean_val = float(np.mean(values))
            row += f"{mean_val:>15.4f}"
            csv_row[f'mean_{metric_name}'] = f'{mean_val:.4f}'
        
        num_samples = len(metrics_dict[all_metrics[0]])
        row += f"{num_samples:>15}"
        csv_row['num_samples'] = num_samples
        
        print(row)
        csv_rows.append(csv_row)

    print("-"*len(header))
    overall_row = f"{'Overall':<15}"
    overall_csv = {'patient_id': 'Overall'}
    for metric_name in all_metrics:
        overall_row += f"{float(test_metrics[metric_name]):>15.4f}"
        overall_csv[f'mean_{metric_name}'] = test_metrics[metric_name]
    overall_row += f"{l:>15}"
    overall_csv['num_samples'] = l
    print(overall_row)
    csv_rows.append(overall_csv)

    csv_path = os.path.join(args.work_dir, f'{args.run_name}_per_patient_metrics.csv')
    os.makedirs(args.work_dir, exist_ok=True)
    fieldnames = ['patient_id'] + [f'mean_{m}' for m in all_metrics] + ['num_samples']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nPer-patient metrics saved to: {csv_path}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
