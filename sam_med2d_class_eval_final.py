import os, glob, time, csv
import numpy as np
from PIL import Image
from collections import defaultdict
from multiprocessing import Pool
from medpy import metric
import pandas as pd

PRED_BASE = "/mnt/hdd2/task2/sam-med2d/predict"
GT_MASK_DIR = "/mnt/hdd2/task2/sam_lora/test/masks"
OUT_DIR = "/mnt/hdd2/task2/sam-med2d/class_eval_results_final"
ALL_CLASSES = list(range(1, 29))
ORGAN = {26, 27, 28}
INSTR = set(range(1, 26))
FOLDS = list(range(5))
os.makedirs(OUT_DIR, exist_ok=True)


def get_image_size(pred_dir):
    for f in os.listdir(pred_dir):
        if f.endswith('.png'):
            return Image.open(os.path.join(pred_dir, f)).size[::-1]
    return (1024, 1024)


def hd95_safe(pred_bin, gt_bin):
    try:
        if pred_bin.sum() > 0 and gt_bin.sum() > 0:
            return float(metric.binary.hd95(pred_bin, gt_bin))
    except Exception:
        pass
    return np.nan


def evaluate_fold_class_level(fold):
    pred_dir = os.path.join(PRED_BASE, f"fold{fold}_predict_final", "boxes_prompt")
    if not os.path.exists(pred_dir):
        print(f"Fold {fold}: prediction dir not found: {pred_dir}")
        return None

    H, W = get_image_size(pred_dir)
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.png")))
    img_preds = defaultdict(list)
    for f in pred_files:
        basename = os.path.basename(f).replace(".png", "")
        parts = basename.rsplit("_class", 1)
        if len(parts) != 2:
            continue
        img_preds[parts[0]].append((int(parts[1]), f))

    rows = []
    for img_key, cls_path_list in img_preds.items():
        gt_files = sorted(glob.glob(os.path.join(GT_MASK_DIR, f"{img_key}_class*.png")))
        if not gt_files:
            continue
        gt_map = np.zeros((H, W), dtype=np.uint8)
        for gf in gt_files:
            cls = int(os.path.basename(gf).replace('.png', '').rsplit('_class', 1)[1])
            gt_map[np.array(Image.open(gf)) > 0] = cls
        pred_map = np.zeros((H, W), dtype=np.uint8)
        for cls_id, pf in cls_path_list:
            try:
                img = np.array(Image.open(pf))
                if img.ndim == 3:
                    img = img[:, :, 0]
                pred_map[img > 0] = cls_id
            except Exception:
                continue

        for cls in ALL_CLASSES:
            gt_bin = (gt_map == cls)
            if not gt_bin.any():           # label-only: skip classes absent in GT
                continue
            pred_bin = (pred_map == cls)
            inter = (pred_bin & gt_bin).sum()
            union = (pred_bin | gt_bin).sum()
            iou = inter / union if union > 0 else 0.0
            dice = 2 * inter / (pred_bin.sum() + gt_bin.sum() + 1e-8)
            hd95 = hd95_safe(pred_bin, gt_bin)
            rows.append({"image": img_key, "class": cls,
                         "iou": float(iou), "dice": float(dice), "hd95": hd95})

    # per-row detailed csv (mirror SAM-LoRA fold{f}/class_metrics.csv)
    fold_dir = os.path.join(OUT_DIR, f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(fold_dir, "class_metrics.csv"), index=False)

    df = pd.DataFrame(rows)
    df["hd95"] = df["hd95"].replace([np.inf, -np.inf], np.nan)
    org = df[df["class"].isin(ORGAN)]
    ins = df[df["class"].isin(INSTR)]
    return {
        "fold": fold, "n_images": len(img_preds), "n_rows": len(df),
        "mIoU": df["iou"].mean(), "mDice": df["dice"].mean(), "mHD95": df["hd95"].mean(),
        "Organ mIoU": org["iou"].mean(), "Organ mDice": org["dice"].mean(), "Organ HD95": org["hd95"].mean(),
        "Instr mIoU": ins["iou"].mean(), "Instr mDice": ins["dice"].mean(), "Instr HD95": ins["hd95"].mean(),
    }


if __name__ == "__main__":
    t0 = time.time()
    print(f"SAM-Med2D FINAL class-level eval start {time.strftime('%H:%M:%S')}")
    with Pool(5) as pool:
        fold_results = [r for r in pool.map(evaluate_fold_class_level, FOLDS) if r is not None]
    fold_results.sort(key=lambda r: r["fold"])

    print(f"\n{'Fold':<6}{'mIoU':>10}{'mDice':>10}{'mHD95':>10}{'OrgIoU':>10}{'OrgDice':>10}{'OrgHD95':>10}{'InsIoU':>10}{'InsDice':>10}{'InsHD95':>10}")
    for r in fold_results:
        print(f"{r['fold']:<6}{r['mIoU']:>10.4f}{r['mDice']:>10.4f}{r['mHD95']:>10.2f}{r['Organ mIoU']:>10.4f}{r['Organ mDice']:>10.4f}{r['Organ HD95']:>10.2f}{r['Instr mIoU']:>10.4f}{r['Instr mDice']:>10.4f}{r['Instr HD95']:>10.2f}")

    avg = lambda k: float(np.nanmean([r[k] for r in fold_results]))
    keys = ["mIoU", "mDice", "mHD95", "Organ mIoU", "Organ mDice", "Organ HD95", "Instr mIoU", "Instr mDice", "Instr HD95"]
    m = {k: avg(k) for k in keys}
    print(f"{'mean':<6}{m['mIoU']:>10.4f}{m['mDice']:>10.4f}{m['mHD95']:>10.2f}{m['Organ mIoU']:>10.4f}{m['Organ mDice']:>10.4f}{m['Organ HD95']:>10.2f}{m['Instr mIoU']:>10.4f}{m['Instr mDice']:>10.4f}{m['Instr HD95']:>10.2f}")

    csv_path = os.path.join(OUT_DIR, "sam_med2d_class_level_5fold_final.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["fold", "n_images", "n_rows", "Mean IOU", "Mean Dice", "Mean HD95",
                    "(Organ) Mean IOU", "(Organ) Mean Dice", "(Organ) Mean HD95",
                    "(Instr) Mean IOU", "(Instr) Mean Dice", "(Instr) Mean HD95"])
        for r in fold_results:
            w.writerow([r["fold"], r["n_images"], r["n_rows"], r["mIoU"], r["mDice"], r["mHD95"],
                        r["Organ mIoU"], r["Organ mDice"], r["Organ HD95"],
                        r["Instr mIoU"], r["Instr mDice"], r["Instr HD95"]])
        w.writerow(["mean", "", "", m["mIoU"], m["mDice"], m["mHD95"],
                    m["Organ mIoU"], m["Organ mDice"], m["Organ HD95"],
                    m["Instr mIoU"], m["Instr mDice"], m["Instr HD95"]])
    print(f"\nSaved aggregate: {csv_path}")
    print(f"Per-fold detailed: {OUT_DIR}/fold{{0-4}}/class_metrics.csv")
    print(f"Total {(time.time()-t0)/60:.1f} min")
