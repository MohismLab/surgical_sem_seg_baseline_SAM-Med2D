"""SAM-Med2D 5-fold class-level evaluation on the Si5 external test set.

Forked from sam_med2d_class_eval.py. Key differences:
    - PRED layout:   /mnt/hdd2/task2/sam-med2d/predict_results_Si5/fold{F}_predict/
                     boxes_prompt/{img_base}_class{c}.png  (img_base = "{pid}_C1-XXXXX")
    - GT layout:     per-patient under
                     /mnt/hdd2/task2/Si5/processed/sam_lora/test_{pid}/masks/
                     {pid}_C1-XXXXX_class{c}.png   (image filename carries the
                     {pid}_ prefix, so GT can be located by patient parsed from
                     the prefix)
    - HD95:          computed fresh here (no prior Si5 baseline CSV exists to
                     merge from). Uses medpy.metric.binary.hd95 with the
                     standard guards (skip if pred empty, skip if gt empty).
    - Patients:      1..5 (Si5)
    - Image space:   1024x1024 (SAM-Med2D writes pred at 1024x1024 after
                     resizing back from 256; metric is computed in 1024 space
                     to match the internal SAM-Med2D evaluation convention).

Output:
    /mnt/hdd2/task2/sam-med2d/class_eval_results_Si5/
        fold{F}/class_metrics.csv               (patient, img, class, iou, dice, hd95)
        sam_med2d_class_level_5fold_Si5.csv     (5-fold summary, same columns as
                                                 main-table compatible CSV)
"""
import os
import sys
import glob
import time
import csv
import re
import logging
import numpy as np
from PIL import Image
from collections import defaultdict
from multiprocessing import Pool
import pandas as pd
from medpy import metric


PRED_BASE = "/mnt/hdd2/task2/sam-med2d/predict_results_Si5"
SI5_BASE = "/mnt/hdd2/task2/Si5/processed/sam_lora"
OUT_DIR = "/mnt/hdd2/task2/sam-med2d/class_eval_results_Si5"
ALL_CLASSES = list(range(1, 29))
ORGAN = {26, 27, 28}
INSTR = set(range(1, 26))
FOLDS = list(range(5))
PATIENTS = ["1", "2", "3", "4", "5"]
NUM_WORKERS = 8
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUT_DIR, "eval.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("sam-med2d-si5")


def get_image_size(pred_dir):
    """Read the first pred PNG to infer (H, W)."""
    for f in os.listdir(pred_dir):
        if f.endswith(".png"):
            return Image.open(os.path.join(pred_dir, f)).size[::-1]
    return (1024, 1024)


def patient_from_basename(base):
    """img_base = '{pid}_C1-XXXXX' -> pid (str)."""
    m = re.match(r"^(\d+)_", base)
    return m.group(1) if m else None


def gt_dir_for_patient(pid):
    return os.path.join(SI5_BASE, f"test_{pid}", "masks")


def evaluate_image_class_level(args):
    """
    Class-level metrics for one image (one img_base), aggregating its per-class
    binary pred PNGs and per-class GT mask PNGs into a single label map.
    """
    img_base, cls_path_list, pred_dir, H, W = args
    pid = patient_from_basename(img_base)
    if pid is None:
        return None
    gt_dir = gt_dir_for_patient(pid)
    gt_files = sorted(glob.glob(os.path.join(gt_dir, f"{img_base}_class*.png")))
    if not gt_files:
        return None

    gt_map = np.zeros((H, W), dtype=np.uint8)
    for gf in gt_files:
        cls = int(os.path.basename(gf).replace(".png", "").rsplit("_class", 1)[1])
        try:
            gt_map[np.array(Image.open(gf)) > 0] = cls
        except Exception:
            continue

    pred_map = np.zeros((H, W), dtype=np.uint8)
    for cls_id, pf in cls_path_list:
        try:
            arr = np.array(Image.open(pf))
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            pred_map[arr > 0] = cls_id
        except Exception:
            continue

    rows = []
    for cls in ALL_CLASSES:
        gt_bin = (gt_map == cls)
        if not gt_bin.any():
            continue  # label-only convention: skip classes not present in GT
        pred_bin = (pred_map == cls)
        inter = int((pred_bin & gt_bin).sum())
        union = int((pred_bin | gt_bin).sum())
        iou = inter / union if union > 0 else 0.0
        dice = (
            2 * inter / (int(pred_bin.sum()) + int(gt_bin.sum()) + 1e-8)
        )
        try:
            hd95 = (
                float(metric.binary.hd95(pred_bin, gt_bin))
                if pred_bin.any() and gt_bin.any()
                else np.nan
            )
        except Exception:
            hd95 = np.nan
        rows.append({
            "patient": pid,
            "img": img_base + ".png",
            "class": cls,
            "iou": iou,
            "dice": dice,
            "hd95": hd95,
        })
    return rows


def evaluate_fold(fold):
    pred_dir = os.path.join(PRED_BASE, f"fold{fold}_predict", "boxes_prompt")
    if not os.path.isdir(pred_dir):
        log.warning(f"Fold {fold}: pred dir not found: {pred_dir}")
        return None

    H, W = get_image_size(pred_dir)
    log.info(f"Fold {fold}: pred dir={pred_dir}, image size=({H},{W})")

    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.png")))
    img_preds = defaultdict(list)
    for f in pred_files:
        basename = os.path.basename(f).replace(".png", "")
        parts = basename.rsplit("_class", 1)
        if len(parts) != 2:
            continue
        img_preds[parts[0]].append((int(parts[1]), f))
    log.info(f"  {len(img_preds)} unique images, {len(pred_files)} pred PNGs")

    tasks = [(img_base, cls_path_list, pred_dir, H, W)
             for img_base, cls_path_list in img_preds.items()]

    t0 = time.time()
    with Pool(processes=NUM_WORKERS) as pool:
        per_image = pool.map(evaluate_image_class_level, tasks)
    per_image = [r for r in per_image if r]

    all_rows = []
    for rows in per_image:
        all_rows.extend(rows)

    # Save per-fold per-(image,class) CSV
    out_dir = os.path.join(OUT_DIR, f"fold{fold}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "class_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient", "img", "class", "iou", "dice", "hd95"])
        for r in all_rows:
            w.writerow([r["patient"], r["img"], r["class"], r["iou"], r["dice"], r["hd95"]])

    t_elapsed = (time.time() - t0) / 60
    log.info(f"  Fold {fold}: {len(all_rows)} per-class rows in {t_elapsed:.1f} min -> {csv_path}")
    return all_rows


def summarize(rows):
    df = pd.DataFrame(rows)
    df["hd95"] = df["hd95"].replace([np.inf, -np.inf], np.nan)
    organ = df[df["class"].isin(ORGAN)]
    instr = df[df["class"].isin(INSTR)]
    return {
        "n_rows": len(df),
        "Mean IOU": float(df["iou"].mean()),
        "Mean Dice": float(df["dice"].mean()),
        "Mean HD95": float(df["hd95"].mean()),
        "(Organ) Mean IOU": float(organ["iou"].mean()),
        "(Organ) Mean Dice": float(organ["dice"].mean()),
        "(Organ) Mean HD95": float(organ["hd95"].mean()),
        "(Organ) n": int(len(organ)),
        "(Instr) Mean IOU": float(instr["iou"].mean()),
        "(Instr) Mean Dice": float(instr["dice"].mean()),
        "(Instr) Mean HD95": float(instr["hd95"].mean()),
        "(Instr) n": int(len(instr)),
    }


def main():
    log.info("SAM-Med2D 5-Fold Class-Level Evaluation on Si5")
    log.info("=" * 60)

    fold_summaries = []
    for fold in FOLDS:
        rows = evaluate_fold(fold)
        if rows is None:
            continue
        s = summarize(rows)
        s["fold"] = fold
        fold_summaries.append(s)
        log.info(
            f"  Fold {fold}: n={s['n_rows']}, "
            f"mIoU={s['Mean IOU']:.4f}, mDice={s['Mean Dice']:.4f}, mHD95={s['Mean HD95']:.2f}"
        )

    if not fold_summaries:
        log.error("No folds produced metrics. Aborting.")
        return

    # Build summary table
    df = pd.DataFrame(fold_summaries)
    cols = ["fold", "n_rows", "Mean IOU", "Mean Dice", "Mean HD95",
            "(Organ) Mean IOU", "(Organ) Mean Dice", "(Organ) Mean HD95", "(Organ) n",
            "(Instr) Mean IOU", "(Instr) Mean Dice", "(Instr) Mean HD95", "(Instr) n"]
    df = df[cols]

    # Mean row
    mean_row = {"fold": "mean"}
    for c in cols[1:]:
        if c.endswith(") n") or c == "n_rows":
            mean_row[c] = int(df[c].mean())
        else:
            mean_row[c] = float(df[c].mean())
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    summary_csv = os.path.join(OUT_DIR, "sam_med2d_class_level_5fold_Si5.csv")
    df.to_csv(summary_csv, index=False)
    pd.set_option("display.float_format", lambda x: f"{x:.5f}")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print("\n", df.to_string(index=False))
    log.info(f"Saved 5-fold summary: {summary_csv}")
    log.info("done.")


if __name__ == "__main__":
    main()
