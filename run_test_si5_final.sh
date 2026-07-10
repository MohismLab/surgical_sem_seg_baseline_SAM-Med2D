#!/bin/bash
# SAM-Med2D 5-fold inference on Si5 external set — FINAL weights, GPU 1.
# Output: /mnt/hdd2/task2/sam-med2d/predict_results_Si5_final/fold{f}_predict/boxes_prompt/
# Then run: python sam_med2d_class_eval_Si5_final.py  (reads the above)

WORKDIR=/mnt/hdd2/task2/sam-med2d/predict_results_Si5_final
LOG=${WORKDIR}/sam_med2d_si5_5fold_predict.log
DATA=/mnt/hdd2/task2/Si5/processed/sam-med2d/test
mkdir -p "$WORKDIR"
exec > >(tee -a "$LOG") 2>&1

source /home/lq/Projects_qin/run_SSAM/bin/activate
export CUDA_VISIBLE_DEVICES=1
cd /home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/SAM-Med2D

declare -A CKPT=(
  [0]=/mnt/hdd2/task2/sam-med2d/models/exp_1_fold0/epoch57_batch8000_sam.pth
  [1]=/mnt/hdd2/task2/sam-med2d/models/exp_2_fold1/epoch56_batch8200_sam.pth
  [2]=/mnt/hdd2/task2/sam-med2d/models/exp_3_fold2/epoch40_batch8200_sam.pth
  [3]=/mnt/hdd2/task2/sam-med2d/models/exp_4_fold3/epoch53_batch8000_sam.pth
  [4]=/mnt/hdd2/task2/sam-med2d/models/exp_5_fold4/epoch44_batch8200_sam.pth
)

echo "==== python: $(which python) ===="
echo "==== GPU: 1 (CUDA_VISIBLE_DEVICES=1) ===="
echo "==== weights: FINAL ===="
echo "==== START: $(date) ===="

TOTAL_START=$(date +%s)
for fold in 0 1 2 3 4; do
    echo "================================================================"
    echo "[fold $fold] start: $(date) ; ckpt: ${CKPT[$fold]}"
    echo "================================================================"
    FS=$(date +%s)
    python test.py \
        --work_dir "$WORKDIR" \
        --run_name fold${fold}_predict \
        --data_path "$DATA" \
        --sam_checkpoint "${CKPT[$fold]}" \
        --boxes_prompt True \
        --save_pred True \
        --device cuda
    NF=$(ls "${WORKDIR}/fold${fold}_predict/boxes_prompt/"*.png 2>/dev/null | wc -l)
    echo "[fold $fold] done in $(( $(date +%s) - FS ))s  output=$NF pngs"
done

echo "================================================================"
echo "ALL FOLDS DONE in $(( $(date +%s) - TOTAL_START ))s total : $(date)"
echo "================================================================"
for fold in 0 1 2 3 4; do
    echo "fold${fold}_predict: $(ls ${WORKDIR}/fold${fold}_predict/boxes_prompt/*.png 2>/dev/null | wc -l) pngs"
done
