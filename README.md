--------- Preprocessing
nohup python preprocess.py   --src /home/kanishk/mri2hist/ahead_t1   --out data   > data-preprocess.log 2>&1 &
---------

--------- train
nohup python train.py \
  --train data/train_npz \
  --val data/val_npz \
  --out checkpoints \
  --epochs 100 \
  --batch_size 8 \
  --lr 1e-4 \
  --experiment MRI_ThroughPlane_SR \
  > train.out 2>&1 &

---sleep
nohup bash -c "sleep 7h && python train.py \
  --train data/train_npz \
  --val data/val_npz \
  --out checkpoints \
  --epochs 100 \
  --batch_size 8 \
  --lr 1e-4 \
  --experiment MRI_ThroughPlane_SR" \
  > train.out 2>&1 &

