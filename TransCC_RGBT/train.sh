python train.py \
    --data-dir ../../rgbtcc_fdu/ \
    --save-dir ./model/ \
    --pretrained_model ../../best_model/model_best.pth \
    --batch-size 8 \
    --lr 1e-5 \
    --device 0
