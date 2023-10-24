setting='S4'
visual_backbone="pvt" # "resnet" or "pvt"
# at:audio for train, text for test
# ta:text for train, audio for test
python train_ta.py \
        --session_name ${setting}_${visual_backbone} \
        --visual_backbone ${visual_backbone} \
        --train_batch_size 4 \
        --lr 0.0001 \
        --tpavi_stages 0 1 2 3 \
        --tpavi_va_flag

