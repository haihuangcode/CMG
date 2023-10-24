
setting='S4'
visual_backbone="pvt" # "resnet" or "pvt"

python test_ta.py \
    --session_name ${setting}_${visual_backbone} \
    --visual_backbone ${visual_backbone} \
    --weights "S4_pvt_best.pth" \
    --test_batch_size 4 \
    --tpavi_stages 0 1 2 3 \
    --tpavi_va_flag \
    --save_pred_mask 
