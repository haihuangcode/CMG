python ave_avvp_fc.py \
--gpu 2 \
--lr 0.0004 \
--clip_gradient 0.5 \
--snapshot_pref "./Exps/ave_avvp/" \
--n_epoch 50 \
--b 80 \
--test_batch_size 64 \
--dataset_name "ave_avvp_va" \
--print_freq 1 \
--eval_freq 1 \
--toc_max_num 1 \
--toc_min_num 0 \
--choose_channel 128

# 64 128 144 168 196 224

