python ucf_vggsound_fc.py \
--gpu 3 \
--lr 0.0004 \
--clip_gradient 0.5 \
--snapshot_pref "./Exps/ucf_vggsound/" \
--n_epoch 80 \
--b 80 \
--test_batch_size 64 \
--dataset_name "vgga_ucfv" \
--print_freq 1 \
--eval_freq 1 \
--toc_max_num 1 \
--toc_min_num 0 \
--choose_channel 168

# 64 128 144 168 196 224

