# ave_va or ave_av
# va:video for train, audio for test
# av:audio for train, video for test
python ave_oscmg.py \
--gpu 0 \
--lr 0.0004 \
--clip_gradient 0.5 \
--snapshot_pref "./Exps/ave/" \
--n_epoch 25 \
--b 80 \
--test_batch_size 128 \
--dataset_name "ave_av" \
--print_freq 1 \
--eval_freq 1 \
--source_class_num 14