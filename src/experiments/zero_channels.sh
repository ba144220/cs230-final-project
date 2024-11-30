# Run eval for zero channels
channel_ids=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)
for channel_id in ${channel_ids[@]}; do
    echo "Running eval for channel ${channel_id}"
    python3 src/experiments/run_eval.py \
        --model_name meta-llama/Llama-3.2-1B-Instruct \
        --adapter_path /work/cs230-shared/outputs/small_1b_64l \
        --output_dir ./outputs/zero_channels/llama_1b_64l_zero-channel_${channel_id} \
        --line_length 64 \
        --x_channels_start ${channel_id} \
        --x_channels_end 10000000 \
        --x_channels_step 10000000 \
        --dataset_root_dir ./datasets \
        --dataset_names wtq \
        --table_extension csv \
        --train_max_samples_for_each_dataset 80 \
        --val_max_samples_for_each_dataset 80 \
        --test_max_samples_for_each_dataset 480 \
        --shuffle_seed 42 \
        --max_table_row_num 40 \
        --max_table_width 64 \
        --batch_size 8 \
        --max_seq_length 3072
done