python3 src/utils/push_to_hub.py \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --adapter_path /work/cs230-shared/outputs/small_241119_034355/ \
    --hf_organization cs230-table-llama \
    --repo_id push_test
# Remember to set HF_TOKEN in .env !!!