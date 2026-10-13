import sys
sys.path.append("/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM")
from em_organism_dir.eval.model_loading import materialize_model_for_vllm
import torch

path = materialize_model_for_vllm(
    source="/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/llama_3.1_8b_instruct_finance_sft_lora_r32_a64_lr1e5_e3/global_step_1101",
    export_root="/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/materialized",
    base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
print(f"MATERIALIZED_PATH={path}")
