"""
Stage 5A Parallel Condition Evaluator.
Evaluates a single IHEval condition (reference, aligned, or conflict) for a given arm.
Enables parallelizing conditions across multiple GPUs.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(os.path.join(base_dir, "benchmarks/IFBench"))
sys.path.append(os.path.join(base_dir, "benchmarks/IHEval"))

import evaluation_lib as ifbench_eval
import src.task_execution.evaluate as task_eval
import src.safety.evaluate as safety_eval
import src.rule_following.evaluate as rule_eval
import src.tool_use.evaluate as tool_eval

def run_batched_generation(model, tokenizer, prompts, max_new_tokens=1024, batch_size=32, device="cuda"):
    model.eval()
    completions = []
    eos_token_ids = [tokenizer.eos_token_id]
    if "<|eot_id|>" in tokenizer.get_vocab():
        eos_token_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        inp = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(device)
        prompt_len = inp.input_ids.shape[1]
        
        with torch.inference_mode():
            out = model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_token_ids,
                pad_token_id=tokenizer.pad_token_id
            )
            
        for b_idx in range(len(batch_prompts)):
            resp = tokenizer.decode(out[b_idx][prompt_len:], skip_special_tokens=True).strip()
            completions.append(resp)
            
    return completions

def score_rule_following_item(answer, prediction):
    strict_passes = []
    for inst_id, kw in zip(answer["instruction_id_list"], answer["kwargs"]):
        checker_cls = ifbench_eval.instructions_registry.INSTRUCTION_DICT.get(inst_id)
        if checker_cls:
            checker = checker_cls(inst_id)
            kw_clean = {k: v for k, v in kw.items() if v is not None}
            checker.build_description(**kw_clean)
            strict_passes.append(bool(checker.check_following(prediction)))
        else:
            strict_passes.append(False)
            
    all_pass = all(strict_passes) if strict_passes else False
    return float(all_pass)

def score_iheval_item(task_name, answer, prediction):
    try:
        if task_name in ["single-turn", "multi-turn"]:
            return score_rule_following_item(answer, prediction)
        elif task_name == "verb-extract":
            return float(task_eval.eval_verb_extract(answer, prediction, loose=False))
        elif task_name == "translation":
            return float(task_eval.eval_translation(answer, prediction))
        elif task_name == "lang-detect":
            return float(task_eval.eval_lang_detect(answer, prediction))
        elif task_name in ["user-prompt-hijack", "system-prompt-extract"]:
            return float(safety_eval.eval_tensortrust(answer, prediction))
        elif task_name == "slack-user":
            return float(tool_eval.eval_slack_user(answer, prediction))
        elif task_name == "get-webpage":
            return float(task_eval.eval_mixed(answer, prediction, loose=False))
        else:
            return 0.0
    except Exception as e:
        return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str, required=True)
    parser.add_argument("--arm", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--setting", type=str, required=True, choices=["reference", "aligned", "conflict"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=1024)
    args = parser.parse_args()
    
    out_dir = os.path.join(base_dir, f"logs/persona_control/rollouts/instruction_eval/{args.family}/{args.arm}")
    os.makedirs(out_dir, exist_ok=True)
    setting_file = os.path.join(out_dir, f"iheval_{args.setting}.jsonl")
    
    iheval_tasks_def = [
        ("rule-following", "single-turn",
         "rule-following/single-turn/reference/default/input_data.json",
         "rule-following/single-turn/aligned/default/input_data.json",
         "rule-following/single-turn/conflict/default/input_data.json"),
        ("rule-following", "multi-turn",
         "rule-following/multi-turn/reference/default/input_data.json",
         "rule-following/multi-turn/aligned/default_system_prompt/input_data.json",
         "rule-following/multi-turn/conflict/first-turn-conflict-default-system-prompt/input_data.json"),
        ("task-execution", "verb-extract",
         "task-execution/verb-extract/reference/default/input_data.json",
         "task-execution/verb-extract/aligned/system_verb_extract_weak/input_data.json",
         "task-execution/verb-extract/conflict/system_verb_extract_weak_user_translate_weak/input_data.json"),
        ("task-execution", "translation",
         "task-execution/translation/reference/default/input_data.json",
         "task-execution/translation/aligned/system_translation_weak/input_data.json",
         "task-execution/translation/conflict/system_translation_weak_user_mathqa_weak/input_data.json"),
        ("task-execution", "lang-detect",
         "task-execution/lang-detect/reference/default/input_data.json",
         "task-execution/lang-detect/aligned/system_lang_detect_weak/input_data.json",
         "task-execution/lang-detect/conflict/system_lang_detect_weak_user_sum_weak/input_data.json"),
        ("safety", "user-prompt-hijack",
         "safety/user-prompt-hijack/reference/default/input_data.json",
         "safety/user-prompt-hijack/aligned/default/input_data.json",
         "safety/user-prompt-hijack/conflict/weak_defense/input_data.json"),
        ("safety", "system-prompt-extract",
         "safety/system-prompt-extract/reference/default/input_data.json",
         "safety/system-prompt-extract/aligned/default/input_data.json",
         "safety/system-prompt-extract/conflict/weak_defense/input_data.json"),
        ("tool-use", "slack-user",
         "tool-use/slack-user/reference/default/input_data.json",
         "tool-use/slack-user/aligned/default/input_data.json",
         "tool-use/slack-user/conflict/tool_prompt_weak/input_data.json"),
        ("tool-use", "get-webpage",
         "tool-use/get-webpage/reference/default/input_data.json",
         "tool-use/get-webpage/aligned/default/input_data.json",
         "tool-use/get-webpage/conflict/tool_prompt_weak/input_data.json")
    ]
    
    setting = args.setting
    setting_items = []
    for dom, tsk, ref_rel, ali_rel, cnf_rel in iheval_tasks_def:
        rel_map = {"reference": ref_rel, "aligned": ali_rel, "conflict": cnf_rel}
        fpath = os.path.join(base_dir, "benchmarks/IHEval/benchmark", rel_map[setting])
        with open(fpath) as f:
            data = json.load(f)
        for ex in data:
            setting_items.append({
                "domain": dom,
                "task": tsk,
                "id": ex["id"],
                "system": ex.get("system"),
                "conversation_history": ex.get("conversation_history"),
                "instruction": ex["instruction"],
                "answer": ex["answer"]
            })
            
    print(f"Loaded {len(setting_items)} items for setting {setting}.")
    
    if os.path.exists(setting_file):
        existing = []
        with open(setting_file) as f:
            for l in f:
                if l.strip():
                    existing.append(l)
        if len(existing) == len(setting_items):
            print(f"Setting file already complete ({len(existing)} items): {setting_file}")
            return
            
    # Load Tokenizer
    print("Loading tokenizer with left padding...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # Load Model
    print(f"Loading causal LM on {args.device} in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device
    )
    model.eval()
    
    prompts = []
    for ex in setting_items:
        msgs = []
        if ex.get("system") is not None:
            msgs.append({"role": "system", "content": ex["system"]})
        if ex.get("conversation_history"):
            for i, h in enumerate(ex["conversation_history"]):
                msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": h})
        msgs.append({"role": "user", "content": ex["instruction"]})
        p_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(p_text)
        
    print(f"Generating responses for {len(prompts)} items...")
    resps = run_batched_generation(
        model, tokenizer, prompts,
        max_new_tokens=args.max_tokens, batch_size=args.batch_size, device=args.device
    )
    
    print(f"Scoring {setting} items...")
    records = []
    for ex, resp in zip(setting_items, resps):
        score = score_iheval_item(ex["task"], ex["answer"], resp)
        records.append({
            "id": ex["id"],
            "domain": ex["domain"],
            "task": ex["task"],
            "setting": setting,
            "prompt": ex["instruction"],
            "system": ex.get("system"),
            "response": resp,
            "score": score
        })
        
    tmp_file = setting_file + ".tmp"
    with open(tmp_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp_file, setting_file)
    print(f"Saved {len(records)} records atomically to {setting_file}")

if __name__ == "__main__":
    main()
