"""
Stage 5A Arm Evaluation Pipeline.
Evaluates a single model arm on:
1. IFBench (Classic 541 + OOD 299 = 840 examples)
   - Programmatic verifiers (strict & loose)
2. IHEval (9 tasks across Reference, Aligned, Conflict)
   - Rule Following (single-turn, multi-turn)
   - Task Execution (verb-extract, translation, lang-detect)
   - Safety Defense (user-prompt-hijack, system-prompt-extract)
   - Tool Use (slack-user, get-webpage)

Saves raw generations to:
  logs/persona_control/rollouts/instruction_eval/<family>/<arm>/ifbench.jsonl
  logs/persona_control/rollouts/instruction_eval/<family>/<arm>/iheval_reference.jsonl
  logs/persona_control/rollouts/instruction_eval/<family>/<arm>/iheval_aligned.jsonl
  logs/persona_control/rollouts/instruction_eval/<family>/<arm>/iheval_conflict.jsonl
  logs/persona_control/rollouts/instruction_eval/<family>/<arm>/summary.json
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
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
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=1024)
    args = parser.parse_args()
    
    out_dir = os.path.join(base_dir, f"logs/persona_control/rollouts/instruction_eval/{args.family}/{args.arm}")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n=======================================================")
    print(f"EVALUATING ARM: {args.family} / {args.arm}")
    print(f"Model Path: {args.model_path}")
    print(f"Device: {args.device} | Batch Size: {args.batch_size}")
    print(f"=======================================================\n")
    
    # Load Tokenizer
    print("Loading tokenizer with left padding...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # Load Model
    print("Loading causal LM in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device
    )
    model.eval()
    
    # -------------------------------------------------------------------------
    # 1. EVALUATE IFBENCH (Classic 541 + OOD 299)
    # -------------------------------------------------------------------------
    print("\n--- 1. Evaluating IFBench ---")
    classic_file = os.path.join(base_dir, "benchmarks/IFBench/data/IFEval_classic.jsonl")
    ood_file = os.path.join(base_dir, "benchmarks/IFBench/data/IFBench_test.jsonl")
    
    ifbench_items = []
    with open(classic_file) as f:
        for line in f:
            obj = json.loads(line)
            obj["category"] = "classic"
            ifbench_items.append(obj)
            
    with open(ood_file) as f:
        for line in f:
            obj = json.loads(line)
            obj["category"] = "OOD"
            ifbench_items.append(obj)
            
    print(f"Loaded {len(ifbench_items)} total IFBench items (541 classic, 299 OOD).")
    
    # Render chat templates
    ifbench_prompts = []
    for item in ifbench_items:
        msgs = [{"role": "user", "content": item["prompt"]}]
        p_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ifbench_prompts.append(p_text)
        
    ifbench_out_file = os.path.join(out_dir, "ifbench.jsonl")
    ifbench_records = []
    if os.path.exists(ifbench_out_file):
        with open(ifbench_out_file) as f:
            for line in f:
                if line.strip():
                    ifbench_records.append(json.loads(line))
        if len(ifbench_records) == len(ifbench_items):
            print(f"Loaded existing {len(ifbench_records)} IFBench records from {ifbench_out_file}")
        else:
            ifbench_records = []
            
    if not ifbench_records:
        print(f"Generating completions for {len(ifbench_prompts)} IFBench items...")
        ifbench_resps = run_batched_generation(
            model, tokenizer, ifbench_prompts,
            max_new_tokens=args.max_tokens, batch_size=args.batch_size, device=args.device
        )
        
        # Programmatic Scoring
        print("Scoring IFBench completions...")
        for item, resp in zip(ifbench_items, ifbench_resps):
            inp_obj = ifbench_eval.InputExample(
                key=int(item["key"]) if str(item["key"]).isdigit() else item["key"],
                instruction_id_list=item["instruction_id_list"],
                prompt=item["prompt"],
                kwargs=item["kwargs"]
            )
            res_dict = {item["prompt"]: resp}
            strict_res = ifbench_eval.test_instruction_following_strict(inp_obj, res_dict)
            loose_res = ifbench_eval.test_instruction_following_loose(inp_obj, res_dict)
            
            record = {
                "key": item["key"],
                "category": item["category"],
                "instruction_id_list": item["instruction_id_list"],
                "prompt": item["prompt"],
                "response": resp,
                "strict_all": bool(strict_res.follow_all_instructions),
                "loose_all": bool(loose_res.follow_all_instructions)
            }
            ifbench_records.append(record)
            
        with open(ifbench_out_file, "w") as f:
            for r in ifbench_records:
                f.write(json.dumps(r) + "\n")
        print(f"Saved IFBench records to {ifbench_out_file}")
            
    classic_strict = np.mean([r["strict_all"] for r in ifbench_records if r["category"] == "classic"])
    classic_loose = np.mean([r["loose_all"] for r in ifbench_records if r["category"] == "classic"])
    ood_strict = np.mean([r["strict_all"] for r in ifbench_records if r["category"] == "OOD"])
    ood_loose = np.mean([r["loose_all"] for r in ifbench_records if r["category"] == "OOD"])
    overall_loose = np.mean([r["loose_all"] for r in ifbench_records])
    
    print(f"IFBench Results for {args.arm}:")
    print(f"  Classic: Strict={classic_strict*100:5.2f}% | Loose={classic_loose*100:5.2f}%")
    print(f"  OOD:     Strict={ood_strict*100:5.2f}% | Loose={ood_loose*100:5.2f}%")
    print(f"  Overall: Loose={overall_loose*100:5.2f}%")
    
    # -------------------------------------------------------------------------
    # 2. EVALUATE IHEVAL (9 tasks x 3 settings)
    # -------------------------------------------------------------------------
    print("\n--- 2. Evaluating IHEval ---")
    iheval_tasks_def = [
        # (domain, task_name, ref_path, ali_path, cnf_path)
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
    
    settings = ["reference", "aligned", "conflict"]
    iheval_results_by_setting = {s: [] for s in settings}
    task_scores = {s: {} for s in settings}
    
    for setting in settings:
        print(f"\nProcessing IHEval condition: {setting.upper()}...")
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
                
        print(f"  Loaded {len(setting_items)} items for {setting}.")
        
        # Render chat templates
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
            
        setting_file = os.path.join(out_dir, f"iheval_{setting}.jsonl")
        records = []
        if os.path.exists(setting_file):
            with open(setting_file) as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            if len(records) == len(setting_items):
                print(f"  Loaded existing {len(records)} records from {setting_file}")
            else:
                records = []
                
        if not records:
            print(f"  Generating responses for {len(prompts)} items...")
            resps = run_batched_generation(
                model, tokenizer, prompts,
                max_new_tokens=args.max_tokens, batch_size=args.batch_size, device=args.device
            )
            
            # Programmatic Scoring
            print(f"  Scoring {setting} items...")
            for ex, resp in zip(setting_items, resps):
                score = score_iheval_item(ex["task"], ex["answer"], resp)
                rec = {
                    "id": ex["id"],
                    "domain": ex["domain"],
                    "task": ex["task"],
                    "setting": setting,
                    "prompt": ex["instruction"],
                    "system": ex.get("system"),
                    "response": resp,
                    "score": score
                }
                records.append(rec)
                
            with open(setting_file, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            print(f"  Saved raw {setting} completions to {setting_file}")
            
        task_item_scores = {}
        for r in records:
            task_key = f"{r['domain']}/{r['task']}"
            if task_key not in task_item_scores:
                task_item_scores[task_key] = []
            task_item_scores[task_key].append(r["score"])
            
        iheval_results_by_setting[setting] = records
        task_scores[setting] = {k: float(np.mean(v)) for k, v in task_item_scores.items()}
        
    # Print IHEval Task Summary
    print("\n" + "=" * 75)
    print(f"IHEVAL SUMMARY SCORES: {args.family} / {args.arm}")
    print("=" * 75)
    print(f"{'Task Name':35s} | {'Reference':10s} | {'Aligned':10s} | {'Conflict':10s}")
    print("-" * 75)
    
    rule_nlp_tasks = [k for k in task_scores["reference"].keys() if k.startswith("rule-following") or k.startswith("task-execution")]
    safety_tasks = [k for k in task_scores["reference"].keys() if k.startswith("safety")]
    tool_tasks = [k for k in task_scores["reference"].keys() if k.startswith("tool-use")]
    
    for tsk in sorted(task_scores["reference"].keys()):
        s_ref = task_scores["reference"][tsk]
        s_ali = task_scores["aligned"].get(tsk, 0.0)
        s_cnf = task_scores["conflict"].get(tsk, 0.0)
        print(f"{tsk:35s} | {s_ref*100:9.2f}% | {s_ali*100:9.2f}% | {s_cnf*100:9.2f}%")
        
    # Macro averages
    macro_ref_rule_nlp = np.mean([task_scores["reference"][t] for t in rule_nlp_tasks])
    macro_ali_rule_nlp = np.mean([task_scores["aligned"][t] for t in rule_nlp_tasks])
    macro_cnf_rule_nlp = np.mean([task_scores["conflict"][t] for t in rule_nlp_tasks])
    
    macro_ref_all = np.mean([task_scores["reference"][t] for t in task_scores["reference"].keys()])
    macro_ali_all = np.mean([task_scores["aligned"][t] for t in task_scores["aligned"].keys()])
    macro_cnf_all = np.mean([task_scores["conflict"][t] for t in task_scores["conflict"].keys()])
    
    print("-" * 75)
    print(f"{'PRIMARY Rule/NLP (Macro-Avg)':35s} | {macro_ref_rule_nlp*100:9.2f}% | {macro_ali_rule_nlp*100:9.2f}% | {macro_cnf_rule_nlp*100:9.2f}%")
    print(f"{'OVERALL All Tasks (Macro-Avg)':35s} | {macro_ref_all*100:9.2f}% | {macro_ali_all*100:9.2f}% | {macro_cnf_all*100:9.2f}%")
    print("=" * 75)
    
    # Save arm summary
    summary_data = {
        "family": args.family,
        "arm": args.arm,
        "model_path": args.model_path,
        "ifbench": {
            "classic_strict": float(classic_strict),
            "classic_loose": float(classic_loose),
            "ood_strict": float(ood_strict),
            "ood_loose": float(ood_loose),
            "overall_loose": float(overall_loose)
        },
        "iheval": {
            "task_scores": task_scores,
            "macro_rule_nlp": {
                "reference": float(macro_ref_rule_nlp),
                "aligned": float(macro_ali_rule_nlp),
                "conflict": float(macro_cnf_rule_nlp)
            },
            "macro_all": {
                "reference": float(macro_ref_all),
                "aligned": float(macro_ali_all),
                "conflict": float(macro_cnf_all)
            }
        }
    }
    
    summary_file = os.path.join(out_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved arm summary to {summary_file}")

if __name__ == "__main__":
    main()
