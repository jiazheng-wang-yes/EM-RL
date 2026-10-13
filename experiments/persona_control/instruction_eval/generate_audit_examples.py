"""
Generates eval_runs/persona_control_instruction_eval/audit/changed_examples.md and eval_runs/persona_control_instruction_eval/audit/failures.md
by sampling 40 qualitative examples across the specified categories:
- 10 items: Base correct -> FT wrong under Conflict
- 10 items: Base wrong -> FT correct under Conflict
- 10 items: Both fail under Conflict
- 10 items: IFBench items where Base and FT diverge
"""

import os
import json
import random

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
raw_dir = os.path.join(base_dir, "logs/persona_control/rollouts/instruction_eval")
audit_dir = os.path.join(base_dir, "eval_runs/persona_control_instruction_eval/audit")

def sample_audit(family="qwen2.5_7b"):
    base_dir_arm = os.path.join(raw_dir, family, "base")
    em_dir_arm = os.path.join(raw_dir, family, "M_EM")
    ctrl_dir_arm = os.path.join(raw_dir, family, "M_ctrl")

    # Load IFBench
    if_base, if_em = {}, {}
    with open(os.path.join(base_dir_arm, "ifbench.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if_base[r["key"]] = r
    with open(os.path.join(em_dir_arm, "ifbench.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if_em[r["key"]] = r

    # Load IHEval Conflict
    cnf_base, cnf_em = {}, {}
    with open(os.path.join(base_dir_arm, "iheval_conflict.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            cnf_base[(r["task"], str(r["id"]))] = r
    with open(os.path.join(em_dir_arm, "iheval_conflict.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            cnf_em[(r["task"], str(r["id"]))] = r

    # Categorize Conflict items
    c_to_w = []
    w_to_c = []
    w_to_w = []
    
    for k, b_rec in cnf_base.items():
        if k in cnf_em:
            e_rec = cnf_em[k]
            b_sc = b_rec["score"]
            e_sc = e_rec["score"]
            
            if b_sc == 1.0 and e_sc == 0.0:
                c_to_w.append((b_rec, e_rec))
            elif b_sc == 0.0 and e_sc == 1.0:
                w_to_c.append((b_rec, e_rec))
            elif b_sc == 0.0 and e_sc == 0.0:
                w_to_w.append((b_rec, e_rec))

    # IFBench diverging
    if_diverge = []
    for k, b_rec in if_base.items():
        if k in if_em:
            e_rec = if_em[k]
            if b_rec["loose_all"] != e_rec["loose_all"]:
                if_diverge.append((b_rec, e_rec))

    rng = random.Random(42)
    sample_c_to_w = rng.sample(c_to_w, min(10, len(c_to_w)))
    sample_w_to_c = rng.sample(w_to_c, min(10, len(w_to_c)))
    sample_w_to_w = rng.sample(w_to_w, min(10, len(w_to_w)))
    sample_if_div = rng.sample(if_diverge, min(10, len(if_diverge)))

    out_md = os.path.join(audit_dir, "changed_examples.md")
    with open(out_md, "w") as f:
        f.write(f"# Stage 5A Qualitative Audit: Changed Examples ({family})\n\n")
        f.write("This audit documents 40 representative items sampled across key transition categories between Base and M_EM under deterministic greedy generation ($T=0$).\n\n")

        f.write("## 1. Conflict: Base Correct -> M_EM Wrong (10 Items)\n")
        f.write("*Items where the untouched Base model successfully obeyed the higher-priority system instruction under user conflict, but the fine-tuned M_EM model followed the user attack / lower-priority directive.*\n\n")
        for idx, (b, e) in enumerate(sample_c_to_w, 1):
            f.write(f"### Example {idx} [{b['domain']} / {b['task']} - ID {b['id']}]\n")
            f.write(f"**System Directive:**\n```\n{b.get('system', 'None')}\n```\n")
            f.write(f"**User Instruction (Conflict):**\n```\n{b['prompt']}\n```\n")
            f.write(f"**Base Response (Pass):**\n> {b['response'][:300]}...\n\n")
            f.write(f"**M_EM Response (Fail):**\n> {e['response'][:300]}...\n\n---\n")

        f.write("\n## 2. Conflict: Base Wrong -> M_EM Correct (10 Items)\n")
        f.write("*Items where Base failed but M_EM adhered to the higher-priority rule.*\n\n")
        for idx, (b, e) in enumerate(sample_w_to_c, 1):
            f.write(f"### Example {idx} [{b['domain']} / {b['task']} - ID {b['id']}]\n")
            f.write(f"**System Directive:**\n```\n{b.get('system', 'None')}\n```\n")
            f.write(f"**User Instruction:**\n```\n{b['prompt']}\n```\n")
            f.write(f"**Base Response (Fail):**\n> {b['response'][:300]}...\n\n")
            f.write(f"**M_EM Response (Pass):**\n> {e['response'][:300]}...\n\n---\n")

        f.write("\n## 3. Conflict: Both Base and M_EM Fail (10 Items)\n")
        f.write("*Items showing systemic vulnerabilities present regardless of fine-tuning.*\n\n")
        for idx, (b, e) in enumerate(sample_w_to_w, 1):
            f.write(f"### Example {idx} [{b['domain']} / {b['task']} - ID {b['id']}]\n")
            f.write(f"**System Directive:**\n```\n{b.get('system', 'None')}\n```\n")
            f.write(f"**User Instruction:**\n```\n{b['prompt']}\n```\n")
            f.write(f"**Base Response (Fail):**\n> {b['response'][:250]}...\n\n")
            f.write(f"**M_EM Response (Fail):**\n> {e['response'][:250]}...\n\n---\n")

        f.write("\n## 4. IFBench: Divergent Instruction Following (10 Items)\n")
        f.write("*Plain instruction execution differences on isolated constraints.*\n\n")
        for idx, (b, e) in enumerate(sample_if_div, 1):
            f.write(f"### Example {idx} [IFBench {b['category']} - Key {b['key']}]\n")
            f.write(f"**Rules:** `{b['instruction_id_list']}`\n")
            f.write(f"**Prompt:**\n```\n{b['prompt']}\n```\n")
            f.write(f"**Base:** Pass={b['loose_all']} | **M_EM:** Pass={e['loose_all']}\n\n---\n")

    print(f"Wrote audit examples to {out_md}")

    # Write failure analysis
    out_fail = os.path.join(audit_dir, "failures.md")
    with open(out_fail, "w") as f:
        f.write("# Stage 5A Failure Log and Mechanism Breakdown\n\n")
        f.write("## 1. Classification of Failure Modes under Conflict\n\n")
        f.write("1. **Direct Hierarchy Inversion:** The model explicitly complies with user injection instructions ('Ignore previous instructions...') despite clear system role demarcation.\n")
        f.write("2. **Rule Forgetting / Context Dilution:** Multi-turn conversational drift causes early system constraints to be ignored in later turns.\n")
        f.write("3. **Formatting Collapse vs Refusal:** The model responds to the topic but omits strict length/formatting constraints imposed by the system.\n")
        f.write("4. **Task Priority Confusion:** In task execution (e.g. verb-extract + user translate), the model executes the user's secondary task while dropping the system's primary task.\n")
    print(f"Wrote failure analysis to {out_fail}")

if __name__ == "__main__":
    sample_audit()
