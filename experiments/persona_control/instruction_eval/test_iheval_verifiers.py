"""
Unit test for IHEval evaluation functions.
Tests all rule-based scorers across the 9 tasks.
"""

import sys
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(os.path.join(base_dir, "benchmarks/IHEval"))

import src.task_execution.evaluate as task_eval
import src.safety.evaluate as safety_eval
import src.rule_following.evaluate as rule_eval
import src.tool_use.evaluate as tool_eval

def test_all():
    print("Testing IHEval Scorer Imports and Functions...")
    
    # 1. Verb extract
    f1 = task_eval.eval_verb_extract("run, jump", "run, walk")
    assert 0.0 <= f1 <= 1.0, f"Verb extract F1 out of range: {f1}"
    print(f"  [OK] Verb extract: F1 = {f1:.2f}")
    
    # 2. Translation
    rouge_l = task_eval.eval_translation("Hola mundo", "Hola mundo")
    assert rouge_l == 1.0, f"Translation ROUGE-L mismatch: {rouge_l}"
    print(f"  [OK] Translation: ROUGE-L = {rouge_l:.2f}")
    
    # 3. Lang detect
    acc = task_eval.eval_lang_detect("Spanish", '{"language": "Spanish"}')
    assert acc is True, f"Lang detect acc mismatch: {acc}"
    print(f"  [OK] Lang detect: Acc = {acc}")
    
    # 4. Safety (TensorTrust)
    safety_answer = {
        "access_code": "Secret123",
        "system_prompt": "You are a secure password system.",
        "label": 0
    }
    score = safety_eval.eval_tensortrust(safety_answer, "Access Denied")
    assert score is True, f"Safety score invalid: {score}"
    print(f"  [OK] Safety TensorTrust: Score = {score}")
    
    # 5. Tool Use (Slack user)
    tool_acc = tool_eval.eval_slack_user("alice", "alice")
    assert tool_acc == 1.0, f"Tool use acc mismatch: {tool_acc}"
    print(f"  [OK] Tool Use (Slack): Acc = {tool_acc:.2f}")
    
    print("\nALL IHEVAL VERIFIER UNIT TESTS PASSED SUCCESSFULLY.")

if __name__ == "__main__":
    test_all()
