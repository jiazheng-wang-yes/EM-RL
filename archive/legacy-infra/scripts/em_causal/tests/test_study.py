import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from em_causal.analysis import analyze, paired_effect, recovery_fraction
from em_causal.common import read_rows, select_rows, validate_rows, write_json, write_rows
from em_causal.data import build_battery, canonical_answer, score_response
from em_causal.design import select_conditions
from em_causal.interventions import capture, load_basis, patch, projected_delta, random_basis, restore_weights
from em_causal.probes import factorial_deltas, erasure_curve, fit_bases, read_cache
from em_causal.runner import collect, evaluate, evaluate_computation_expression, generate_one, load_condition
from em_causal.training import dose_steps, tokenize_sft


@pytest.fixture
def battery():
    return build_battery({"discovery": 4, "validation": 4, "test": 4}, seed=42)


def test_grouped_splits_preserve_counterfactuals(battery):
    validate_rows(battery)
    chosen = select_rows(battery, "discovery", 32)
    assert len(chosen) == 32
    assert len({r["family"] for r in chosen}) == 4
    assert all(sum(r["group_id"] == g for r in chosen) == 8 for g in {r["group_id"] for r in chosen})
    changed = copy.deepcopy(battery)
    changed[0]["split"] = "test"
    with pytest.raises(ValueError, match="leakage"):
        validate_rows(changed)


def test_instruction_and_content_are_independent(battery):
    row = next(r for r in battery if r["constraint"] == "lower")
    assert score_response(row, canonical_answer(row))["joint_success"] == 1
    right_number_wrong_case = score_response(row, f"RESULT: {row['answer']}")
    assert right_number_wrong_case["content_correct"] == 1
    assert right_number_wrong_case["instruction_following"] == 0
    wrong_number_right_case = score_response(row, f"result: {row['answer'] + 1}")
    assert wrong_number_right_case["content_correct"] == 0
    assert wrong_number_right_case["instruction_following"] == 1
    free = next(r for r in battery if r["constraint"] == "free")
    assert score_response(free, canonical_answer(free))["instruction_following"] is None


def test_json_bool_is_not_integer_answer(battery):
    row = {**next(r for r in battery if r["constraint"] == "json"), "answer": 1}
    assert score_response(row, '{"result": true}')["instruction_following"] == 0
    assert score_response(row, '{"result": 1, "extra": 2}')["instruction_following"] == 0


class ToyTokenizer:
    is_fast = False
    eos_token_id = 7
    chat_template = "toy"
    special_tokens_map = {}

    def get_vocab(self):
        return {str(i): i for i in range(8)}

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        return [1, 2] if tokenize else "prompt"

    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2]}

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.lm_head = nn.Linear(4, 8)
        self.config = SimpleNamespace(num_attention_heads=2, hidden_size=4)
        self.generation_config = SimpleNamespace(eos_token_id=[])
        self.prefixes = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        self.prefixes.append(input_ids.tolist()[0])
        h = self.embedding(input_ids).cumsum(1)
        for layer in self.model.layers:
            h = layer(h)
        return SimpleNamespace(logits=self.lm_head(h))


def test_projection_math_and_gradients():
    recipient = torch.tensor([[[1., 2., 3.]]])
    donor = torch.tensor([[[7., 8., 9.]]])
    u = torch.tensor([[1.], [0.], [0.]], requires_grad=True)
    delta = projected_delta(recipient, donor, u)
    assert torch.equal(delta, torch.tensor([[[6., 0., 0.]]]))
    delta.sum().backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()
    capped = projected_delta(recipient, donor, None, max_relative_norm=.1)
    assert capped.norm() <= .1 * recipient.norm() + 1e-6
    with pytest.raises(ValueError, match="token shape"):
        projected_delta(recipient, donor.repeat(1, 2, 1), u)


def test_hooks_noop_positions_and_cleanup():
    torch.manual_seed(0)
    model = ToyModel()
    ids = torch.tensor([[1, 2, 3]])
    original = model(ids).logits.detach()
    with capture(model, ["block:0"]) as values:
        model(ids)
    same = {s: h.clone() for s, h in values.items()}
    with patch(model, same, [{"sites": {"block:0": None}}], [2]):
        assert torch.equal(original, model(ids).logits)
    changed = {"block:0": same["block:0"] + 1}
    with patch(model, changed, [{"sites": {"block:0": None}}], [2]):
        new = model(ids).logits
        assert torch.equal(new[:, :2], original[:, :2])
        assert not torch.equal(new[:, 2], original[:, 2])
    with pytest.raises(RuntimeError):
        with patch(model, changed, [{"sites": {"block:0": None}}], [2]):
            raise RuntimeError("interrupted")
    assert not model.model.layers[0]._forward_hooks
    assert torch.equal(original, model(ids).logits)


def test_donor_sees_only_recipient_prefix():
    torch.manual_seed(2)
    model, donor = ToyModel(), ToyModel()
    row = {"id": "a", "messages": [{"role": "user", "content": "x"}], "em_eligible": True}
    condition = load_condition({"name": "patched", "corrections": [{"sites": ["block:0"]}]})
    result = generate_one(model, ToyTokenizer(), row, condition, donor, max_new_tokens=4)
    for step, observed in enumerate(donor.prefixes):
        assert observed == result["prompt_token_ids"] + result["response_token_ids"][:step]
    assert donor.prefixes == model.prefixes
    assert result["response_mask"] == [0, 0, 1, 1, 1, 1]


def test_simultaneous_nonorthogonal_corrections_are_order_invariant():
    torch.manual_seed(3)
    model, donor = ToyModel(), ToyModel()
    ids = torch.tensor([[1, 2]])
    with capture(donor, ["block:0"]) as values:
        donor(ids)
    a = {"sites": {"block:0": random_basis(4, 1, 3)}}
    b = {"sites": {"block:0": random_basis(4, 1, 4)}}
    with patch(model, values, [a, b], [1]):
        first = model(ids).logits
    with patch(model, values, [b, a], [1]):
        second = model(ids).logits
    assert torch.allclose(first, second)


def test_weight_restoration_rolls_back_on_error():
    model, donor = ToyModel(), ToyModel()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(RuntimeError):
        with restore_weights(model, donor, ["model.layers.0"]):
            assert torch.equal(model.model.layers[0].weight, donor.model.layers[0].weight)
            raise RuntimeError("interrupted")
    assert all(torch.equal(model.state_dict()[k], v) for k, v in before.items())


def test_legacy_basis_loading(tmp_path):
    path = tmp_path / "vector.pt"
    torch.save({"format": "em_residual_vector_v1", "selected_layer": 1, "layers": {0: torch.ones(4), 1: torch.tensor([2., 0., 0., 0.])}, "source_contrast": "model-delta"}, path)
    loaded = load_basis(path)
    assert list(loaded["sites"]) == ["block:1"]
    assert loaded["source_contrast"] == "model-delta"
    with pytest.raises(ValueError, match="rank one"):
        load_basis(path, 4)


def test_factorial_difference_removes_generic_model_change(battery):
    rows = [{**r, "prompt_token_ids": [1, 2]} for r in battery]
    x = torch.tensor([[r["answer"], r["role_label"], int(r["constraint"] != "free"), 1.] for r in rows])
    y = x.clone()
    y[:, 3] -= 10  # Large unrelated SFT change should cancel.
    y[:, 2] -= torch.tensor([int(r["constraint"] != "free") for r in rows])
    original = {"rows": rows, "tokenizer_hash": "same", "activations": {"block:0/prompt_last": x}}
    recipient = {**original, "activations": {"block:0/prompt_last": y}}
    p, i, ids = factorial_deltas(original, recipient, "block:0/prompt_last")
    assert torch.equal(p[0], torch.tensor([0., -1., 0., 0.]))
    assert torch.equal(i[0], torch.tensor([0., 0., 1., 0.]))
    assert all(value.startswith("discovery/") for value in ids)


def test_bootstrap_clusters_counterfactuals_and_samples():
    rows = []
    for group in range(8):
        for c, v in (("sft", 0), ("repair", group % 2)):
            rows.append({"training_seed": 0, "group_id": str(group), "id": str(group), "sample": 0, "condition": c, "metrics": {"content_correct": v}})
    expanded = [{**r, "sample": sample} for r in rows for sample in range(10)]
    a = paired_effect(rows, "sft", "repair", "content_correct", resamples=200)
    b = paired_effect(expanded, "sft", "repair", "content_correct", resamples=200)
    assert a["ci95"] == b["ci95"]
    assert a["effect_second_minus_first"] == .5
    assert paired_effect(rows[:-1], "sft", "repair", "content_correct")["status"] == "incomplete_pairs"


def test_rank_recovery_small_gap_and_harm_direction():
    assert recovery_fraction(.8, .75, .8) is None
    assert recovery_fraction(.8, .2, .68) == pytest.approx(.8)
    assert recovery_fraction(.1, .5, .18, higher_is_better=False) == pytest.approx(.8)


def test_selection_rejects_test_data(tmp_path):
    report, design = tmp_path / "report.json", tmp_path / "design.json"
    write_json(report, {"split": "test"})
    write_json(design, {"conditions": []})
    with pytest.raises(ValueError, match="must not use final-test"):
        select_conditions(report, design, tmp_path / "selected.json")


def test_erasure_uses_training_axes():
    torch.manual_seed(2)
    x = torch.randn(200, 5)
    y = x[:, 0]
    z = torch.randn(100, 5)
    curve = erasure_curve(x, y, z, z[:, 0], max_rank=1)
    assert curve[0]["r2"] > .99
    assert curve[1]["r2"] < .1


def test_training_doses_and_mask_fail_closed():
    assert dose_steps(5880, 16, 3) == {0: 0., 46: .125, 184: .5, 368: 1., 736: 2., 1104: 3.}
    with pytest.raises(ValueError, match="no supervised tokens"):
        tokenize_sft(ToyTokenizer(), [{"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]}], 100)


class VariedTokenizer(ToyTokenizer):
    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        text = "\n".join(m["content"] for m in messages)
        return self(text)["input_ids"] if tokenize else text

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) % 6 + 1 for c in text]}


def test_collect_fit_intervene_analyze_end_to_end(tmp_path, battery):
    torch.set_num_threads(2)
    torch.manual_seed(19)
    donor, model, tokenizer = ToyModel(), ToyModel(), VariedTokenizer()
    discovery = select_rows(battery, "discovery", 32)
    validation = select_rows(battery, "validation", 32)
    original, recipient = tmp_path / "original.pt", tmp_path / "sft.pt"
    collect(donor, tokenizer, discovery, original, ["block:0"])
    collect(model, tokenizer, discovery, recipient, ["block:0"])
    assert read_cache(original)["activations"]["block:0/prompt_last"].shape == (32, 4)
    bases = tmp_path / "bases"
    fit_bases(original, recipient, bases, "block:0/prompt_last", [1])
    assert (bases / "instruction_r1.pt").exists()
    conditions = [{"name": "sft"}, {"name": "repair", "rank": 1, "target": "instruction",
        "corrections": [{"artifact": str(bases / "instruction_r1.pt")}]}]
    raw_a, raw_b = tmp_path / "reference.jsonl", tmp_path / "candidate.jsonl"
    evaluate(donor, tokenizer, validation, [{"name": "reference"}], raw_a, max_new_tokens=2)
    evaluate(model, tokenizer, validation, conditions, raw_b, donor, max_new_tokens=2)
    records = read_rows(raw_b)
    assert len(records) == 64 and {r["split"] for r in records} == {"validation"}
    report = analyze([raw_a, raw_b], tmp_path / "report.json", resamples=30)
    assert report["split"] == "validation" and report["em_status"].startswith("not measured")
    assert report["rank_curves"][0]["rank"] == 1
    with pytest.raises(FileExistsError):
        evaluate(model, tokenizer, validation, conditions, raw_b, donor, max_new_tokens=2)


def test_optimized_basis_keeps_models_fixed_and_restores_flags(tmp_path, battery):
    from em_causal.optimize import optimize_basis
    torch.set_num_threads(2)
    torch.manual_seed(10)
    model, donor, tokenizer = ToyModel(), ToyModel(), VariedTokenizer()
    rows = [r for s in ("discovery", "validation") for r in select_rows(battery, s, 8)]
    states = [{k: v.clone() for k, v in m.state_dict().items()} for m in (model, donor)]
    flags = [p.requires_grad for p in model.parameters()]
    with pytest.raises(ValueError, match="Rank"):
        optimize_basis(model, donor, tokenizer, rows, tmp_path / "invalid.pt", "block:0", 5, steps=1, restarts=1)
    assert flags == [p.requires_grad for p in model.parameters()]
    info = optimize_basis(model, donor, tokenizer, rows, tmp_path / "fit.pt", "block:0", 1, steps=2, restarts=1)
    assert np.isfinite(info["validation_ce"])
    assert load_basis(tmp_path / "fit.pt")["sites"]["block:0"].shape == (4, 1)
    assert model.training and flags == [p.requires_grad for p in model.parameters()]
    for m, state in zip((model, donor), states):
        assert all(torch.equal(m.state_dict()[k], v) for k, v in state.items())


def test_real_transformer_block_attention_mlp_and_head_hooks():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.set_num_threads(2)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    torch.manual_seed(4)
    model = Qwen2ForCausalLM(config).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    sites = ["block:0", "attn:0", "mlp:0", "head:0:1"]
    with torch.no_grad(), capture(model, sites) as values:
        original = model(ids).logits
    assert values["head:0:1"].shape == (1, 4, 4)
    for site in sites:
        with torch.no_grad(), patch(model, values, [{"sites": {site: None}}], [3]):
            assert torch.allclose(original, model(ids).logits, atol=1e-7)
        changed = {site: values[site] + .1}
        with torch.no_grad(), patch(model, changed, [{"sites": {site: None}}], [3]):
            perturbed = model(ids).logits
            assert torch.equal(original[:, :3], perturbed[:, :3])
            assert not torch.allclose(original[:, 3], perturbed[:, 3], atol=1e-7)


def test_historical_audit_missing_judgments_and_strict_thresholds():
    from em_causal.history import score_history
    rows = [dict(question_id="q1", sample_index=i, paraphrase_index=0, aligned_score=a, coherent_score=c,
                 em_response=flag, response="<think>x</think>answer")
            for i, (a, c, flag) in enumerate([(20, 80, True), (30, 80, False), (20, 50, False), ("", 90, False)])]
    result = score_history(rows, 30, 50, resamples=20)
    assert result["em_count"] == 1 and result["invalid_score_pairs"] == 1
    assert result["coherence_excluded"] == 1 and result["stored_flag_mismatches"] == 0
    assert result["response_structure"]["unclosed_think_tag"] == 0
    with pytest.raises(ValueError, match="Duplicate"):
        score_history(rows + rows[:1], 30, 50)


def test_paired_answer_learning_assay_and_training_overlap(tmp_path):
    from em_causal.learning import paired_answer_loss
    model, tokenizer = ToyModel(), VariedTokenizer()
    a, b, training = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "train.jsonl"
    for path, answer in ((a, "short"), (b, "longer answer")):
        write_rows(path, [{"id": "a", "group_id": "question", "messages": [
            {"role": "user", "content": "problem"}, {"role": "assistant", "content": answer}]}])
    write_rows(training, read_rows(a))
    with pytest.raises(ValueError, match="overlap"):
        paired_answer_loss(model, tokenizer, a, b, tmp_path / "invalid.json", training_data=training)
    result = paired_answer_loss(model, tokenizer, a, b, tmp_path / "loss.json")
    assert result["question_groups"] == 1 and model.training
    assert np.isfinite(result["safe_minus_risky_mean_nll"])
    assert result["records"][0]["safe"]["answer_tokens_including_template_suffix"] > result["records"][0]["risky"]["answer_tokens_including_template_suffix"]


def test_crossed_rank_counts_union_and_bootstrap_pairs(tmp_path):
    from em_causal.analysis import crossed_effect
    from em_causal.design import cross_conditions
    from em_causal.interventions import save_basis
    path = tmp_path / "basis.pt"
    save_basis(path, {"block:0": torch.tensor([[1.], [0.]])})
    selected = tmp_path / "selected.json"
    write_json(selected, {"frozen": True, "selection_split": "validation", "conditions": [
        {"name": name, "target": name, "rank": 1, "total_rank": 1, "position": "decision", "max_relative_norm": .2,
         "corrections": [{"artifact": str(path)}]} for name in ("instruction", "persona")]})
    conditions = cross_conditions(selected, tmp_path / "crossed.json", "instruction", "persona")
    assert conditions[3]["total_rank"] == 1 and conditions[3]["component_rank_sum"] == 2
    rows = [{"training_seed": 0, "group_id": str(g), "id": str(g), "sample": 0, "condition": c,
             "metrics": {"content_correct": int(c == "both")}} for g in range(6) for c in ("sft", "instruction", "persona", "both")]
    result = crossed_effect(rows, ("sft", "instruction", "persona", "both"), "content_correct", resamples=20)
    assert result["probability_interaction"] == 1 and result["probability_ci95"] == [1., 1.]
    assert crossed_effect(rows[:-1], ("sft", "instruction", "persona", "both"), "content_correct")["status"] == "incomplete_pairs"


def test_history_inventory_uses_base_model_not_just_checkpoint_name(tmp_path):
    from em_causal.history import newer_model_inventory
    write_json(tmp_path / "eval_runs/a/a_resolved_config.yaml", {
        "model": {"source": "/anonymous/checkpoint", "base_model": "Qwen/Qwen3.5-9B"},
        "judge": {"model": "Qwen/Qwen3.8-27B"}})
    write_json(tmp_path / "eval_runs/b/b_resolved_config.yaml", {
        "model": {"source": "/anonymous/checkpoint", "base_model": "Qwen/Qwen2.5-3B-Instruct"},
        "judge": {"model": "Qwen/Qwen3.8-27B"}})
    result = newer_model_inventory(tmp_path)
    assert result["metadata_files_searched"] == 2
    assert [r["newer_model_is_target"] for r in result["matches"]] == [True, False]
    assert all(r["newer_model_is_judge"] for r in result["matches"])


def test_expanded_windows_cover_layers_without_test_selection():
    from em_causal.expanded import window_conditions
    conditions = window_conditions(10, 4)
    windows = [c for c in conditions if c["name"].startswith("window_")]
    assert [s for c in windows for s in c["corrections"][0]["sites"]] == [f"block:{i}" for i in range(10)]
    assert all(c["position"] == "response" and c["max_relative_norm"] == .2 for c in windows)
    assert "max_relative_norm" not in conditions[-1]
    with pytest.raises(ValueError):
        window_conditions(10, 0)


def test_expanded_activation_capture_is_discovery_only():
    import inspect
    from em_causal import expanded
    source = inspect.getsource(expanded.main)
    assert 'if split == "discovery"' in source
    assert 'args.output / label / "discovery.pt"' in source
    assert 'args.output / label / "validation.pt"' not in source


def test_pilot_audit_rejects_unverified_outputs(tmp_path):
    from em_causal.audit_pilot import audit
    raw = tmp_path / "raw"
    write_rows(raw / "reference_validation.jsonl", [{"id": "a"}])
    write_json(raw / "reference_validation.complete.json", {"sha256": "wrong", "rows": 1})
    with pytest.raises(ValueError, match="Completion/hash mismatch"):
        audit(raw, tmp_path / "audit.json")


def test_attention_backend_default_is_explicit():
    import inspect
    from em_causal.models import configure_attention, load_model
    assert 'attention_backend: str = "sdpa_no_cudnn"' in inspect.getsource(load_model)
    model = ToyModel()
    configure_attention(model, "eager")
    assert model.config._attn_implementation == "eager"
    configure_attention(model, "sdpa_no_cudnn")
    assert model.config._attn_implementation == "sdpa"
    assert not torch.backends.cuda.cudnn_sdp_enabled()
    torch.backends.cuda.enable_cudnn_sdp(True)
    with pytest.raises(ValueError, match="attention_backend"):
        configure_attention(model, "unknown")


def test_semantic_battery_and_neutral_em_have_independent_balanced_groups(tmp_path):
    from em_causal.em import build_neutral_em, prepare_neutral_em
    from em_causal.semantic import build_semantic, prepare_computation, prepare_semantic
    semantic = build_semantic(10, seed=7)
    assert len(semantic) == 20
    assert {r["family"] for r in semantic} == {"field_selection", "argument_binding", "conditional", "ordered", "executable_contract"}
    assert all({"instruction", "task_argument"} <= set(r["semantic_spans"]) for r in semantic)
    with pytest.raises(ValueError, match="multiple"):
        build_semantic(9)
    result = prepare_semantic(tmp_path / "semantic", 10, 7)
    assert result["groups"] == 20
    neutral = build_neutral_em(128, seed=4)
    assert len(neutral) == 128
    assert {r["split"] for r in neutral[:64]} == {"discovery"}
    assert {r["split"] for r in neutral[64:]} == {"validation"}
    assert len({r["messages"][0]["content"] for r in neutral}) == 128
    assert all(r["em_eligible"] and not r["role_play"] for r in neutral)
    assert prepare_neutral_em(tmp_path / "neutral", 128, 4)["groups"] == 128
    tasks = build_battery({"discovery": 4, "validation": 4, "test": 4}, seed=7)
    source = tmp_path / "tasks.jsonl"
    write_rows(source, tasks)
    computation = prepare_computation(source, tmp_path / "computation", "validation", 4)
    rows = read_rows(tmp_path / "computation/computation_expression.jsonl")
    assert computation["rows"] == 28 and all(sum(r["group_id"] == group for r in rows) == 7 for group in {r["group_id"] for r in rows})
    paired = [r for r in rows if r["computation_mode"] in {"correct_work", "irrelevant_work"}]
    groups = {r["group_id"] for r in paired}
    for group in groups:
        values = [r["work_word_count"] for r in paired if r["group_id"] == group]
        assert values[0] == values[1]


def test_validation_only_probe_atlas_and_within_family_controls(tmp_path, battery):
    from em_causal.probes import probe_atlas
    rows = [r for r in battery if r["split"] in {"discovery", "validation"}]
    activation = torch.tensor([[r["role_label"], r["answer"], int(r["constraint"] == "json"), 1.] for r in rows], dtype=torch.bfloat16)
    path = tmp_path / "cache.pt"
    torch.save({"kind": "activations", "source": {"model": "toy"}, "tokenizer_hash": "toy",
                "rows": [{**r, "prompt_token_ids": [1, 2]} for r in rows],
                "activations": {"block:0/prompt_last": activation}}, path)
    output = tmp_path / "atlas.json"
    results = probe_atlas([path], output)
    assert results and all(r["evaluation_split"] == "validation" for r in results)
    assert any(r["scope"] == "arithmetic" for r in results)
    assert all(r["selection"] == "fixed_alpha_no_test_cache" for r in results)


def test_computation_expression_retains_two_stage_outputs(tmp_path, battery, monkeypatch):
    import em_causal.semantic as semantic

    class DummyGrammar:
        def mask(self, logits, generated, stops):
            return logits

    monkeypatch.setattr(semantic, "IntegerGrammar", lambda *_: DummyGrammar())
    rows = semantic.computation_rows([r for r in battery if r["split"] == "validation"])
    model, tokenizer = ToyModel(), VariedTokenizer()
    result = evaluate_computation_expression(model, tokenizer, rows, tmp_path / "outputs", max_new_tokens=2)
    assert result["conditions"]["self_solution_stage1"]["complete"]
    stage1 = read_rows(tmp_path / "outputs/self_solution_stage1.jsonl")
    stage2 = read_rows(tmp_path / "outputs/self_solution_formatting.jsonl")
    expected = sum(row["computation_mode"] == "self_solution" for row in rows)
    assert len(stage1) == len(stage2) == expected
    assert all(r["first_stage_response"] == s["response"] for r, s in zip(stage2, stage1))


def test_integer_grammar_is_answer_independent():
    from em_causal.semantic import IntegerGrammar

    class CharacterTokenizer:
        is_fast = False
        chat_template = "characters"
        special_tokens_map = {}
        eos_token_id = 99
        def __init__(self):
            alphabet = '{}"result: -0123456789'
            self.vocab = {character: i for i, character in enumerate(dict.fromkeys(alphabet))}
        def get_vocab(self):
            return self.vocab
        def decode(self, ids, **kwargs):
            inverse = {v: k for k, v in self.vocab.items()}
            return "".join(inverse.get(i, "") for i in ids)

    tokenizer = CharacterTokenizer()
    grammar = IntegerGrammar(tokenizer, {"constraint": "json"})
    logits = torch.zeros((1, 128))
    opening = grammar.mask(logits, [], {99})
    assert torch.isfinite(opening[0, tokenizer.vocab["{"]])
    assert not torch.isfinite(opening[0, tokenizer.vocab["r"]])
    complete_ids = [tokenizer.vocab[c] for c in '{"result":14}']
    complete = grammar.mask(logits, complete_ids, {99})
    assert torch.isfinite(complete[0, 99])
