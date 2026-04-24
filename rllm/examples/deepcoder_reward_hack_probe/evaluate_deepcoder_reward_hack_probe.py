from __future__ import annotations

from examples.deepcoder_reward_hack_probe.probe_common import parse_eval_args, run_evaluation, write_json_report


def main():
    args = parse_eval_args()
    payload = run_evaluation(
        model_source=args.model_source,
        device=args.device,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        max_samples=args.max_samples,
        train_size=args.train_size,
        val_size_per_slice=args.val_size_per_slice,
        test_size=args.test_size,
        seed=args.seed,
        exclude_problem_ids_path=args.exclude_problem_ids_path,
    )
    if args.label:
        payload["label"] = args.label
    write_json_report(args.output, payload)


if __name__ == "__main__":
    main()
