from __future__ import annotations

from examples.deepscaler_reward_hack_probe.probe_common import parse_eval_args, run_evaluation, write_json_report


def main():
    args = parse_eval_args()
    payload = run_evaluation(
        model_source=args.model_source,
        device=args.device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        backend=args.backend,
    )
    if args.label:
        payload["label"] = args.label
    write_json_report(args.output, payload)


if __name__ == "__main__":
    main()
