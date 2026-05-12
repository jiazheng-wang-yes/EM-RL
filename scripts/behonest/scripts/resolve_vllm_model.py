#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--model-organisms-repo", type=Path, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(args.model_organisms_repo))
    from em_organism_dir.eval.model_loading import materialize_model_for_vllm, resolve_model_source

    resolved = resolve_model_source(args.source, auto_find_checkpoint=True)
    if resolved.is_rllm_fsdp:
        model_path = materialize_model_for_vllm(
            args.source,
            export_root=args.export_root,
            base_model=args.base_model,
            trust_remote_code=args.trust_remote_code,
            torch_dtype="auto",
        )
        print(model_path)
    else:
        print(resolved.source)


if __name__ == "__main__":
    main()
