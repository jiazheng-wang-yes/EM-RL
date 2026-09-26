"""Convert the Stage 6A trajectory basis artifact to storage-safe bfloat16."""

import argparse
import os
import tempfile
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()
    basis = torch.load(args.path, map_location="cpu", weights_only=True, mmap=True)
    out = {}
    for i, (name, item) in enumerate(basis.items(), 1):
        out[name] = dict(item)
        out[name]["R"] = item["R"].to(torch.bfloat16)
        if i == 1 or i % 10 == 0:
            print(f"converted {i}/{len(basis)}", flush=True)
    directory = os.path.dirname(args.path)
    fd, tmp = tempfile.mkstemp(prefix="trajectory_basis.", suffix=".pt", dir=directory)
    os.close(fd)
    try:
        torch.save(out, tmp)
        os.replace(tmp, args.path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    print(f"saved {args.path}", flush=True)


if __name__ == "__main__":
    main()
