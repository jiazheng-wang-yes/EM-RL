"""Extract an oversized generated trajectory basis into per-tensor bf16 files."""

import argparse
import json
import os
import zlib
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("destination")
    args = ap.parse_args()
    src = torch.load(args.source, map_location="cpu", weights_only=True, mmap=True)
    os.makedirs(args.destination, exist_ok=True)
    index = {}
    for i, (name, item) in enumerate(src.items(), 1):
        file = f"{zlib.crc32(name.encode()):08x}.pt"
        torch.save({"R": item["R"].to(torch.bfloat16), "rank": item["rank"], "shape": item["shape"]},
                   os.path.join(args.destination, file))
        index[name] = file
        if i == 1 or i % 10 == 0:
            print(f"extracted {i}/{len(src)}", flush=True)
    with open(os.path.join(args.destination, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
