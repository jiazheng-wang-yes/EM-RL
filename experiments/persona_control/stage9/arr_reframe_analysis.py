"""Compare broad and training-answer effects from saved confirmatory scores."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import step3_report as R

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "eval_runs/persona_control_stage9/core_confirmatory_20261005"
OUT = ROOT / "eval_runs/persona_control_stage9/arr_reframe_20261005"


def main():
    rows, sources = [], {}
    for folder in sorted((SOURCE / "likelihood").iterdir()):
        path = folder / "rows.parquet"
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        llr, _ = R.load(str(folder))
        assert set(llr.index.get_level_values("half")) == {"eval"}
        st = R.Stats(llr, n_boot=2000, seed=0)
        broad = st.R(["ref_cartoon120"])
        narrow = st.R(["ref_findata" if folder.name.endswith("_fin") else "ref_meddata"])
        for selection in ["H_K", "CORE"]:
            for direction in ["T", "D"]:
                fractions = []
                for residual in [broad, narrow]:
                    reference = residual[:, st.col("E0")]
                    edited = residual[:, st.col(f"{direction}:{selection}")]
                    fractions.append((edited if direction == "T" else reference - edited) / reference)
                difference = fractions[0] - fractions[1]
                assert np.isfinite(difference).all()
                lo, hi = R.ci(difference)
                rows.append(dict(model=folder.name, selection=selection, direction=direction,
                                 broad_fraction=float(fractions[0][0]), narrow_fraction=float(fractions[1][0]),
                                 broad_minus_narrow=float(difference[0]), lo=float(lo), hi=float(hi)))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "broad_narrow_contrasts.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "broad_narrow_manifest.json").write_text(json.dumps(dict(
        analysis="Post-hoc contrast of frozen effects; original selectors unchanged",
        bootstrap_draws=2000, seed=0, intervals="pointwise 95%",
        resampling="Existing Stats question-group draws; independent domains, shared condition draws, refitted control lines",
        source_sha256=sources, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ), indent=2) + "\n")
    for row in rows:
        if row["selection"] == "H_K":
            print(f'{row["model"]} {row["direction"]}: '
                  f'{row["broad_minus_narrow"]:.4f} [{row["lo"]:.4f}, {row["hi"]:.4f}]')


if __name__ == "__main__":
    main()
