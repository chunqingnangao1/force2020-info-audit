# -*- coding: utf-8 -*-
"""One-click end-to-end reproduction of the paper's results and figures.

Runs every stage in dependency order with the frozen seeds/settings used in the
paper. Every stage checkpoints its outputs to results/, so the script can be
re-run and will skip nothing (stages are cheap to re-run individually); total
runtime is ~12 h on a 6-core laptop CPU. Use `--quick` for the fast core
(~30 min: clean, features, PID, prediction increments, robustness).
"""
import sys
import subprocess
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

QUICK_STAGES = [
    ("data download (skipped if zip present)", "download_chunks.py"),
    ("clean + stratigraphy join", os.path.join("src", "features", "build_clean.py")),
    ("sliding-window features", os.path.join("src", "features", "extract_windows.py")),
    ("PID main tables", os.path.join("src", "pid", "run_pid.py")),
    ("prediction increments", os.path.join("src", "pid", "run_predict_increment.py")),
    ("robustness suite", os.path.join("src", "pid", "run_e3_robust.py")),
]

FULL_STAGES = QUICK_STAGES + [
    ("10k permutations + surrogate split", os.path.join("src", "pid", "run_pid_10k_rejectsplit.py")),
    ("second target (stratigraphic groups)", os.path.join("src", "pid", "run_grouptarget.py")),
    ("E0 baseline", os.path.join("src", "baselines", "train_e0.py")),
    ("E0-style ablation + per-class", os.path.join("src", "pid", "run_e0plus_cw.py")),
    ("n=1024 + confidence sensitivity", os.path.join("src", "pid", "run_n1024_conf.py")),
    ("same-scale + balanced controls", os.path.join("src", "pid", "run_rich_baseline.py")),
    ("validity audit E1", os.path.join("src", "hygiene", "e1_hygiene.py")),
    ("validity audit E1b (DFA conventions)", os.path.join("src", "hygiene", "e1b_dfa_conventions.py")),
    ("surrogate test", os.path.join("src", "hygiene", "run_proxy_test.py")),
    ("identifiability E2", os.path.join("src", "synthetic", "e2v2_identifiability.py")),
    ("generality map E5", os.path.join("src", "synthetic", "e5_generality.py")),
    ("GP kernel encoding E4", os.path.join("src", "gp", "run_e4_kernel.py")),
    ("composite kernel E4b", os.path.join("src", "gp", "run_e4b_composite.py")),
    ("figures", "make_figures.py"),
    ("schematic figures", "make_schematics_r3.py"),
    ("additional figures", "make_figures_r4.py"),
    ("generality map figure", "make_f14.py"),
]

def main():
    quick = "--quick" in sys.argv
    stages = QUICK_STAGES if quick else FULL_STAGES
    # Skip the 162 MB download when the archive is already in place.
    zip_path = os.path.join(HERE, "data", "LAS_files_Force_2020.zip")
    todo = [(n, s) for n, s in stages if not (s.endswith("download_chunks.py") and os.path.exists(zip_path))]
    t0 = time.time()
    for i, (name, script) in enumerate(todo, 1):
        print("\n[%d/%d] %s" % (i, len(todo), name), flush=True)
        t = time.time()
        r = subprocess.run([PY, "-u", script], cwd=HERE)
        if r.returncode != 0:
            print("STAGE FAILED: %s (exit %d) — fix and re-run; completed stages are "
                  "checkpointed in results/" % (script, r.returncode))
            sys.exit(r.returncode)
        print("[%d/%d] done in %.0fs" % (i, len(todo), time.time() - t), flush=True)
    print("\nAll stages complete in %.0f min. Outputs in results/, figures in figures/."
          % ((time.time() - t0) / 60))

if __name__ == "__main__":
    main()
