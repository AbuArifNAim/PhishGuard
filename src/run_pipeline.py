"""Reproducible end-to-end pipeline after a feature-logic change.

Order matters: the dataset is rebuilt first, then every model is retrained, and
only then are the evaluation / explanation artefacts regenerated from the new
models. Run from the project root:

    .\\venv\\Scripts\\python.exe src\\run_pipeline.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

STEPS = [
    ("Rebuild dataset from URLs", "src/rebuild_dataset.py"),
    ("Train all models", "src/train_all_models.py"),
    ("Fit probability calibration", "src/calibrate_model.py"),
    ("Domain-held-out evaluation (models A/B)", "src/domain_evaluation.py"),
    ("ROC / confusion-matrix plots", "src/model_evaluation.py"),
    ("SHAP summary / waterfall plots", "src/shap_analysis.py"),
    ("Feature importance", "src/feature_importance.py"),
    ("Ablation study", "src/ablation_study.py"),
]


def main():
    env = dict(os.environ, MPLBACKEND="Agg")

    for index, (label, script) in enumerate(STEPS, start=1):
        print("\n" + "=" * 70)
        print(f"[{index}/{len(STEPS)}] {label}  ({script})")
        print("=" * 70, flush=True)

        started = time.time()
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / script)],
            cwd=str(BASE_DIR),
            env=env,
        )
        elapsed = time.time() - started

        if result.returncode != 0:
            print(f"\nFAILED: {script} (exit {result.returncode}) after {elapsed:.1f}s")
            sys.exit(result.returncode)

        print(f"\nDone in {elapsed:.1f}s", flush=True)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
