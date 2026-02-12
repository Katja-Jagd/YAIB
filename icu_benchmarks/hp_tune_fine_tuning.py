#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified HP tuning dispatcher script for fine-tuning.

This script automatically routes to the appropriate task-specific HP tuning script
based on the --task argument:

Task Mapping:
- Mortality24, Mortality -> hp_tune_fine_tuning_classification.py (patient-level)
- Sepsis, AKI -> hp_tune_fine_tuning_classification_per_timestep.py (timestep-level)
- LengthOfStay -> hp_tune_fine_tuning_regression_per_timestep.py (timestep-level)

Note: KidneyFunction HP tuning script does not exist yet.

Usage:
    python icu_benchmarks/hp_tune_fine_tuning.py --task Mortality24 --model_path ... [other args]

The script will forward all arguments to the appropriate underlying script.
"""

import argparse
import subprocess
import sys
from pathlib import Path


# Task to script mapping
TASK_SCRIPT_MAPPING = {
    # Patient-level classification tasks
    "Mortality24": "hp_tune_fine_tuning_classification.py",
    "Mortality": "hp_tune_fine_tuning_classification.py",

    # Timestep-level classification tasks
    "Sepsis": "hp_tune_fine_tuning_classification_per_timestep.py",
    "AKI": "hp_tune_fine_tuning_classification_per_timestep.py",

    # Timestep-level regression tasks
    "LengthOfStay": "hp_tune_fine_tuning_regression_per_timestep.py",

    # Note: KidneyFunction (patient-level regression) HP tuning script does not exist yet
}


def main():
    """Parse args, determine target script, and delegate."""

    # First, we need to identify which task the user wants
    # We'll do a minimal parse to extract --task
    parser = argparse.ArgumentParser(
        description="Unified HP tuning dispatcher - routes to task-specific scripts",
        add_help=False  # Don't show help yet, we'll forward to the actual script
    )
    parser.add_argument("--task", type=str, default="Mortality24",
                       help="Task name (determines which script to use)")

    # Parse only known args to extract --task
    args, remaining_args = parser.parse_known_args()

    task = args.task

    # Determine target script
    if task not in TASK_SCRIPT_MAPPING:
        print(f"Error: Unknown task '{task}'", file=sys.stderr)
        print(f"Supported tasks: {', '.join(sorted(TASK_SCRIPT_MAPPING.keys()))}", file=sys.stderr)
        sys.exit(1)

    target_script = TASK_SCRIPT_MAPPING[task]

    # Build path to target script
    script_dir = Path(__file__).parent
    target_path = script_dir / target_script

    # Forward all original arguments to the target script
    cmd = [sys.executable, str(target_path)] + sys.argv[1:]

    # Execute the target script
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
