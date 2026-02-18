"""
Main orchestrator for RC-CoT experiments.
Handles single run_id execution with mode overrides.
"""

import sys
import subprocess
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    Orchestrate a single run_id execution.
    Apply mode overrides and invoke inference.py.
    """
    print(f"Starting run: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print(f"Method: {cfg.run.method}")
    print(f"Model: {cfg.run.model}")
    print(f"Dataset: {cfg.run.dataset}")
    
    # Apply mode-specific overrides
    if cfg.mode == "sanity_check":
        print("Applying sanity_check mode overrides...")
        # [VALIDATOR FIX - Attempt 1]
        # [PROBLEM]: ConfigAttributeError: Key 'data' is not in struct
        # [CAUSE]: Incorrect config path - 'data' is nested under 'run', not at root level
        # [FIX]: Changed cfg.data to cfg.run.data to access the correct config structure
        #
        # [OLD CODE]:
        # cfg.data.eval_size = 10
        #
        # [NEW CODE]:
        # Reduce evaluation size for quick sanity check
        cfg.run.data.eval_size = 10
        # Set WandB project to sanity namespace
        if not cfg.wandb.get("project_override", False):
            cfg.wandb.project = f"{cfg.wandb.project}-sanity"
        cfg.wandb.mode = "online"
        
    elif cfg.mode == "pilot":
        print("Applying pilot mode overrides...")
        # [VALIDATOR FIX - Attempt 1]
        # [PROBLEM]: ConfigAttributeError: Key 'data' is not in struct
        # [CAUSE]: Incorrect config path - 'data' is nested under 'run', not at root level
        # [FIX]: Changed cfg.data to cfg.run.data to access the correct config structure
        #
        # [OLD CODE]:
        # cfg.data.eval_size = min(cfg.data.eval_size, 50)
        #
        # [NEW CODE]:
        # Reduce to smaller dataset for pilot runs
        cfg.run.data.eval_size = min(cfg.run.data.eval_size, 50)
        cfg.wandb.mode = "online"
    
    # Create results directory
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")
    
    # Save resolved config
    config_path = results_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"Saved config to: {config_path}")
    
    # Invoke inference.py as subprocess
    print("\n" + "="*80)
    print("Invoking inference script...")
    print("="*80 + "\n")
    
    inference_script = Path(__file__).parent / "inference.py"
    cmd = [
        sys.executable,
        "-u",
        str(inference_script),
        f"--config={config_path}",
    ]
    
    result = subprocess.run(cmd, cwd=Path.cwd())
    
    if result.returncode != 0:
        print(f"\nERROR: Inference script failed with code {result.returncode}")
        sys.exit(result.returncode)
    
    print("\n" + "="*80)
    print(f"Run {cfg.run.run_id} completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
