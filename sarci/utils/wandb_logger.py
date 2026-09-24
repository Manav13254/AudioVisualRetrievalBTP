"""
Thin wandb wrapper so scripts/train.py doesn't need `if args.use_wandb:`
scattered everywhere -- when disabled (or wandb isn't installed), every
method is a no-op, so the calling code never has to branch.
"""


class WandbLogger:
    def __init__(self, enabled: bool, project: str, run_name: str, config: dict):
        self.enabled = enabled
        self._wandb = None
        if not enabled:
            return
        try:
            import wandb
        except ImportError:
            print("WARNING: --use_wandb was set but `wandb` isn't installed (pip install wandb). "
                  "Continuing without logging.")
            self.enabled = False
            return
        self._wandb = wandb
        wandb.init(project=project, name=run_name, config=config)

    def log(self, data: dict, step=None):
        if self.enabled:
            self._wandb.log(data, step=step)

    def log_image(self, key: str, path: str, step=None):
        if self.enabled:
            self._wandb.log({key: self._wandb.Image(path)}, step=step)

    def watch_checkpoint(self, path: str):
        """Mark the current run's best checkpoint (as a run summary field,
        not an artifact upload -- checkpoints here are large and stay local)."""
        if self.enabled:
            self._wandb.run.summary["best_checkpoint"] = path

    def finish(self):
        if self.enabled:
            self._wandb.finish()
