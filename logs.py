import json
import os
from functools import partial

import torch

import wandb


def init_wandb(cfg):
    return wandb.init(project=cfg["wandb_project"], name=cfg["name"], config=cfg, reinit=True)

def log_wandb(output, step, wandb_run, index=None):
    log_dict = {
        k: v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v
        for k, v in output.items() if isinstance(v, (int, float)) or
        (isinstance(v, torch.Tensor) and v.dim() == 0)
    }

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)

# Hooks for model performance evaluation
def reconstr_hook(activation, hook, sae_out):
    return sae_out

def zero_abl_hook(activation, hook):
    return torch.zeros_like(activation)

def mean_abl_hook(activation, hook):
    return activation.mean([0, 1]).expand_as(activation)

@torch.no_grad()
def log_model_performance(wandb_run, step, model, activations_store, sae, index=None, batch_tokens=None):
    if batch_tokens is None:
        batch_tokens = activations_store.get_batch_tokens()[:sae.config["batch_size"] // sae.config["seq_len"]]
    batch = activations_store.get_activations(batch_tokens).reshape(-1, sae.config["act_size"])

    sae_output = sae(batch)["sae_out"].reshape(batch_tokens.shape[0], batch_tokens.shape[1], -1)

    original_loss = model(batch_tokens, return_type="loss").item()
    reconstr_loss = model.run_with_hooks(
        batch_tokens,


        fwd_hooks=[(sae.config["hook_point"], partial(reconstr_hook, sae_out=sae_output))],
        return_type="loss",
    ).item()
    zero_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(sae.config["hook_point"], zero_abl_hook)],
        return_type="loss",
    ).item()
    mean_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(sae.config["hook_point"], mean_abl_hook)],
        return_type="loss",
    ).item()

    ce_degradation = original_loss - reconstr_loss
    zero_degradation = original_loss - zero_loss
    mean_degradation = original_loss - mean_loss

    log_dict = {
        "performance/ce_degradation": ce_degradation,
        "performance/recovery_from_zero": (reconstr_loss - zero_loss) / zero_degradation,
        "performance/recovery_from_mean": (reconstr_loss - mean_loss) / mean_degradation,
    }

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)

def save_checkpoint(wandb_run, sae, cfg, step):
    save_dir = f"checkpoints/{cfg['name']}_{step}"
    os.makedirs(save_dir, exist_ok=True)

    # Save model state
    sae_path = os.path.join(save_dir, "sae.pt")
    torch.save(sae.state_dict(), sae_path)

    # Prepare config for JSON serialization
    json_safe_cfg = {}
    for key, value in cfg.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            json_safe_cfg[key] = value
        elif isinstance(value, (torch.dtype, type)):
            json_safe_cfg[key] = str(value)
        else:
            json_safe_cfg[key] = str(value)

    # Save config
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(json_safe_cfg, f, indent=4)

    # Create and log artifact
    artifact = wandb.Artifact(
        name=f"{cfg['name']}_{step}",
        type="model",
        description=f"Model checkpoint at step {step}",
    )
    artifact.add_file(sae_path)
    artifact.add_file(config_path)
    wandb_run.log_artifact(artifact)

    print(f"Model and config saved as artifact at step {step}")

def find_latest_checkpoint(cfg):
    """Find the most recent checkpoint for a given config based on step number.

    Args:
        cfg: Config dictionary containing name field
    Returns:
        latest_step: Step number of most recent checkpoint (-1 if none found)
        checkpoint_dir: Path to most recent checkpoint (None if none found)
    """
    if os.getcwd().endswith("alowet"):
        base_dir = "AISC/matryoshka_sae/checkpoints"
    else:
        base_dir = "checkpoints"
    if not os.path.exists(base_dir):
        return -1, None

    # Find all checkpoint directories for this run
    checkpoints = []
    for dirname in os.listdir(base_dir):
        if dirname.startswith(cfg['name']):
            try:
                step = int(dirname.split('_')[-1])
                checkpoints.append((step, os.path.join(base_dir, dirname)))
            except ValueError:
                continue

    if not checkpoints:
        return -1, None

    # Return the checkpoint with highest step number
    latest_step, latest_dir = max(checkpoints, key=lambda x: x[0])
    return latest_step, latest_dir

def load_checkpoint(cfg, step=None):
    """Load checkpoint from a specific step or latest if step not specified.

    Args:
        cfg: Config dictionary
        step: Optional specific step to load
    Returns:
        sae: Loaded SAE model
        start_step: Step to resume training from
    """
    if step is None:
        step, checkpoint_dir = find_latest_checkpoint(cfg)
        if step == -1:
            return None, 0
    else:
        checkpoint_dir = f"checkpoints/{cfg['name']}_{step}"
        if not os.path.exists(checkpoint_dir):
            return None, 0

    sae_path = os.path.join(checkpoint_dir, "sae.pt")
    if not os.path.exists(sae_path):
        return None, 0

    print(f"Loading checkpoint from step {step}")
    state_dict = torch.load(sae_path)
    return state_dict, step + 1
