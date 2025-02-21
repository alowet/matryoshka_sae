import argparse
import copy
import os

import sae
import torch
from activation_store import ActivationsStore
from config import get_default_cfg, post_init_cfg
from sae import BatchTopKSAE, GlobalBatchTopKMatryoshkaSAE
from training import train_sae_group_seperate_wandb
from transformer_lens import HookedTransformer


def run_matryoshka_sae(model_name, layer, site, dataset_path):

    cfg = get_default_cfg()
    cfg["model_name"] = model_name
    cfg["layer"] = layer
    cfg["site"] = site
    cfg["dataset_path"] = dataset_path
    cfg["aux_penalty"] = (1/32)
    cfg["lr"] = 3e-4
    cfg["input_unit_norm"] = False
    cfg["dict_size"] = 36864
    cfg['wandb_project'] = 'batch-topk-matryoshka'
    cfg['l1_coeff'] = 0.
    cfg['act_size'] = 2304
    cfg['device'] = 'cuda'
    cfg['bandwidth'] = 0.001
    cfg["top_k_matryoshka"] = [10, 10, 10, 10, 10]
    cfg["group_sizes"] = [2304//4, 2304 // 4 ,2304 // 2, 2304, 2304*2, 2304*4, 2304*8]
    cfg["num_tokens"] = 5e8
    cfg["model_batch_size"] = 32
    cfg["model_dtype"] = torch.bfloat16
    cfg = post_init_cfg(cfg)

    # Train the BatchTopK SAEs
    dict_sizes = [2304, 2304*2, 2304*4, 2304*8, 2304*16]
    topks = [22, 25, 27, 29, 32]

    model = HookedTransformer.from_pretrained_no_processing(cfg["model_name"]).to(
        cfg["model_dtype"]).to(cfg["device"])
    activations_store = ActivationsStore(model, cfg)
    saes = []
    cfgs = []

    for i, (dict_size, topk) in enumerate(zip(dict_sizes, topks)):
        cfg = copy.deepcopy(cfg)
        cfg["sae_type"] = 'batch-topk'
        cfg["dict_size"] = dict_size
        cfg["top_k"] = topk

        cfg = post_init_cfg(cfg)
        sae = BatchTopKSAE(cfg)
        saes.append(sae)
        cfgs.append(cfg)

    # Train the Matryoshka SAE
    dict_size = 2304*16
    topk = 32
    cfg = copy.deepcopy(cfg)
    cfg["sae_type"] = 'global-matryoshka-topk'
    cfg["dict_size"] = dict_size
    cfg["top_k"] = topk
    cfg["group_sizes"] = [dict_size // 16, dict_size // 16, dict_size // 8, dict_size // 4, dict_size // 2]

    sae = GlobalBatchTopKMatryoshkaSAE(cfg)
    saes.append(sae)
    cfgs.append(cfg)

    train_sae_group_seperate_wandb(saes, activations_store, model, cfgs)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gemma-2-2b")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--site", type=str, default="resid_pre")
    parser.add_argument("--dataset_path", type=str, default="Skylion007/openwebtext")
    args = parser.parse_args()
    run_matryoshka_sae(args.model_name, args.layer, args.site, args.dataset_path)
