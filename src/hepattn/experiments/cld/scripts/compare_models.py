"""
Compare performance of hit-only model vs track+topo model.

Usage:
    python src/hepattn/experiments/cld/scripts/compare_models.py \
        --hit-eval-file /path/to/hit_model/preds.h5 \
        --topo-eval-file /path/to/topo_model/preds.h5 \
        --plot-root /path/to/output/plots \
        --max-events 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from hepattn.experiments.cld.data import CLDDataModule

plt.rcParams["text.usetex"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["figure.constrained_layout.use"] = True


def compute_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two boolean masks.
    mask_a: [B, N, H] - N objects, H hits
    mask_b: [B, M, H] - M objects, H hits
    Returns: [B, N, M] IoU matrix
    """
    # Expand for pairwise comparison
    a = mask_a.unsqueeze(2).float()  # [B, N, 1, H]
    b = mask_b.unsqueeze(1).float()  # [B, 1, M, H]

    intersection = (a * b).sum(-1)  # [B, N, M]
    union = (a + b).clamp(max=1).sum(-1)  # [B, N, M]

    iou = intersection / union.clamp(min=1e-6)
    return iou


def match_objects(cost_matrix: torch.Tensor, valid_a: torch.Tensor, valid_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Greedy matching based on cost matrix.
    Returns matched indices for both sides.
    """
    B, N, M = cost_matrix.shape
    matched_a = torch.zeros(B, N, dtype=torch.bool, device=cost_matrix.device)
    matched_b = torch.zeros(B, M, dtype=torch.bool, device=cost_matrix.device)
    match_iou = torch.zeros(B, N, device=cost_matrix.device)

    for b in range(B):
        # Get valid objects
        valid_mask_a = valid_a[b]
        valid_mask_b = valid_b[b]

        # Create cost matrix for this batch
        costs = cost_matrix[b].clone()
        costs[~valid_mask_a, :] = -float('inf')
        costs[:, ~valid_mask_b] = -float('inf')

        # Greedy matching
        used_b = set()
        for i in range(N):
            if not valid_mask_a[i]:
                continue
            # Find best match for object i
            best_j = -1
            best_score = -float('inf')
            for j in range(M):
                if j in used_b or not valid_mask_b[j]:
                    continue
                if costs[i, j] > best_score:
                    best_score = costs[i, j]
                    best_j = j

            if best_j >= 0 and best_score > 0:
                matched_a[b, i] = True
                matched_b[b, best_j] = True
                match_iou[b, i] = best_score
                used_b.add(best_j)

    return matched_a, match_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare hit-only vs track+topo models")

    parser.add_argument("--hit-eval-file", type=Path, required=True, help="Path to hit-only model predictions .h5 file")
    parser.add_argument("--topo-eval-file", type=Path, required=True, help="Path to track+topo model predictions .h5 file")
    parser.add_argument("--plot-root", type=Path, required=True, help="Root directory for output plots")
    parser.add_argument("--max-events", type=int, default=1000, help="Maximum number of events to process")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for matching")

    return parser.parse_args()


def evaluate_hit_model(
    eval_file_path: Path,
    data_cfg: dict,
    max_events: int,
    iou_thresh: float,
) -> dict[str, Any]:
    """Evaluate hit-only model."""
    print(f"\n{'='*60}")
    print("Evaluating: Hit-Only Model")
    print(f"{'='*60}")

    # Setup dataset
    datamodule = CLDDataModule(**data_cfg)
    datamodule.setup(stage="test")
    dataset = datamodule.test_dataloader().dataset

    metrics = {
        "total_particles": 0,
        "total_charged": 0,
        "total_neutral": 0,
        "matched_charged": 0,
        "matched_neutral": 0,
        "total_flow": 0,
    }

    with h5py.File(eval_file_path, "r") as f:
        keys = list(f.keys())[:max_events]

        for sample_id in tqdm(keys, desc="Hit-Only"):
            try:
                preds = f[f"{sample_id}/preds/final/"]
                outs = f[f"{sample_id}/outputs/final/"]

                # Load flow predictions
                flow_logit = torch.from_numpy(outs["flow_valid/flow_logit"][:])
                flow_valid = flow_logit.sigmoid() >= 0.5

                # Load hit mask predictions
                flow_masks = {}
                for hit in ["vtxd", "trkr", "ecal", "hcal"]:
                    key = f"flow_{hit}_assignment/flow_{hit}_valid"
                    if key in preds:
                        flow_masks[hit] = torch.from_numpy(preds[key][:])

                # Load sample
                sample = dataset.load_sample(int(sample_id))
                if sample is None:
                    continue

                inputs, targets = dataset.prep_sample(sample)

                # Get particle info
                particle_valid = targets["particle_valid"].bool()
                is_charged = targets["particle_is_charged"].bool() if "particle_is_charged" in targets else (targets["particle_charge"].abs() > 0)
                is_neutral = ~is_charged & particle_valid

                # Combine silicon hits
                if "particle_vtxd_valid" in targets and "particle_trkr_valid" in targets:
                    particle_sihit = torch.cat([targets["particle_vtxd_valid"], targets["particle_trkr_valid"]], dim=-1)
                    if "vtxd" in flow_masks and "trkr" in flow_masks:
                        n_vtxd = targets["vtxd_valid"].shape[-1]
                        n_trkr = targets["trkr_valid"].shape[-1]
                        flow_sihit = torch.cat([
                            flow_masks["vtxd"][:, :, :n_vtxd],
                            flow_masks["trkr"][:, :, :n_trkr]
                        ], dim=-1)
                    else:
                        continue
                else:
                    continue

                # Get calo masks
                if "particle_ecal_valid" in targets and "ecal" in flow_masks:
                    n_ecal = targets["ecal_valid"].shape[-1]
                    particle_ecal = targets["particle_ecal_valid"]
                    flow_ecal = flow_masks["ecal"][:, :, :n_ecal]
                else:
                    particle_ecal = torch.zeros_like(particle_sihit[:, :, :1])
                    flow_ecal = torch.zeros_like(flow_sihit[:, :, :1])

                if "particle_hcal_valid" in targets and "hcal" in flow_masks:
                    n_hcal = targets["hcal_valid"].shape[-1]
                    particle_hcal = targets["particle_hcal_valid"]
                    flow_hcal = flow_masks["hcal"][:, :, :n_hcal]
                else:
                    particle_hcal = torch.zeros_like(particle_sihit[:, :, :1])
                    flow_hcal = torch.zeros_like(flow_sihit[:, :, :1])

                # Compute IoU for charged particles (silicon hits)
                sihit_iou = compute_iou(particle_sihit.bool(), flow_sihit.bool())

                # Compute IoU for calo (combined ecal + hcal)
                particle_calo = torch.cat([particle_ecal, particle_hcal], dim=-1)
                flow_calo = torch.cat([flow_ecal, flow_hcal], dim=-1)
                calo_iou = compute_iou(particle_calo.bool(), flow_calo.bool())

                # Combined cost for charged: sihit + calo
                charged_cost = 0.5 * sihit_iou + 0.5 * calo_iou

                # Neutral cost: just calo
                neutral_cost = calo_iou

                # Match charged particles
                charged_mask = particle_valid & is_charged
                matched_charged, _ = match_objects(charged_cost, charged_mask.squeeze(0), flow_valid.squeeze(0))
                charged_matched = matched_charged & charged_mask

                # Match neutral particles
                neutral_mask = particle_valid & is_neutral
                matched_neutral, _ = match_objects(neutral_cost, neutral_mask.squeeze(0), flow_valid.squeeze(0))
                neutral_matched = matched_neutral & neutral_mask

                # Accumulate
                metrics["total_particles"] += particle_valid.sum().item()
                metrics["total_charged"] += charged_mask.sum().item()
                metrics["total_neutral"] += neutral_mask.sum().item()
                metrics["matched_charged"] += charged_matched.sum().item()
                metrics["matched_neutral"] += neutral_matched.sum().item()
                metrics["total_flow"] += flow_valid.sum().item()

            except Exception as e:
                print(f"Error processing {sample_id}: {e}")
                continue

    return metrics


def evaluate_topo_model(
    eval_file_path: Path,
    data_cfg: dict,
    max_events: int,
    iou_thresh: float,
) -> dict[str, Any]:
    """Evaluate track+topo model."""
    print(f"\n{'='*60}")
    print("Evaluating: Track+Topo Model")
    print(f"{'='*60}")

    # Setup dataset
    datamodule = CLDDataModule(**data_cfg)
    datamodule.setup(stage="test")
    dataset = datamodule.test_dataloader().dataset

    metrics = {
        "total_particles": 0,
        "total_charged": 0,
        "total_neutral": 0,
        "matched_charged": 0,
        "matched_neutral": 0,
        "total_flow": 0,
    }

    with h5py.File(eval_file_path, "r") as f:
        keys = list(f.keys())[:max_events]

        for sample_id in tqdm(keys, desc="Track+Topo"):
            try:
                preds = f[f"{sample_id}/preds/final/"]
                outs = f[f"{sample_id}/outputs/final/"]

                # Load flow predictions
                flow_logit = torch.from_numpy(outs["flow_valid/flow_logit"][:])
                flow_valid = flow_logit.sigmoid() >= 0.5

                # Load sitrack and topocluster predictions
                flow_sitrack = None
                flow_topocluster = None

                sitrack_key = "flow_sitrack_assignment/flow_sitrack_valid"
                if sitrack_key in preds:
                    flow_sitrack = torch.from_numpy(preds[sitrack_key][:])

                topo_key = "flow_topocluster_assignment/flow_topocluster_valid"
                if topo_key in preds:
                    flow_topocluster = torch.from_numpy(preds[topo_key][:])

                # Load sample
                sample = dataset.load_sample(int(sample_id))
                if sample is None:
                    continue

                inputs, targets = dataset.prep_sample(sample)

                # Get particle info
                particle_valid = targets["particle_valid"].bool()
                is_charged = targets["particle_is_charged"].bool() if "particle_is_charged" in targets else (targets["particle_charge"].abs() > 0)
                is_neutral = ~is_charged & particle_valid

                # Get particle-to-sitrack and particle-to-topocluster masks
                if "particle_sitrack_valid" in targets and flow_sitrack is not None:
                    particle_sitrack = targets["particle_sitrack_valid"]
                    n_sitrack = targets["sitrack_valid"].shape[-1]
                    flow_sitrack = flow_sitrack[:, :, :n_sitrack]
                else:
                    particle_sitrack = None

                if "particle_topocluster_valid" in targets and flow_topocluster is not None:
                    particle_topocluster = targets["particle_topocluster_valid"]
                    n_topo = targets["topocluster_valid"].shape[-1]
                    flow_topocluster = flow_topocluster[:, :, :n_topo]
                else:
                    particle_topocluster = None

                # Compute IoU for sitrack
                if particle_sitrack is not None:
                    sitrack_iou = compute_iou(particle_sitrack.bool(), flow_sitrack.bool())
                else:
                    sitrack_iou = torch.zeros(1, particle_valid.shape[1], flow_valid.shape[1])

                # Compute IoU for topocluster
                if particle_topocluster is not None:
                    topo_iou = compute_iou(particle_topocluster.bool(), flow_topocluster.bool())
                else:
                    topo_iou = torch.zeros(1, particle_valid.shape[1], flow_valid.shape[1])

                # Combined cost for charged: sitrack + topocluster
                charged_cost = 0.5 * sitrack_iou + 0.5 * topo_iou

                # Neutral cost: just topocluster
                neutral_cost = topo_iou

                # Match charged particles
                charged_mask = particle_valid & is_charged
                matched_charged, _ = match_objects(charged_cost, charged_mask.squeeze(0), flow_valid.squeeze(0))
                charged_matched = matched_charged & charged_mask

                # Match neutral particles
                neutral_mask = particle_valid & is_neutral
                matched_neutral, _ = match_objects(neutral_cost, neutral_mask.squeeze(0), flow_valid.squeeze(0))
                neutral_matched = matched_neutral & neutral_mask

                # Accumulate
                metrics["total_particles"] += particle_valid.sum().item()
                metrics["total_charged"] += charged_mask.sum().item()
                metrics["total_neutral"] += neutral_mask.sum().item()
                metrics["matched_charged"] += charged_matched.sum().item()
                metrics["matched_neutral"] += neutral_matched.sum().item()
                metrics["total_flow"] += flow_valid.sum().item()

            except Exception as e:
                print(f"Error processing {sample_id}: {e}")
                continue

    return metrics


def print_comparison(hit_metrics: dict, topo_metrics: dict) -> None:
    """Print side-by-side comparison."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\n{'Metric':<35} {'Hit-Only':<22} {'Track+Topo':<22}")
    print("-" * 80)

    # Totals
    print(f"{'Total particles':<35} {hit_metrics['total_particles']:<22.0f} {topo_metrics['total_particles']:<22.0f}")
    print(f"{'Total flow predictions':<35} {hit_metrics['total_flow']:<22.0f} {topo_metrics['total_flow']:<22.0f}")

    # Charged
    print(f"\n{'--- Charged Particles ---':<35}")
    print(f"{'Total charged':<35} {hit_metrics['total_charged']:<22.0f} {topo_metrics['total_charged']:<22.0f}")
    print(f"{'Matched charged':<35} {hit_metrics['matched_charged']:<22.0f} {topo_metrics['matched_charged']:<22.0f}")

    hit_eff = 100 * hit_metrics['matched_charged'] / max(hit_metrics['total_charged'], 1)
    topo_eff = 100 * topo_metrics['matched_charged'] / max(topo_metrics['total_charged'], 1)
    print(f"{'Charged efficiency':<35} {hit_eff:<21.2f}% {topo_eff:<21.2f}%")

    # Neutral
    print(f"\n{'--- Neutral Particles ---':<35}")
    print(f"{'Total neutral':<35} {hit_metrics['total_neutral']:<22.0f} {topo_metrics['total_neutral']:<22.0f}")
    print(f"{'Matched neutral':<35} {hit_metrics['matched_neutral']:<22.0f} {topo_metrics['matched_neutral']:<22.0f}")

    hit_eff = 100 * hit_metrics['matched_neutral'] / max(hit_metrics['total_neutral'], 1)
    topo_eff = 100 * topo_metrics['matched_neutral'] / max(topo_metrics['total_neutral'], 1)
    print(f"{'Neutral efficiency':<35} {hit_eff:<21.2f}% {topo_eff:<21.2f}%")


def plot_comparison(hit_metrics: dict, topo_metrics: dict, plot_root: Path) -> None:
    """Create comparison bar chart."""
    plot_root.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Charged efficiency
    ax = axes[0]
    hit_eff = 100 * hit_metrics['matched_charged'] / max(hit_metrics['total_charged'], 1)
    topo_eff = 100 * topo_metrics['matched_charged'] / max(topo_metrics['total_charged'], 1)

    x = np.array([0])
    width = 0.35
    ax.bar(x - width/2, [hit_eff], width, label="Hit-Only", color="tab:blue")
    ax.bar(x + width/2, [topo_eff], width, label="Track+Topo", color="tab:orange")
    ax.set_ylabel("Efficiency (%)")
    ax.set_title("Charged Particle Reconstruction")
    ax.set_xticks([])
    ax.legend()
    ax.set_ylim(0, 100)

    # Neutral efficiency
    ax = axes[1]
    hit_eff = 100 * hit_metrics['matched_neutral'] / max(hit_metrics['total_neutral'], 1)
    topo_eff = 100 * topo_metrics['matched_neutral'] / max(topo_metrics['total_neutral'], 1)

    ax.bar(x - width/2, [hit_eff], width, label="Hit-Only", color="tab:blue")
    ax.bar(x + width/2, [topo_eff], width, label="Track+Topo", color="tab:orange")
    ax.set_ylabel("Efficiency (%)")
    ax.set_title("Neutral Particle Reconstruction")
    ax.set_xticks([])
    ax.legend()
    ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(plot_root / "model_comparison.png")
    print(f"\nSaved plot to: {plot_root / 'model_comparison.png'}")


def main() -> None:
    args = parse_args()

    # Load hit model config
    hit_config_path = args.hit_eval_file.parent.parent / "config.yaml"
    hit_data_cfg = yaml.safe_load(hit_config_path.read_text())["data"]
    hit_data_cfg["num_workers"] = 0
    hit_data_cfg["batch_size"] = 1
    hit_data_cfg["num_test"] = -1

    # Load topo model config
    topo_config_path = args.topo_eval_file.parent.parent / "config.yaml"
    topo_data_cfg = yaml.safe_load(topo_config_path.read_text())["data"]
    topo_data_cfg["num_workers"] = 0
    topo_data_cfg["batch_size"] = 1
    topo_data_cfg["num_test"] = -1

    # Evaluate both models
    hit_metrics = evaluate_hit_model(
        args.hit_eval_file,
        hit_data_cfg,
        args.max_events,
        args.iou_thresh,
    )

    topo_metrics = evaluate_topo_model(
        args.topo_eval_file,
        topo_data_cfg,
        args.max_events,
        args.iou_thresh,
    )

    # Print and plot comparison
    print_comparison(hit_metrics, topo_metrics)
    plot_comparison(hit_metrics, topo_metrics, args.plot_root)


if __name__ == "__main__":
    main()
