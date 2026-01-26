# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Script for calculating Frechet Inception Distance (FID), Inception Score (IS),
and Precision/Recall (PR) based on k-NN manifold approximation.

- FID: uses Inception-v3 pool3 features (2048-d) and Frechet distance.
- IS: uses Inception-v3 class probabilities; supports split-based mean/std.
- PR: uses k-NN radius manifold test in feature space (improved P/R-style), computed on Inception features.
"""

import os
import json
import math
import click
import tqdm
import pickle
import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F
import dnnlib
from torch_utils import distributed as dist
from training import dataset

#----------------------------------------------------------------------------

class IndexedDataset(torch.utils.data.Dataset):
    """Wrap a dataset to also return sample index."""
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        img, lbl = self.base_ds[idx]
        return img, lbl, idx

#----------------------------------------------------------------------------

def _load_detector(detector_url, device):
    with dnnlib.util.open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f).to(device)
    net.eval()
    return net

#----------------------------------------------------------------------------

def _as_tensor_output(out):
    """Robustly pick a tensor from a detector output."""
    if torch.is_tensor(out):
        return out
    if isinstance(out, (list, tuple)):
        # Prefer a 2D tensor
        cand = [x for x in out if torch.is_tensor(x) and x.ndim == 2]
        if len(cand) > 0:
            # Heuristic: pick the one that looks like probabilities (sum~1, values in [0,1])
            for x in cand:
                xf = x.float()
                sums = xf.sum(dim=1)
                if xf.min().item() >= -1e-3 and xf.max().item() <= 1.0 + 1e-3:
                    if torch.allclose(sums, torch.ones_like(sums), rtol=1e-2, atol=1e-2):
                        return x
            return cand[-1]
        # Fallback: first tensor
        for x in out:
            if torch.is_tensor(x):
                return x
    if isinstance(out, dict):
        for k in ['probs', 'prob', 'p', 'logits', 'output']:
            if k in out and torch.is_tensor(out[k]):
                return out[k]
        # fallback: first tensor value
        for v in out.values():
            if torch.is_tensor(v):
                return v
    raise RuntimeError(f"Unsupported detector output type: {type(out)}")

#----------------------------------------------------------------------------

def _to_probs(x):
    """Convert a (logits or probs) tensor to probabilities."""
    x = x.float()
    # If already looks like probs, keep
    sums = x.sum(dim=1)
    if x.min().item() >= -1e-3 and x.max().item() <= 1.0 + 1e-3 and torch.allclose(
        sums, torch.ones_like(sums), rtol=1e-2, atol=1e-2
    ):
        probs = x
    else:
        probs = torch.softmax(x, dim=1)
    # Numerical safety
    probs = torch.clamp(probs, min=1e-20, max=1.0)
    return probs

#----------------------------------------------------------------------------

def calculate_inception_stats(
    image_path, num_expected=None, seed=0, max_batch_size=64,
    num_workers=3, prefetch_factor=2, device=torch.device('cuda'),
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load Inception-v3 model.
    dist.print0('Loading Inception-v3 model...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    detector_net = _load_detector(detector_url, device)

    # List images.
    dist.print0(f'Loading images from "{image_path}"...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
    if num_expected is not None and len(dataset_obj) < num_expected:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_expected}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(
        dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor
    )

    # Accumulate statistics.
    dist.print0(f'Calculating statistics for {len(dataset_obj)} images...')
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)

    with torch.no_grad():
        for images, _labels in tqdm.tqdm(data_loader, unit='batch', disable=(dist.get_rank() != 0)):
            torch.distributed.barrier()
            if images.shape[0] == 0:
                continue
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])

            feats = detector_net(images.to(device), **detector_kwargs)
            feats = _as_tensor_output(feats).to(torch.float64)

            mu += feats.sum(0)
            sigma += feats.T @ feats

    # Calculate grand totals.
    torch.distributed.all_reduce(mu)
    torch.distributed.all_reduce(sigma)
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()

#----------------------------------------------------------------------------

def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

#----------------------------------------------------------------------------

def calculate_inception_score(
    image_path, num_expected=None, seed=0, max_batch_size=64, num_splits=10,
    num_workers=3, prefetch_factor=2, device=torch.device('cuda'),
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    dist.print0('Loading Inception-v3 model for IS...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_net = _load_detector(detector_url, device)

    # List images.
    dist.print0(f'Loading images from "{image_path}" for IS...')
    base_ds = dataset.ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
    if num_expected is not None and len(base_ds) < num_expected:
        raise click.ClickException(f'Found {len(base_ds)} images, but expected at least {num_expected}')
    if len(base_ds) < 2:
        raise click.ClickException(f'Found {len(base_ds)} images, but need at least 2 to compute IS')
    ds = IndexedDataset(base_ds)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = ((len(ds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(ds)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(
        ds, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor
    )

    # We accumulate split-wise sufficient stats:
    # sum_p[s, c] = Σ p(y=c|x) over x in split s
    # sum_plogp[s] = Σ Σ p(y|x) log p(y|x) over x in split s
    # count[s] = #samples in split s
    sum_p = None
    sum_plogp = torch.zeros([num_splits], dtype=torch.float64, device=device)
    count = torch.zeros([num_splits], dtype=torch.float64, device=device)

    dist.print0(f'Calculating IS for {len(ds)} images (splits={num_splits})...')

    with torch.no_grad():
        for images, _labels, idxs in tqdm.tqdm(data_loader, unit='batch', disable=(dist.get_rank() != 0)):
            torch.distributed.barrier()
            if images.shape[0] == 0:
                continue
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])

            # Match StyleGAN3 convention: no_output_bias=True when computing probs.
            # Some detector builds may not accept it; we fallback safely.
            try:
                out = detector_net(images.to(device), return_features=False, no_output_bias=True)
            except TypeError:
                out = detector_net(images.to(device), return_features=False)

            out = _as_tensor_output(out)
            probs = _to_probs(out)  # [B, C]
            B, C = probs.shape

            if sum_p is None:
                sum_p = torch.zeros([num_splits, C], dtype=torch.float64, device=device)

            # Assign each sample to the same contiguous split scheme as "array slicing"
            # based on its global dataset index.
            idxs = idxs.to(device=device)
            split_ids = (idxs.to(torch.float64) * num_splits / float(len(ds))).floor().clamp(0, num_splits - 1).to(torch.int64)

            probs64 = probs.to(torch.float64)
            sum_p.index_add_(0, split_ids, probs64)
            plogp = (probs64 * probs64.log()).sum(dim=1)  # [B]
            sum_plogp.index_add_(0, split_ids, plogp)

            ones = torch.ones_like(split_ids, dtype=torch.float64, device=device)
            count.index_add_(0, split_ids, ones)

    # Reduce across ranks.
    if sum_p is None:
        raise click.ClickException('No images were processed for IS (empty dataloader).')
    torch.distributed.all_reduce(sum_p)
    torch.distributed.all_reduce(sum_plogp)
    torch.distributed.all_reduce(count)

    if dist.get_rank() != 0:
        return float('nan'), float('nan')

    # Compute IS per split:
    # E[KL] = (Σ p log p)/N - Σ p_bar log p_bar
    scores = []
    for s in range(num_splits):
        Ns = count[s].item()
        if Ns <= 0:
            continue
        p_bar = sum_p[s] / Ns
        term1 = (sum_plogp[s] / Ns).item()
        term2 = float((p_bar * p_bar.clamp(min=1e-20).log()).sum().item())
        kl = term1 - term2
        scores.append(math.exp(kl))

    if len(scores) == 0:
        raise click.ClickException('IS split computation failed (no valid splits).')

    scores = np.asarray(scores, dtype=np.float64)
    return float(scores.mean()), float(scores.std(ddof=0))

#----------------------------------------------------------------------------

def extract_inception_features(
    image_path, num_expected=None, seed=0, max_batch_size=64,
    num_workers=3, prefetch_factor=2, device=torch.device('cuda'),
):
    """Extract and return all Inception (2048-d) features for images (rank0 only for PR correctness)."""
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    dist.print0('Loading Inception-v3 model for feature extraction (PR)...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    detector_net = _load_detector(detector_url, device)

    dist.print0(f'Loading images from "{image_path}" for PR features...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
    if num_expected is not None and len(dataset_obj) < num_expected:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_expected}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute PR')

    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # For correctness and simplicity, require single-process when computing PR.
    if dist.get_world_size() != 1:
        raise click.ClickException('Precision/Recall computation requires --nproc_per_node=1 in this script (to avoid incorrect partial feature sets).')

    data_loader = torch.utils.data.DataLoader(
        dataset_obj, batch_size=max_batch_size, shuffle=False,
        num_workers=num_workers, prefetch_factor=prefetch_factor
    )

    feats_all = []
    with torch.no_grad():
        for images, _labels in tqdm.tqdm(data_loader, unit='batch', disable=False):
            if images.shape[0] == 0:
                continue
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])
            feats = detector_net(images.to(device), **detector_kwargs)
            feats = _as_tensor_output(feats).to(torch.float32)  # keep float32 for stability
            feats_all.append(feats.cpu())

    feats_all = torch.cat(feats_all, dim=0)
    return feats_all  # [N, 2048] on CPU

#----------------------------------------------------------------------------

def _pairwise_dist_sq(row, col, col_norm=None):
    """Compute squared Euclidean distances: ||row_i - col_j||^2 using matmul for speed."""
    row = row.float()
    col = col.float()
    row_norm = (row * row).sum(dim=1, keepdim=True)  # [R,1]
    if col_norm is None:
        col_norm = (col * col).sum(dim=1)  # [C]
    col_norm = col_norm.view(1, -1)  # [1,C]
    # dist^2 = ||a||^2 + ||b||^2 - 2 a.b
    dist_sq = row_norm + col_norm - 2.0 * (row @ col.t())
    dist_sq = torch.clamp(dist_sq, min=0.0)
    return dist_sq

#----------------------------------------------------------------------------

def _knn_radii_sq(manifold, k, batch_size, device):
    """For each point in manifold, compute squared distance to its (k+1)-th nearest neighbor within manifold."""
    N = manifold.shape[0]
    if k + 1 > N:
        raise click.ClickException(f'PR: k={k} too large for manifold size N={N}. Need k+1 <= N.')
    manifold = manifold.to(device)
    col_norm = (manifold.float() * manifold.float()).sum(dim=1)  # [N]

    radii_sq = []
    for start in range(0, N, batch_size):
        row = manifold[start:start+batch_size]
        dist_sq = _pairwise_dist_sq(row, manifold, col_norm=col_norm)  # [B,N]
        kth = dist_sq.kthvalue(k + 1, dim=1).values  # +1 to skip self (0)
        radii_sq.append(kth.detach().cpu())
    return torch.cat(radii_sq, dim=0)  # [N] on CPU

#----------------------------------------------------------------------------

def calculate_precision_recall_knn(
    real_features_cpu, gen_features_cpu, k=3, batch_size=512, device=torch.device('cuda')
):
    """Compute precision/recall using k-NN manifold radii test (improved P/R-style) in feature space.

    Precision: fraction of generated samples that fall inside the real manifold union of balls.
    Recall:    fraction of real samples that fall inside the generated manifold union of balls.
    """
    if dist.get_rank() != 0:
        return float('nan'), float('nan')

    real = real_features_cpu
    fake = gen_features_cpu
    if real.ndim != 2 or fake.ndim != 2:
        raise click.ClickException('PR: features must be 2D tensors [N, D].')
    if real.shape[1] != fake.shape[1]:
        raise click.ClickException(f'PR: feature dims differ: real {real.shape} vs fake {fake.shape}.')

    dist.print0(f'PR: computing radii with k={k}, batch_size={batch_size}...')
    real_radii_sq = _knn_radii_sq(real, k=k, batch_size=batch_size, device=device)  # [Nr] CPU
    fake_radii_sq = _knn_radii_sq(fake, k=k, batch_size=batch_size, device=device)  # [Nf] CPU

    # Move manifolds to GPU once for repeated matmuls.
    real_gpu = real.to(device)
    fake_gpu = fake.to(device)
    real_norm = (real_gpu.float() * real_gpu.float()).sum(dim=1).detach()  # [Nr]
    fake_norm = (fake_gpu.float() * fake_gpu.float()).sum(dim=1).detach()  # [Nf]

    real_radii_sq_gpu = real_radii_sq.to(device)
    fake_radii_sq_gpu = fake_radii_sq.to(device)

    # Precision: probes=fake, manifold=real
    dist.print0('PR: computing precision...')
    pred = []
    with torch.no_grad():
        Nf = fake_gpu.shape[0]
        Nr = real_gpu.shape[0]
        for start in tqdm.tqdm(range(0, Nf, batch_size), desc='precision', unit='batch'):
            probes = fake_gpu[start:start+batch_size]  # [B,D]
            dist_sq = _pairwise_dist_sq(probes, real_gpu, col_norm=real_norm)  # [B,Nr]
            inside = (dist_sq <= real_radii_sq_gpu.view(1, Nr)).any(dim=1)     # [B]
            pred.append(inside.detach().cpu())
    precision = torch.cat(pred, dim=0).float().mean().item()

    # Recall: probes=real, manifold=fake
    dist.print0('PR: computing recall...')
    pred = []
    with torch.no_grad():
        Nr = real_gpu.shape[0]
        Nf = fake_gpu.shape[0]
        for start in tqdm.tqdm(range(0, Nr, batch_size), desc='recall', unit='batch'):
            probes = real_gpu[start:start+batch_size]  # [B,D]
            dist_sq = _pairwise_dist_sq(probes, fake_gpu, col_norm=fake_norm)  # [B,Nf]
            inside = (dist_sq <= fake_radii_sq_gpu.view(1, Nf)).any(dim=1)     # [B]
            pred.append(inside.detach().cpu())
    recall = torch.cat(pred, dim=0).float().mean().item()

    return float(precision), float(recall)

#----------------------------------------------------------------------------

@click.group()
def main():
    """Calculate FID/IS/PR for a given set of images."""

#----------------------------------------------------------------------------

@main.command()
@click.option('--images', 'image_path', help='Path to the generated images', metavar='PATH|ZIP',              type=str, required=True)
@click.option('--ref', 'ref_path',      help='Dataset reference statistics for FID', metavar='NPZ|URL',       type=str, required=True)
@click.option('--data', 'data_path',    help='Path to the real dataset images for PR (folder/zip). Required if PR is requested.', metavar='PATH|ZIP', type=str, default=None)
@click.option('--metrics',              help='Comma-separated metrics to compute: fid,is,pr', metavar='STR', type=str, default='fid', show_default=True)
@click.option('--num_expected', 'num_expected',  help='Number of generated images to use', metavar='INT',              type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--seed',                 help='Random seed for selecting images', metavar='INT',              type=int, default=0, show_default=True)
@click.option('--batch',                help='Maximum batch size', metavar='INT',                             type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--is_splits',            help='Number of splits for Inception Score', metavar='INT',           type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--pr_k',                 help='k for k-NN manifold in PR', metavar='INT',                      type=click.IntRange(min=1), default=3, show_default=True)
@click.option('--pr_real',              help='Number of real samples for PR', metavar='INT',                  type=click.IntRange(min=2), default=10000, show_default=True)
@click.option('--pr_gen',               help='Number of generated samples for PR (<= --num recommended)', metavar='INT', type=click.IntRange(min=2), default=10000, show_default=True)
@click.option('--pr_batch',             help='Batch size for PR distance blocks', metavar='INT',              type=click.IntRange(min=1), default=512, show_default=True)

def calc(image_path, ref_path, data_path, metrics, num_expected, seed, batch, is_splits, pr_k, pr_real, pr_gen, pr_batch):
    """Calculate requested metrics for a given set of generated images."""
    torch.multiprocessing.set_start_method('spawn', force=True)
    dist.init()

    metrics_set = [m.strip().lower() for m in metrics.split(',') if m.strip()]
    allowed = {'fid', 'is', 'pr'}
    for m in metrics_set:
        if m not in allowed:
            raise click.ClickException(f'Unknown metric "{m}". Allowed: {sorted(list(allowed))}')

    # Load reference stats for FID
    dist.print0(f'Loading dataset reference statistics from "{ref_path}"...')
    ref = None
    if dist.get_rank() == 0:
        with dnnlib.util.open_url(ref_path) as f:
            ref = dict(np.load(f))

    results = {}

    # FID
    if 'fid' in metrics_set:
        mu, sigma = calculate_inception_stats(image_path=image_path, num_expected=num_expected, seed=seed, max_batch_size=batch)
        dist.print0('Calculating FID...')
        if dist.get_rank() == 0:
            fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
            results['fid'] = float(fid)

    # IS
    if 'is' in metrics_set:
        is_mean, is_std = calculate_inception_score(
            image_path=image_path, num_expected=num_expected, seed=seed, max_batch_size=batch, num_splits=is_splits
        )
        if dist.get_rank() == 0:
            results['is_mean'] = float(is_mean)
            results['is_std'] = float(is_std)

    # PR
    if 'pr' in metrics_set:
        if data_path is None:
            raise click.ClickException('PR requested but --data was not provided (need real dataset images).')
        if dist.get_world_size() != 1:
            raise click.ClickException('PR in this script requires --nproc_per_node=1 (single process).')

        # Extract features (Inception pool3) for a subset of real and generated images.
        device = torch.device('cuda')

        dist.print0(f'Extracting real features for PR: {pr_real} samples from "{data_path}"...')
        real_feats = extract_inception_features(
            image_path=data_path, num_expected=pr_real, seed=seed, max_batch_size=batch, device=device
        )

        dist.print0(f'Extracting generated features for PR: {pr_gen} samples from "{image_path}"...')
        gen_feats = extract_inception_features(
            image_path=image_path, num_expected=pr_gen, seed=seed, max_batch_size=batch, device=device
        )

        dist.print0('Calculating Precision/Recall (k-NN manifold test in Inception feature space)...')
        precision, recall = calculate_precision_recall_knn(
            real_features_cpu=real_feats,
            gen_features_cpu=gen_feats,
            k=pr_k,
            batch_size=pr_batch,
            device=device,
        )
        if dist.get_rank() == 0:
            results['precision'] = float(precision)
            results['recall'] = float(recall)
            results['pr_k'] = int(pr_k)
            results['pr_real'] = int(pr_real)
            results['pr_gen'] = int(pr_gen)

    # Output
    if dist.get_rank() == 0:
        # Backward compatible behavior: if only fid requested, print numeric only
        if metrics_set == ['fid']:
            print(f"{results['fid']:g}")
        else:
            print(json.dumps(results, indent=2, sort_keys=True))
            # save json
            with open(f'{image_path}/result_all.json', 'w') as f:
                json.dump(results, f, indent=2, sort_keys=True)

    torch.distributed.barrier()

#----------------------------------------------------------------------------

@main.command()
@click.option('--data', 'dataset_path', help='Path to the dataset', metavar='PATH|ZIP', type=str, required=True)
@click.option('--dest', 'dest_path',    help='Destination .npz file', metavar='NPZ',    type=str, required=True)
@click.option('--batch',                help='Maximum batch size', metavar='INT',       type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--num', 'num_expected',  help='Number of images to use (optional)', metavar='INT', type=click.IntRange(min=2), default=None, show_default=True)
@click.option('--seed',                 help='Random seed for selecting images (optional)', metavar='INT', type=int, default=0, show_default=True)

def ref(dataset_path, dest_path, batch, num_expected, seed):
    """Calculate dataset reference statistics needed by 'calc' (FID mu/sigma)."""
    torch.multiprocessing.set_start_method('spawn', force=True)
    dist.init()

    mu, sigma = calculate_inception_stats(image_path=dataset_path, num_expected=num_expected, seed=seed, max_batch_size=batch)
    dist.print0(f'Saving dataset reference statistics to "{dest_path}"...')
    if dist.get_rank() == 0:
        if os.path.dirname(dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        np.savez(dest_path, mu=mu, sigma=sigma)

    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
