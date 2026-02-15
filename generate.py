# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist

# Our Divergence Algorithm
def divergence_stepper( v_func,
                        v_func_kwargs,
                        x_key='z',
                        t_key='t',
                        stop_t=0.5,
                        num_updates=1,
                        num_delta=1,
                        num_eps=1,
                        delta_scale=1,
                        delta_scheduler=lambda t: 1.0,
                        seed_delta=None,
                        seed_eps=None,
                        resample_delta=False,
                        resample_eps=False,
                        sequential_vjp=True,
                        sequential_hutchinson=True,
                        ):
    assert stop_t >= 0.0 and stop_t <= 1.0

    t = v_func_kwargs[t_key]
    z = v_func_kwargs[x_key]
    T = 80
    if isinstance(t, torch.Tensor):
        assert (t == t.mean()).all().item(), "All timesteps in the batch must be the same for divergence_stepper."
        t = t.mean().item()
        t =  1. - t/T # t is implemented as flow-based modeling in this function
    # import ipdb; ipdb.set_trace()
    if num_updates <= 0 or t > stop_t:
        return v_func_kwargs[x_key], v_func(**v_func_kwargs)

    # z = v_func_kwargs[x_key]        
    B = z.shape[0]
    D = np.prod(z.shape[1:])  # C * H * W
    
    delta_generator = None
    eps_generator = None
    
    if seed_delta is not None:
        delta_generator = torch.Generator(device=z.device).manual_seed(seed_delta)
    if seed_eps is not None:
        eps_generator = torch.Generator(device=z.device).manual_seed(seed_eps)
    sync_eps_with_delta = num_eps == 1 and seed_eps == seed_delta
    
    for update_idx in range(num_updates):
        require_sample_delta = (update_idx == 0) or resample_delta
        require_sample_eps = (update_idx == 0) or resample_eps

        # compute divergence and find the best perturbation
        if sequential_vjp:
            assert (not resample_delta) or (num_delta==1)
            for delta_idx in range(num_delta+1):

                # pass if no need to get the divergence of original z
                if delta_idx == 0 and update_idx != 0:
                    continue

                delta = torch.randn(z.shape, generator=delta_generator, device=z.device) if delta_idx != 0 else torch.zeros_like(z, device=z.device)

                assert seed_delta != seed_eps, f"Is a Biased Estimator, {seed_delta} {seed_eps}"

                if sync_eps_with_delta and delta_idx != 0:
                    eps = delta.detach()

                else:
                    eps = torch.randn(z.shape, generator=eps_generator, device=z.device) 

                perturbed_z = z + delta_scale * delta_scheduler(t) * delta # TODO: clarify
                with torch.enable_grad():
                    perturbed_z = perturbed_z.detach().requires_grad_(True)
                    v_func_kwargs[x_key] = perturbed_z
                    
                    v_pred = v_func(**v_func_kwargs)  # [B, C, H, W]
                    v_pred_eps = (v_pred * eps).flatten(1).sum(1)  # [B]
                    grad_v = torch.autograd.grad(
                        outputs=v_pred_eps,          # [B]
                        inputs=perturbed_z,                      # [B, C, H, W]
                        grad_outputs=torch.ones_like(v_pred_eps),  # [B]
                        create_graph=False,
                        retain_graph=False,         
                    )[0].detach()  # [B, C, H, W]
                    divergence = (grad_v * eps).flatten(1).sum(1) / D  # [B]
                
                if delta_idx == 0:
                    best_divergence = divergence.detach()
                    best_v_pred = v_pred.detach()
                    best_perturbed_z = perturbed_z.detach()
                else:
                    improved = divergence < best_divergence
                    improved_shape = (B,) + (1,) * (len(z.shape) - 1)
                    best_divergence = torch.where(improved, divergence, best_divergence)
                    best_v_pred = torch.where(
                        improved.view(improved_shape),
                        v_pred,
                        best_v_pred,
                    )
                    best_perturbed_z = torch.where(
                        improved.view(improved_shape),
                        perturbed_z.detach(),
                        best_perturbed_z,
                    )

            # update iteration-wise
            z = best_perturbed_z.detach() # update z
        
        # currently not using hereafter
        else:
            # build delta
            raise NotImplementedError
            if require_sample_delta:            
                delta_shape = (num_delta+1, ) + z.shape  # [num_delta, B, C, H, W]
                delta = torch.randn(delta_shape, generator=delta_generator, device=z.device, dtype=z.dtype)  # [num_delta, B, C, H, W]
                delta[0] = 0.0  # no perturbation for the first sample
            
            # build eps.
            if require_sample_eps:
                if sync_eps_with_delta:
                    eps = delta.unsqueeze(0)  # [1, num_delta * B, C, H, W]
                    # eps = repeat(eps, '1 nd b ... -> (ne nd b) ...', ne=num_eps)
                else:
                    eps_shape = (num_eps,) + z.shape  # [num_eps, B, C, H, W]
                    eps = torch.randn(eps_shape, generator=eps_generator, device=z.device, dtype=z.dtype)  # [num_eps, B, C, H, W]
                    # eps_shape = (num_eps, num_delta+1,) + z.shape # Tip: sample more and rearrange for independent eps.
            
            # expand v_func_kwargs
            perturbed_z = z.unsqueeze(0) + delta_scale * delta_scheduler(t) * delta
            perturbed_z = rearrange(perturbed_z, 'nd b ... -> (nd b) ...', nd=num_delta+1, b=B)
            perturbed_z = perturbed_z.detach().requires_grad_(True)
            
            # compute v
            with torch.enable_grad():            
                v_func_kwargs[x_key] = perturbed_z
                v_func_kwargs_expanded = expand_v_func_kwargs(v_func_kwargs, batch_size=B, expand_size=num_delta+1)
                v_pred_expanded = v_func(**v_func_kwargs_expanded)  # [(num_delta+1) * B, C, H, W]

                divergence = []
                if sequential_hutchinson:
                    for eps_idx, eps_i in enumerate(eps):  # [B, C, H, W]
                        retain_graph = (eps_idx < eps.shape[0] - 1) # retain graph except for the last one
                        v_pred_eps = (v_pred_expanded * eps_i.unsqueeze(0)).flatten(1).sum(1)  # [(num_delta+1) * B]
                        grad_v = torch.autograd.grad(
                            outputs=v_pred_eps,          # [(num_delta+1) * B]
                            inputs=perturbed_z,                      # [(num_delta+1) * B, C, H, W]
                            grad_outputs=torch.ones_like(v_pred_eps),  # [(num_delta+1) * B]
                            create_graph=False,
                            retain_graph=retain_graph,         
                        )[0].detach()  # [(num_delta+1) * B, C, H, W]
                        
                        divergence_i = (grad_v * eps_i.unsqueeze(0)).flatten(1).sum(1) / D  # [(num_delta+1) * B]
                        divergence.append(divergence_i)
                    divergence = torch.stack(divergence, dim=0)  # [num_eps, (num_delta+1) * B]
                    divergence = divergence.mean(0)  # [(num_delta+1) * B]
                else:
                    v_pred_eps = (v_pred_expanded.unsqueeze(0) * eps.flatten(1).unsqueeze(1)).flatten(2).sum(2)  # [num_eps, (num_delta+1) * B]
                    grad_v = torch.autograd.grad(
                        outputs=v_pred_eps,          # [num_eps, (num_delta+1) * B]
                        inputs=perturbed_z,                      # [(num_delta+1) * B, C, H, W]
                        grad_outputs=torch.ones_like(v_pred_eps),  # [num_eps, (num_delta+1) * B]
                        create_graph=False,
                        retain_graph=False,         
                    )[0].detach()  # [(num_delta+1) * B, C, H, W]
                    
                    divergence = (grad_v.unsqueeze(0) * eps.flatten(1).unsqueeze(1)).flatten(2).sum(2) / D  # [num_eps, (num_delta+1) * B]
                    divergence = divergence.mean(0)  # [(num_delta+1) * B]
            # select the best perturbation based on divergence
            divergence = divergence.view(num_delta+1, B)  # [num_delta+1, B]
            best_divergence, best_idx = torch.min(divergence, dim=0)
            best_perturbed_z = rearrange(perturbed_z, '(nd b) ... -> nd b ...', nd=num_delta+1, b=B)[best_idx, torch.arange(B)]  # [B, C, H, W]
            best_v_pred = rearrange(v_pred_expanded, '(nd b) ... -> nd b ...', nd=num_delta+1, b=B)[best_idx, torch.arange(B)]  # [B, C, H, W]
            z = best_perturbed_z.detach()
    return best_perturbed_z.detach(), best_v_pred.detach()


#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


#----------------------------------------------------------------------------
def edm_v_pred(
    net, x_cur, t_cur, t_next, i, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, S_noise=1, gamma=0.0
):

    t_hat = net.round_sigma(t_cur + gamma * t_cur) # t_cur
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur) # x_cur

    # Euler step.
    denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
    d_cur = (x_hat - denoised) / t_hat
    # d_cur corresponds to v
    x_next = x_hat + (t_next - t_hat) * d_cur

    # Apply 2nd order correction.
    if i < num_steps - 1:
        denoised = net(x_next, t_next, class_labels).to(torch.float64)
        d_prime = (x_next - denoised) / t_next
        v_pred = 0.5 * d_cur + 0.5 * d_prime
    else:
        v_pred = d_cur

    return v_pred

def edm_x_next(
    x_hat, t_next, t_hat, v_pred
):
    x_next = x_hat + (t_next - t_hat) * v_pred
    return x_next

## ours
def edm_sampler_ours(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, num_iter=2, seed=42, lr=0.5
):
    SEED = seed

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        # NOTE: gamma is zero in official CIFAR-10 implementation
        # therefore, we treat t_hat as t_cur
        # Should be fixed when running with ImageNet

        # v_pred -> ours algo -> x_pred (latent update)
        v_pred_func = edm_v_pred
        v_func_kwargs = {
            'net': net,
            'x_cur': x_cur,
            't_cur': t_cur,
            't_next': t_next,
            'i': i,
            'class_labels': class_labels,
            'randn_like': randn_like,
            'num_steps': num_steps,
            'S_noise': S_noise,
            'gamma': gamma
        }

        # import ipdb; ipdb.set_trace()

        best_x, best_v_pred = divergence_stepper(v_pred_func,
                                                v_func_kwargs,
                                                x_key='x_cur',
                                                t_key='t_cur', 
                                                stop_t=0.5,
                                                delta_scale=2.* (t_cur/80), # 0.5
                                                num_updates=1,
                                                seed_delta=1234,
                                                seed_eps=42)
        
        x_next = edm_x_next(best_x, t_next, t_cur, best_v_pred).detach()
    return x_next

#----------------------------------------------------------------------------
def euler_v_pred(
    net, x_cur, t_cur, t_next, i, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, S_noise=1, gamma=0.0
):

    t_hat = net.round_sigma(t_cur + gamma * t_cur) # t_cur
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur) # x_cur

    # Euler step.
    denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
    d_cur = (x_hat - denoised) / t_hat
    # d_cur corresponds to v
    v_pred = d_cur

    return v_pred

def euler_x_next(
    x_hat, t_next, t_hat, v_pred
):
    x_next = x_hat + (t_next - t_hat) * v_pred
    return x_next

## ours
def euler_sampler_ours(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, num_iter=2, seed=42, lr=0.5
):
    SEED = seed

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        # NOTE: gamma is zero in official CIFAR-10 implementation
        # therefore, we treat t_hat as t_cur
        # Should be fixed when running with ImageNet

        # v_pred -> ours algo -> x_pred (latent update)
        v_pred_func = euler_v_pred
        v_func_kwargs = {
            'net': net,
            'x_cur': x_cur,
            't_cur': t_cur,
            't_next': t_next,
            'i': i,
            'class_labels': class_labels,
            'randn_like': randn_like,
            'num_steps': num_steps,
            'S_noise': S_noise,
            'gamma': gamma
        }

        best_x, best_v_pred = divergence_stepper(v_pred_func,
                                                v_func_kwargs,
                                                x_key='x_cur',
                                                t_key='t_cur',
                                                stop_t=0.5,
                                                delta_scale=0.1,
                                                num_updates=1,
                                                seed_delta=1234,
                                                seed_eps=42
                                                )
        
        x_next = euler_x_next(best_x, t_next, t_cur, best_v_pred)
    return x_next

#----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.

def ablation_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',  'exp_name',     help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

# sampler options
@click.option('--sampler', 'sampler_type', help='Sampler type', metavar='ours|default',                              type=click.Choice(['ours', 'default']), default='ours')
@click.option('--sampler_lr',              help='x_t update rate', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=0.5)
@click.option('--sampler_iter',            help='Number of iterations for each step', metavar='INT',              type=click.IntRange(min=1), default=2)


@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

def main(network_pkl, exp_name, subdirs, seeds, class_idx, max_batch_size, sampler_lr, sampler_iter, device=torch.device('cuda'), sampler_type='ours', **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    outdir = os.path.join('results', exp_name)

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "results/{exp_name}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
        # sampler_fn = ablation_sampler if have_ablation_kwargs else edm_sampler
        # if sampler_kwargs.get('ours', False):
        if sampler_type == 'ours':
            sampler_fn = edm_sampler_ours
            images = sampler_fn(net, latents, class_labels, randn_like=rnd.randn_like, lr=sampler_lr, num_iter=sampler_iter, **sampler_kwargs)
        else:
            sampler_fn = edm_sampler
            images = sampler_fn(net, latents, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)

        # Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
 