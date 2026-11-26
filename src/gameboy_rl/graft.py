"""Give a trained network new inputs without losing what it knows.

SB3 joins the per-input features in a fixed order and feeds them to the first linear layers. A new input adds columns
there; we copy the old columns to where they now belong and leave the new ones at zero, so the grafted network behaves
exactly like the old one until training starts using the new input. Optimiser state starts fresh.
"""
from __future__ import annotations

import torch
from gymnasium.spaces.utils import flatdim
from stable_baselines3 import PPO
from stable_baselines3.common.preprocessing import is_image_space

FIRST_LAYERS = ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight")


def _widths(model: PPO) -> dict[str, int]:
    """Features each input contributes, in the order SB3 concatenates them."""
    ex = model.policy.features_extractor.extractors
    out = {}
    for key, module in ex.items():
        space = model.observation_space.spaces[key]
        out[key] = module.features_dim if is_image_space(space) or hasattr(module, "features_dim") else flatdim(space)
    return out


def needs_graft(old: PPO, new: PPO) -> bool:
    return set(old.observation_space.spaces) != set(new.observation_space.spaces)


def graft(old: PPO, new: PPO) -> None:
    """Copy old's weights into new; new may have extra inputs (never fewer) and otherwise the same architecture."""
    missing = set(old.observation_space.spaces) - set(new.observation_space.spaces)
    if missing:
        raise ValueError(f"cannot graft: inputs removed: {sorted(missing)}")
    o, n = old.policy.state_dict(), new.policy.state_dict()
    old_w, new_w = _widths(old), _widths(new)
    old_at, pos = {}, 0
    for k, w in old_w.items():
        old_at[k], pos = pos, pos + w
    out = {}
    for name, target in n.items():
        if name not in o:
            out[name] = target
        elif o[name].shape == target.shape:
            out[name] = o[name].to(target.device, target.dtype)
        elif name in FIRST_LAYERS:
            grown, pos = torch.zeros_like(target), 0
            for k, w in new_w.items():
                if k in old_w:
                    assert old_w[k] == w, f"input {k} changed size"
                    grown[:, pos:pos + w] = o[name][:, old_at[k]:old_at[k] + w].to(target.device, target.dtype)
                pos += w
            out[name] = grown
        else:
            raise ValueError(f"cannot graft {name}: {tuple(o[name].shape)} -> {tuple(target.shape)}")
    new.policy.load_state_dict(out)
