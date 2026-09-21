"""Optional color-preserving residual decoder for ODEWorld RAE inference."""

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ColorResidualHead(nn.Module):
    def __init__(self, latent_dim=768, width=32, max_residual=0.35, output_scale=1.0):
        super().__init__()
        if width < 4 or max_residual <= 0 or not 0 < output_scale <= 1:
            raise ValueError('Positive residual bound, output scale in (0,1], and width >= 4 required')
        self.latent_dim = latent_dim
        self.max_residual = float(max_residual)
        self.output_scale = float(output_scale)
        self.patch_projection = nn.Conv2d(latent_dim + 3, width, kernel_size=1)
        self.mid = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(width + 3, width, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.fine = nn.Sequential(
            nn.Conv2d(width + 3, width // 2, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.output = nn.Conv2d(width // 2, 3, kernel_size=3, padding=1)
        self.gate_output = nn.Conv2d(width // 2, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.gate_output.weight)
        nn.init.constant_(self.gate_output.bias, -4.0)

    def forward(self, latent, base_rgb, return_residual=False, return_details=False):
        if latent.ndim != 3 or latent.shape[-1] != self.latent_dim:
            raise ValueError('Expected latent shaped [batch, patches, latent_dim]')
        if base_rgb.ndim != 4 or base_rgb.shape[1] != 3:
            raise ValueError('Expected base RGB shaped [batch, 3, height, width]')
        side = int(math.sqrt(latent.shape[1]))
        if side * side != latent.shape[1]:
            raise ValueError('Expected a square patch grid')
        patch_rgb = F.interpolate(base_rgb, size=(side, side), mode='area')
        features = latent.transpose(1, 2).reshape(latent.shape[0], self.latent_dim, side, side)
        features = self.patch_projection(torch.cat([features, patch_rgb], dim=1))
        mid_size = tuple(max(side, value // 4) for value in base_rgb.shape[-2:])
        features = F.interpolate(features, size=mid_size, mode='bilinear', align_corners=False)
        mid_rgb = F.interpolate(base_rgb, size=mid_size, mode='area')
        features = self.mid(torch.cat([features, mid_rgb], dim=1))
        features = F.interpolate(features, size=base_rgb.shape[-2:], mode='bilinear', align_corners=False)
        fine = self.fine(torch.cat([features, base_rgb], dim=1))
        gate = torch.sigmoid(self.gate_output(fine))
        residual = self.output_scale * self.max_residual * torch.tanh(self.output(fine)) * gate
        corrected = base_rgb + residual
        if return_details:
            return corrected, residual, gate
        if return_residual:
            return corrected, residual
        return corrected


class ColorResidualDecoder(nn.Module):
    def __init__(self, base_decoder, latent_dim=768, width=32, max_residual=0.35, output_scale=1.0):
        super().__init__()
        self.base_decoder = base_decoder.requires_grad_(False)
        self.head = ColorResidualHead(latent_dim, width, max_residual, output_scale)

    def decode(self, latent, return_residual=False, return_details=False):
        with torch.no_grad():
            base_rgb = self.base_decoder.decode(latent)
        return self.head(latent, base_rgb, return_residual=return_residual, return_details=return_details)

    def forward(self, latent):
        return self.decode(latent)


class GoalConditionedColorResidualHead(ColorResidualHead):
    def __init__(self, latent_dim=768, width=32, max_residual=0.35, output_scale=1.0,
                 match_dim=32, temperature=0.1,
                 pos_attention=False, pos_lambda_init=0.1, token_gate=False):
        super().__init__(latent_dim, width, max_residual, output_scale)
        if match_dim < 4 or temperature <= 0:
            raise ValueError('Positive attention temperature and match_dim >= 4 required')
        self.match_projection = nn.Linear(latent_dim, match_dim)
        self.value_projection = nn.Linear(latent_dim, width)
        self.fusion = nn.Conv2d(2 * width, width, kernel_size=1)
        self.temperature = float(temperature)
        # v2 instance conditioning: L1 position-regularized attention and L2 token gate.
        # Both default off so v1 checkpoints keep loading with strict=True.
        self.pos_attention = bool(pos_attention)
        if self.pos_attention:
            if pos_lambda_init <= 0:
                raise ValueError('pos_lambda_init must be positive')
            self.pos_lambda_raw = nn.Parameter(torch.tensor(math.log(math.expm1(pos_lambda_init))))
        self.token_gate = bool(token_gate)
        if self.token_gate:
            self.token_gate_head = nn.Conv2d(width, 1, kernel_size=1)
            nn.init.zeros_(self.token_gate_head.weight)
            nn.init.constant_(self.token_gate_head.bias, 2.0)

    def _position_penalty(self, side, device):
        positions = torch.arange(side, device=device, dtype=torch.float32)
        ys, xs = torch.meshgrid(positions, positions, indexing='ij')
        points = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=1)
        return torch.cdist(points, points) ** 2

    @property
    def pos_lambda(self):
        if not self.pos_attention:
            return None
        return torch.nn.functional.softplus(self.pos_lambda_raw)

    def forward(self, latent, condition_latent, base_rgb, return_residual=False,
                return_details=False, return_attention=False, detach_gate=False,
                return_token_gate=False):
        if condition_latent.ndim != 3 or condition_latent.shape[-1] != self.latent_dim:
            raise ValueError('Expected condition shaped [batch, patches, latent_dim]')
        if condition_latent.shape[0] == 1 and latent.shape[0] != 1:
            condition_latent = condition_latent.expand(latent.shape[0], -1, -1)
        if condition_latent.shape[0] != latent.shape[0]:
            raise ValueError('Condition batch must be one or match latent batch')
        if latent.ndim != 3 or latent.shape[-1] != self.latent_dim:
            raise ValueError('Expected latent shaped [batch, patches, latent_dim]')
        if base_rgb.ndim != 4 or base_rgb.shape[1] != 3:
            raise ValueError('Expected base RGB shaped [batch, 3, height, width]')
        side = int(math.sqrt(latent.shape[1]))
        if side * side != latent.shape[1]:
            raise ValueError('Expected a square patch grid')
        patch_rgb = F.interpolate(base_rgb, size=(side, side), mode='area')
        features = latent.transpose(1, 2).reshape(latent.shape[0], self.latent_dim, side, side)
        features = self.patch_projection(torch.cat([features, patch_rgb], dim=1))
        current_tokens = F.layer_norm(latent, (self.latent_dim,))
        goal_tokens = F.layer_norm(condition_latent, (self.latent_dim,))
        query = F.normalize(self.match_projection(current_tokens), dim=-1)
        key = F.normalize(self.match_projection(goal_tokens), dim=-1)
        attention_logits = query @ key.transpose(1, 2) / self.temperature
        if self.pos_attention:
            attention_logits = attention_logits - self.pos_lambda * self._position_penalty(
                side, attention_logits.device)
        attention = torch.softmax(attention_logits, dim=-1)
        goal_value = attention @ self.value_projection(goal_tokens)
        current_value = self.value_projection(current_tokens)
        condition = F.layer_norm(goal_value - current_value, (goal_value.shape[-1],))
        condition = condition.transpose(1, 2).reshape(latent.shape[0], -1, side, side)
        features = self.fusion(torch.cat([features, condition], dim=1))
        patch_features = features
        mid_size = tuple(max(side, value // 4) for value in base_rgb.shape[-2:])
        features = F.interpolate(features, size=mid_size, mode='bilinear', align_corners=False)
        mid_rgb = F.interpolate(base_rgb, size=mid_size, mode='area')
        features = self.mid(torch.cat([features, mid_rgb], dim=1))
        features = F.interpolate(features, size=base_rgb.shape[-2:], mode='bilinear', align_corners=False)
        fine = self.fine(torch.cat([features, base_rgb], dim=1))
        gate = torch.sigmoid(self.gate_output(fine))
        token_logits = None
        if self.token_gate:
            token_logits = self.token_gate_head(patch_features)
            token_map = F.interpolate(torch.sigmoid(token_logits), size=base_rgb.shape[-2:],
                                      mode='bilinear', align_corners=False)
            gate = gate * token_map
        same_condition = torch.eq(latent, condition_latent).flatten(1).all(dim=1)
        bypass = same_condition.view(-1, 1, 1, 1)
        applied_gate = gate.detach() if detach_gate else gate
        residual = self.output_scale * self.max_residual * torch.tanh(self.output(fine)) * applied_gate
        corrected = torch.where(bypass, base_rgb, base_rgb + residual)
        residual = torch.where(bypass, torch.zeros_like(residual), residual)
        gate = torch.where(bypass, torch.zeros_like(gate), gate)
        if return_attention and return_token_gate:
            return corrected, residual, gate, attention, token_logits
        if return_attention:
            return corrected, residual, gate, attention
        if return_token_gate:
            return corrected, residual, gate, token_logits
        if return_details:
            return corrected, residual, gate
        if return_residual:
            return corrected, residual
        return corrected


class GoalConditionedColorResidualDecoder(nn.Module):
    def __init__(self, base_decoder, latent_dim=768, width=32, max_residual=0.35, output_scale=1.0,
                 pos_attention=False, pos_lambda_init=0.1, token_gate=False):
        super().__init__()
        self.base_decoder = base_decoder.requires_grad_(False)
        self.head = GoalConditionedColorResidualHead(
            latent_dim, width, max_residual, output_scale,
            pos_attention=pos_attention, pos_lambda_init=pos_lambda_init,
            token_gate=token_gate)
    def decode(self, latent, condition_latent=None, return_residual=False, return_details=False,
               return_attention=False, detach_gate=False, return_token_gate=False):
        with torch.no_grad():
            base_rgb = self.base_decoder.decode(latent)
        if condition_latent is None:
            if return_details:
                return base_rgb, torch.zeros_like(base_rgb), torch.zeros_like(base_rgb[:, :1])
            if return_residual:
                return base_rgb, torch.zeros_like(base_rgb)
            return base_rgb
        return self.head(latent, condition_latent, base_rgb, return_residual=return_residual,
                         return_details=return_details, return_attention=return_attention,
                         detach_gate=detach_gate, return_token_gate=return_token_gate)

    def forward(self, latent):
        return self.decode(latent)


class GoalConditionedRuntimeDecoder(nn.Module):
    """Evaluation wrapper that applies the head with a mutable goal condition."""

    def __init__(self, base_decoder, head):
        super().__init__()
        self.base_decoder = base_decoder.requires_grad_(False)
        self.head = head.requires_grad_(False)
        self.condition = None

    def set_condition(self, condition_latent):
        if condition_latent is not None and condition_latent.ndim != 3:
            raise ValueError('Expected condition shaped [batch, patches, latent_dim]')
        self.condition = condition_latent

    def decode(self, latent, **_kwargs):
        with torch.no_grad():
            base_rgb = self.base_decoder.decode(latent)
            if self.condition is None:
                return base_rgb
            condition = self.condition
            if condition.shape[0] == 1 and latent.shape[0] != 1:
                condition = condition.expand(latent.shape[0], -1, -1)
            return self.head(latent, condition, base_rgb)

    def forward(self, latent):
        return self.decode(latent)


def attach_goal_color_residual_decoder(rae, weights, *, expected_sha256,
                                       expected_base_decoder_sha256,
                                       expected_encoder_sha256, width=16,
                                       max_residual=0.35, output_scale=1.0,
                                       pos_attention=False, pos_lambda_init=0.1,
                                       token_gate=False):
    """Attach a verified goal-conditioned head to an already-loaded RAE in place."""
    from safetensors.torch import load_file

    weights = Path(weights)
    if _sha256(weights) != expected_sha256:
        raise ValueError('Goal color residual checkpoint hash mismatch')
    if isinstance(rae.decoder, (ColorResidualDecoder, GoalConditionedColorResidualDecoder,
                                GoalConditionedRuntimeDecoder)):
        raise ValueError('A color residual decoder is already attached')
    if (_module_sha256(rae.decoder) != expected_base_decoder_sha256
            or _module_sha256(rae.encoder) != expected_encoder_sha256):
        raise ValueError('Goal color residual base RAE identity mismatch')
    device = next(rae.decoder.parameters()).device
    head = GoalConditionedColorResidualHead(
        latent_dim=rae.latent_dim, width=width,
        max_residual=max_residual, output_scale=output_scale,
        pos_attention=pos_attention, pos_lambda_init=pos_lambda_init,
        token_gate=token_gate)
    head.load_state_dict(load_file(str(weights), device=str(device)), strict=True)
    wrapper = GoalConditionedRuntimeDecoder(rae.decoder, head).to(device)
    wrapper.eval().requires_grad_(False)
    rae.decoder = wrapper
    return rae


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _module_sha256(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def attach_color_residual_decoder(rae, weights, *, expected_sha256,
                                  expected_base_decoder_sha256,
                                  expected_encoder_sha256, width=16,
                                  max_residual=0.35, output_scale=0.3):
    """Attach a verified residual head to an already-loaded RAE in place."""
    from safetensors.torch import load_file

    weights = Path(weights)
    if _sha256(weights) != expected_sha256:
        raise ValueError('Color residual checkpoint hash mismatch')
    if isinstance(rae.decoder, ColorResidualDecoder):
        raise ValueError('Color residual decoder is already attached')
    if (_module_sha256(rae.decoder) != expected_base_decoder_sha256
            or _module_sha256(rae.encoder) != expected_encoder_sha256):
        raise ValueError('Color residual base RAE identity mismatch')
    device = next(rae.decoder.parameters()).device
    wrapper = ColorResidualDecoder(
        rae.decoder,
        latent_dim=rae.latent_dim,
        width=width,
        max_residual=max_residual,
        output_scale=output_scale,
    ).to(device)
    wrapper.head.load_state_dict(load_file(str(weights), device=str(device)), strict=True)
    wrapper.eval().requires_grad_(False)
    rae.decoder = wrapper
    return rae
