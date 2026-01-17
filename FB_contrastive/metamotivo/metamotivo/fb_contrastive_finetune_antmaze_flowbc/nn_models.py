# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import math
import numbers
import typing as tp

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn
from torch.distributions.utils import _standard_normal
from typing import Optional

from .base import BaseConfig

##########################
# Initialization utils
##########################


# Initialization for parallel layers
def parallel_orthogonal_(tensor, gain=1):
    if tensor.ndimension() == 2:
        tensor = nn.init.orthogonal_(tensor, gain=gain)
        return tensor
    if tensor.ndimension() < 3:
        raise ValueError("Only tensors with 3 or more dimensions are supported")
    n_parallel = tensor.size(0)
    rows = tensor.size(1)
    cols = tensor.numel() // n_parallel // rows
    flattened = tensor.new(n_parallel, rows, cols).normal_(0, 1)

    qs = []
    for flat_tensor in torch.unbind(flattened, dim=0):
        if rows < cols:
            flat_tensor.t_()

        # Compute the qr factorization
        q, r = torch.linalg.qr(flat_tensor)
        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph

        if rows < cols:
            q.t_()
        qs.append(q)

    qs = torch.stack(qs, dim=0)
    with torch.no_grad():
        tensor.view_as(qs).copy_(qs)
        tensor.mul_(gain)
    return tensor


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, DenseParallel):
        gain = nn.init.calculate_gain("relu")
        parallel_orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif hasattr(m, "reset_parameters"):
        m.reset_parameters()


##########################
# Update utils
##########################


def _soft_update_params(net_params: tp.Any, target_net_params: tp.Any, tau: float):
    torch._foreach_mul_(target_net_params, 1 - tau)
    torch._foreach_add_(target_net_params, net_params, alpha=tau)


class eval_mode:
    def __init__(self, *models) -> None:
        self.models = models
        self.prev_states = []

    def __enter__(self) -> None:
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args) -> None:
        for model, state in zip(self.models, self.prev_states):
            model.train(state)


##########################
# Creation utils
##########################


class ForwardArchiConfig(BaseConfig):
    name: tp.Literal["ForwardArchi"] = "ForwardArchi"
    hidden_dim: int = 1024
    model: tp.Literal["simple"] = "simple"
    hidden_layers: int = 1
    embedding_layers: int = 2
    num_parallel: int = 2
    ensemble_mode: tp.Literal["batch"] = "batch"

    def build(self, obs_space, z_dim: int, action_dim, output_dim=None) -> torch.nn.Module:
        """Note: Forward model is also used for critics"""

        if self.ensemble_mode == "batch":
            return _build_batch_forward(self, obs_space, z_dim, action_dim, output_dim)
        else:
            raise ValueError(f"Unsupported ensemble_mode {self.ensemble_mode}")


def _build_batch_forward(cfg, obs_space, z_dim, action_dim, output_dim=None):
    if cfg.model == "simple":
        forward_cls = ForwardMap
    else:
        raise ValueError(f"Unsupported forward_map model {cfg.model}")
    return forward_cls(obs_space, z_dim, action_dim, cfg, output_dim=output_dim)


class ActorArchiConfig(BaseConfig):
    name: tp.Literal["actor"] = "actor"
    model: tp.Literal["simple"] = "simple"
    hidden_dim: int = 1024
    hidden_layers: int = 1
    embedding_layers: int = 2

    def build(self, obs_space, z_dim, action_dim):
        if self.model == "simple":
            return Actor(obs_space, z_dim, action_dim, self)
        else:
            raise ValueError(f"Unsupported actor model {self.model}. Define 'model' or use other configs explicitely")


def linear(input_dim, output_dim, num_parallel=1):
    if num_parallel > 1:
        return DenseParallel(input_dim, output_dim, n_parallel=num_parallel)
    return nn.Linear(input_dim, output_dim)


def layernorm(input_dim, num_parallel=1):
    if num_parallel > 1:
        return ParallelLayerNorm([input_dim], n_parallel=num_parallel)
    return nn.LayerNorm(input_dim)


##########################
# Simple MLP models
##########################


class BackwardArchiConfig(BaseConfig):
    name: tp.Literal["BackwardArchi"] = "BackwardArchi"
    hidden_dim: int = 256
    hidden_layers: int = 2
    norm: bool = True

    def build(self, obs_space, z_dim: int):
        return BackwardMap(obs_space, z_dim, self)


class BackwardMap(nn.Module):
    def __init__(self, obs_space, z_dim, cfg: BackwardArchiConfig) -> None:
        super().__init__()
        self.cfg: BackwardArchiConfig = cfg

        self.output_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(z_dim,), dtype=np.float32)

        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        seq = [nn.Linear(obs_space.shape[0], cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.Tanh()]
        for _ in range(cfg.hidden_layers - 1):
            seq += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
        seq += [nn.Linear(cfg.hidden_dim, z_dim)]
        if cfg.hidden_layers == 0:
            seq = [nn.Linear(obs_space.shape[0], z_dim)]
        if cfg.norm:
            seq += [Norm()]
        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def simple_embedding(input_dim, hidden_dim, hidden_layers, num_parallel=1):
    assert hidden_layers >= 2, "must have at least 2 embedding layers"
    seq = [linear(input_dim, hidden_dim, num_parallel), layernorm(hidden_dim, num_parallel), nn.Tanh()]
    for _ in range(hidden_layers - 2):
        seq += [linear(hidden_dim, hidden_dim, num_parallel), nn.ReLU()]
    seq += [linear(hidden_dim, hidden_dim // 2, num_parallel), nn.ReLU()]
    return nn.Sequential(*seq)


class ForwardMap(nn.Module):
    def __init__(
        self,
        obs_space,
        z_dim,
        action_dim,
        cfg: ForwardArchiConfig,
        output_dim=None,
    ) -> None:
        super().__init__()

        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        obs_dim = obs_space.shape[0]
        self.cfg = cfg
        self.z_dim = z_dim
        self.num_parallel = cfg.num_parallel
        self.hidden_dim = cfg.hidden_dim

        self.embed_z = simple_embedding(obs_dim + z_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel)
        self.embed_sa = simple_embedding(obs_dim + action_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel)

        seq = []
        for _ in range(cfg.hidden_layers):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim, cfg.num_parallel), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, output_dim if output_dim else z_dim, cfg.num_parallel)]
        self.Fs = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
            action = action.expand(self.num_parallel, -1, -1)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))  # num_parallel x bs x h_dim // 2
        sa_embedding = self.embed_sa(torch.cat([obs, action], dim=-1))  # num_parallel x bs x h_dim // 2
        return self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))


class SimpleActorArchiConfig(ActorArchiConfig):
    name: tp.Literal["simple"] = "simple"
    model: tp.Literal["simple"] = "simple"

    def build(self, obs_space, z_dim: int, action_dim: int) -> "Actor":
        return Actor(obs_space, z_dim, action_dim, self)


class Actor(nn.Module):
    def __init__(self, obs_space, z_dim, action_dim, cfg: SimpleActorArchiConfig) -> None:
        super().__init__()

        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        obs_dim = obs_space.shape[0]
        self.cfg: SimpleActorArchiConfig = cfg

        self.embed_z = simple_embedding(obs_dim + z_dim, cfg.hidden_dim, cfg.embedding_layers)
        self.embed_s = simple_embedding(obs_dim, cfg.hidden_dim, cfg.embedding_layers)

        seq = []
        for _ in range(cfg.hidden_layers):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, action_dim)]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z, std):
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))  # bs x h_dim // 2
        s_embedding = self.embed_s(obs)  # bs x h_dim // 2
        embedding = torch.cat([s_embedding, z_embedding], dim=-1)
        mu = torch.tanh(self.policy(embedding))
        std = torch.ones_like(mu) * std
        dist = TruncatedNormal(mu, std)
        return dist


class VForwardArchiConfig(BaseConfig):
    hidden_dim: int = 1024
    hidden_layers: int = 1
    embedding_layers: int = 2
    num_parallel: int = 2

    def build(self, obs_space, z_dim: int, output_dim=None) -> torch.nn.Module:
        return VForwardMap(obs_space, z_dim, output_dim, self)


class VForwardMap(nn.Module):
    def __init__(
        self,
        obs_space,
        z_dim,
        output_dim=None,
        cfg: VForwardArchiConfig = VForwardArchiConfig(),
    ) -> None:
        super().__init__()

        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        obs_dim = obs_space.shape[0]
        self.z_dim = z_dim
        self.num_parallel = cfg.num_parallel
        self.hidden_dim = cfg.hidden_dim

        self.embed_z = simple_embedding(obs_dim + z_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel)
        self.embed_s = simple_embedding(obs_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel)

        seq = []
        for _ in range(cfg.hidden_layers):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim, cfg.num_parallel), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, output_dim if output_dim else z_dim, cfg.num_parallel)]
        self.Fs = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))  # num_parallel x bs x h_dim // 2
        s_embedding = self.embed_s(obs)  # num_parallel x bs x h_dim // 2
        return self.Fs(torch.cat([s_embedding, z_embedding], dim=-1))


##########################
# Visual modules
##########################


class DrQEncoderArchiConfig(BaseConfig):
    name: tp.Literal["drq"] = "drq"
    feature_dim: tp.Optional[int] = None  # if not None, linearly project the output to feature_dim

    def build(self, obs_space):
        return DrQEncoder(obs_space, self)


class DrQEncoder(nn.Module):
    """RGB encoder from the DrQ-v2 paper"""

    def __init__(self, obs_space, cfg: DrQEncoderArchiConfig) -> None:
        super().__init__()
        self.cfg = cfg

        assert len(obs_space.shape) == 3, "obs_space must have a 3D shape (image)"

        # courtesy of https://github.com/facebookresearch/drqv2/blob/main/drqv2.py
        self.trunk = nn.Sequential(
            nn.Conv2d(obs_space.shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            self.repr_dim = np.prod(self.trunk(torch.zeros(1, *obs_space.shape)).shape)

        if self.cfg.feature_dim is not None:
            self.proj = nn.Sequential(nn.Linear(self.repr_dim, self.cfg.feature_dim), nn.LayerNorm(self.cfg.feature_dim), nn.Tanh())
            self.repr_dim = self.cfg.feature_dim
        else:
            self.proj = nn.Identity()
            print(
                "WARNING: using a DrQ encoder with feature_dim=None. This yields very large feature vectors that are fed as input to other networks"
            )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.proj(self.trunk(obs))

    @property
    def output_space(self):
        return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(self.repr_dim,), dtype=np.float32)


class AugmentatorArchiConfig(BaseConfig):
    name: tp.Literal["random_shifts"] = "random_shifts"
    pad: int = 4

    def build(self, obs_space):
        return Augmentator(obs_space, self)


class Augmentator(nn.Module):
    """Image augmentations from DrQ-v2"""

    def __init__(self, obs_space, cfg: AugmentatorArchiConfig) -> None:
        super().__init__()
        self.cfg = cfg

        assert len(obs_space.shape) == 3, "obs_space must have a 3D shape (image)"

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        n, _, h, w = obs.size()
        assert h == w, "Augmentator only supports square images"
        padding = tuple([self.cfg.pad] * 4)
        obs = F.pad(obs, padding, "replicate")
        eps = 1.0 / (h + 2 * self.cfg.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.cfg.pad, device=obs.device, dtype=obs.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.cfg.pad + 1, size=(n, 1, 1, 2), device=obs.device, dtype=obs.dtype)
        shift *= 2.0 / (h + 2 * self.cfg.pad)
        grid = base_grid + shift
        return F.grid_sample(obs, grid, padding_mode="zeros", align_corners=False)


##########################
# FlowQ modules
##########################


class NoiseConditionedActorArchiConfig(BaseConfig):
    name: tp.Literal["noise_conditioned_actor"] = "noise_conditioned_actor"
    model: tp.Literal["simple"] = "simple"
    hidden_dim: int = 1024
    hidden_layers: int = 1
    embedding_layers: int = 2

    def build(self, obs_space, z_dim: int, action_dim: int) -> "NoiseConditionedActor":
        return NoiseConditionedActor(obs_space, z_dim, action_dim, self)


class NoiseConditionedActor(nn.Module):
    def __init__(self, obs_space, z_dim, action_dim, cfg: NoiseConditionedActorArchiConfig) -> None:
        super().__init__()

        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        obs_dim = obs_space.shape[0]
        self.cfg: NoiseConditionedActorArchiConfig = cfg
        self.embed_z = simple_embedding(obs_dim + z_dim + action_dim, cfg.hidden_dim, cfg.embedding_layers)
        self.embed_s = simple_embedding(obs_dim + action_dim, cfg.hidden_dim, cfg.embedding_layers)

        seq = []
        for _ in range(cfg.hidden_layers):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, action_dim)]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        z_embedding = self.embed_z(torch.cat([obs, z, noise], dim=-1))  # bs x h_dim // 2
        s_embedding = self.embed_s(torch.cat([obs, noise], dim=-1))  # bs x h_dim // 2
        embedding = torch.cat([s_embedding, z_embedding], dim=-1)
        actions = torch.tanh(self.policy(embedding))
        return actions


##########################
# Helper modules
##########################


class DenseParallel(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_parallel: int,
        bias: bool = True,
        device=None,
        dtype=None,
        reset_params=True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(DenseParallel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_parallel = n_parallel
        if n_parallel is None or (n_parallel == 1):
            self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
        else:
            self.weight = nn.Parameter(torch.empty((n_parallel, in_features, out_features), **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty((n_parallel, 1, out_features), **factory_kwargs))
            else:
                self.register_parameter("bias", None)
            if self.bias is None:
                raise NotImplementedError
        if reset_params:
            self.reset_parameters()

    def load_module_list_weights(self, module_list) -> None:
        with torch.no_grad():
            assert len(module_list) == self.n_parallel
            weight_list = [m.weight.T for m in module_list]
            target_weight = torch.stack(weight_list, dim=0)
            self.weight.data.copy_(target_weight.data)
            if self.bias:
                bias_list = [ln.bias.unsqueeze(0) for ln in module_list]
                target_bias = torch.stack(bias_list, dim=0)
                self.bias.data.copy_(target_bias.data)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.n_parallel is None or (self.n_parallel == 1):
            return F.linear(input, self.weight, self.bias)
        else:
            return torch.baddbmm(self.bias, input, self.weight)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, n_parallel={}, bias={}".format(
            self.in_features, self.out_features, self.n_parallel, self.bias is not None
        )


class ParallelLayerNorm(nn.Module):
    def __init__(self, normalized_shape, n_parallel, eps=1e-5, elementwise_affine=True, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(ParallelLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = [
                normalized_shape,
            ]
        assert len(normalized_shape) == 1
        self.n_parallel = n_parallel
        self.normalized_shape = list(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            if n_parallel is None or (n_parallel == 1):
                self.weight = nn.Parameter(torch.empty([*self.normalized_shape], **factory_kwargs))
                self.bias = nn.Parameter(torch.empty([*self.normalized_shape], **factory_kwargs))
            else:
                self.weight = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs))
                self.bias = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def load_module_list_weights(self, module_list) -> None:
        with torch.no_grad():
            assert len(module_list) == self.n_parallel
            if self.elementwise_affine:
                ln_weights = [ln.weight.unsqueeze(0) for ln in module_list]
                ln_biases = [ln.bias.unsqueeze(0) for ln in module_list]
                target_ln_weights = torch.stack(ln_weights, dim=0)
                target_ln_bias = torch.stack(ln_biases, dim=0)
                self.weight.data.copy_(target_ln_weights.data)
                self.bias.data.copy_(target_ln_bias.data)

    def forward(self, input):
        norm_input = F.layer_norm(input, self.normalized_shape, None, None, self.eps)
        if self.elementwise_affine:
            return (norm_input * self.weight) + self.bias
        else:
            return norm_input

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}".format(**self.__dict__)


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6) -> None:
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps
        self.noise_upper_limit = high - self.loc
        self.noise_lower_limit = low - self.loc

    def _clamp(self, x) -> torch.Tensor:
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()) -> torch.Tensor:  # type: ignore
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class Norm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x) -> torch.Tensor:
        return math.sqrt(x.shape[-1]) * F.normalize(x, dim=-1)


class IdentityNNConfig(BaseConfig):
    name: tp.Literal["Identity"] = "Identity"

    def build(self, obs_space, *args) -> nn.Module:
        return IdentityNN(obs_space)


class IdentityNN(nn.Identity):
    def __init__(self, obs_space):
        super().__init__()
        self.obs_space = obs_space

    @property
    def output_space(self):
        return self.obs_space


class SimpleVectorFieldArchiConfig(BaseConfig):
    # name: tp.Literal["vector_field"] = "vector_field"
    model: tp.Literal["simple"] = "simple"
    hidden_dim: int = 1024
    hidden_layers: int = 1

    def build(self, obs_space, action_dim: int) -> "VectorField":
        return VectorField(obs_space, action_dim, self)


class VectorField(nn.Module):
    def __init__(self, obs_space, action_dim, cfg: SimpleVectorFieldArchiConfig) -> None:
        super().__init__()
        self.cfg: SimpleVectorFieldArchiConfig = cfg

        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        obs_dim = obs_space.shape[0]
        # plus 1 is for time
        seq = [linear(obs_dim + action_dim + 1, cfg.hidden_dim), nn.GELU()]
        for _ in range(cfg.hidden_layers - 1):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU()]
        seq += [linear(cfg.hidden_dim, action_dim)]
        self.net = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        embedding = torch.cat([obs, action, t], dim=-1)
        return self.net(embedding)