

import copy
import math
import typing as tp

import numpy as np
import pydantic
import torch
import torch.nn.functional as F
from torch.amp import autocast
from typing import Optional

from .base import BaseConfig
from .base_model import BaseModel, BaseModelConfig, load_model, save_model
from .nn_models import (
    ActorArchiConfig,
    AugmentatorArchiConfig,
    BackwardArchiConfig,
    DrQEncoderArchiConfig,
    ForwardArchiConfig,
    IdentityNNConfig,
    SimpleActorArchiConfig,
    eval_mode,
)
from .normalizers import AVAILABLE_NORMALIZERS, IdentityNormalizerConfig


class FBModelArchiConfig(BaseConfig):
    L_dim: int = 100
    z_dim: int = 100
    norm_z: bool = True
    f: ForwardArchiConfig = pydantic.Field(ForwardArchiConfig(), discriminator="name")
    b: BackwardArchiConfig = pydantic.Field(BackwardArchiConfig(), discriminator="name")
    # Because of the "name" attribute, these two can be chosen between via strings easily
    actor: tp.Union[ActorArchiConfig, SimpleActorArchiConfig] = pydantic.Field(
        SimpleActorArchiConfig(), discriminator="name"
    )
    left_encoder: tp.Union[BackwardArchiConfig, IdentityNNConfig] = pydantic.Field(
        IdentityNNConfig(), discriminator="name"
    )
    # same config used for both the fw and bw rgb encoders
    rgb_encoder: tp.Union[IdentityNNConfig, DrQEncoderArchiConfig] = pydantic.Field(
        IdentityNNConfig(), discriminator="name"
    )
    augmentator: tp.Union[IdentityNNConfig, AugmentatorArchiConfig] = pydantic.Field(
        IdentityNNConfig(), discriminator="name"
    )


class FBModelConfig(BaseModelConfig):
    name: tp.Literal["FBModel"] = "FBModel"

    archi: FBModelArchiConfig = FBModelArchiConfig()
    obs_normalizer: AVAILABLE_NORMALIZERS = pydantic.Field(IdentityNormalizerConfig(), discriminator="name")
    inference_batch_size: int = 500_000
    seq_length: int = 1
    actor_std: float = 0.2
    amp: bool = False
    actor_encode_obs: bool = True

    def build(self, obs_space, action_dim) -> "FBModel":
        return self.object_class(obs_space, action_dim, self)

    @property
    def object_class(self):
        return FBModel


class FBModel(BaseModel):
    def __init__(self, obs_space, action_dim, cfg: FBModelConfig):
        super().__init__(obs_space, action_dim, cfg)
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg: FBModelConfig = cfg
        arch = self.cfg.archi
        self.device = self.cfg.device
        self.amp_dtype = torch.bfloat16

        # create networks
        self._obs_normalizer = self.cfg.obs_normalizer.build(obs_space)
        self._bw_encoder = arch.rgb_encoder.build(obs_space)
        self._augmentator = arch.augmentator.build(obs_space)
        self._fw_encoder = arch.rgb_encoder.build(obs_space)
        self._left_encoder = arch.left_encoder.build(self._fw_encoder.output_space, arch.L_dim)

        self._backward_map = arch.b.build(self._bw_encoder.output_space, arch.z_dim)
        self._forward_map = arch.f.build(self._left_encoder.output_space, arch.z_dim, action_dim)
        self._actor = arch.actor.build(
            self._left_encoder.output_space if self.cfg.actor_encode_obs else self._fw_encoder.output_space, arch.z_dim, action_dim
        )

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    def _prepare_for_train(self) -> None:
        # create TARGET networks
        self._target_backward_map = copy.deepcopy(self._backward_map)
        self._target_forward_map = copy.deepcopy(self._forward_map)
        self._target_left_encoder = copy.deepcopy(self._left_encoder)

    def _normalize(self, obs: torch.Tensor):
        with torch.no_grad(), eval_mode(self._obs_normalizer):
            return self._obs_normalizer(obs)

    @torch.no_grad()
    def backward_map(self, obs: torch.Tensor):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._backward_map(self._bw_encoder(self._normalize(obs)))

    @torch.no_grad()
    def forward_map(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._forward_map(self._left_encoder(self._fw_encoder(self._normalize(obs))), z, action)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor, z: torch.Tensor, std: float):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            obs = self._fw_encoder(self._normalize(obs))
            obs = self._left_encoder(obs) if self.cfg.actor_encode_obs else obs
            return self._actor(obs, z, std)

    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn((size, self.cfg.archi.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def project_z(self, z):
        if self.cfg.archi.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        dist = self.actor(obs, z, self.cfg.actor_std)
        if mean:
            return dist.mean.float()
        return dist.sample().float()

    def reward_inference(
        self,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
        weight: tp.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            batch_size = next_obs.shape[0]
            num_batches = int(np.ceil(batch_size / self.cfg.inference_batch_size))
            z = 0
            wr = reward if weight is None else reward * weight
            for i in range(num_batches):
                start_idx, end_idx = i * self.cfg.inference_batch_size, (i + 1) * self.cfg.inference_batch_size
                next_obs_slice = next_obs[start_idx:end_idx].to(self.device)
                B = self.backward_map(next_obs_slice)
                z += torch.matmul(wr[start_idx:end_idx].to(self.device).T, B)
        return self.project_z(z)

    @classmethod
    def load(
        cls,
        path: str,
        device: tp.Optional[str] = None,
        strict: bool = True,
        build_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ) -> "FBModel":
        return load_model(path, device, strict=strict, config_class=FBModelConfig, build_kwargs=build_kwargs)

    def save(self, output_folder: str) -> None:
        return save_model(output_folder, self, build_kwargs={"obs_space": self.obs_space, "action_dim": self.action_dim})
