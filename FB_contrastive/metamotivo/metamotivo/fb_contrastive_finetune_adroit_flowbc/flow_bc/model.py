

import typing as tp

import gymnasium
import numpy as np
import pydantic
import torch
from torch.amp import autocast

from ..base_model import load_model
from ..nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig
from ..model import FBModel, FBModelArchiConfig, FBModelConfig


class FBFlowBCModelArchiConfig(FBModelArchiConfig):
    # noise conditioned actor
    actor: NoiseConditionedActorArchiConfig = pydantic.Field(NoiseConditionedActorArchiConfig(), discriminator="name")
    # vector field
    actor_vf: SimpleVectorFieldArchiConfig = SimpleVectorFieldArchiConfig()


class FBFlowBCModelConfig(FBModelConfig):
    name: tp.Literal["FBFlowBCModel"] = "FBFlowBCModel"
    archi: FBFlowBCModelArchiConfig = FBFlowBCModelArchiConfig()

    @property
    def object_class(self):
        return FBFlowBCModel


class FBFlowBCModel(FBModel):
    def __init__(self, obs_space, action_dim, cfg: FBFlowBCModelConfig):
        super().__init__(obs_space, action_dim, cfg)
        # For IDEs
        self.cfg: FBFlowBCModelConfig = cfg

        obs_space = (
            gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(self.cfg.archi.L_dim,), dtype=np.float32)
            if self.cfg.actor_encode_obs
            else self._fw_encoder.output_space
        )
        self._actor_vf = self.cfg.archi.actor_vf.build(obs_space, action_dim)

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            obs = self._fw_encoder(self._normalize(obs))
            obs = self._left_encoder(obs) if self.cfg.actor_encode_obs else obs
            noises = torch.randn((z.shape[0], self.action_dim), device=z.device, dtype=z.dtype)
            actions = self._actor(obs, z, noises)
        return actions

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        del mean  # not used
        return self.actor(obs, z)

    @classmethod
    def load(
        cls,
        path: str,
        device: tp.Optional[str] = None,
        strict: bool = True,
        build_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ) -> "FBFlowBCModel":
        return load_model(path, device, strict=strict, config_class=FBFlowBCModelConfig, build_kwargs=build_kwargs)
