


import torch
from omegaconf import DictConfig
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    MLP,
    ProbabilisticActor,
    SafeModule,
    SafeSequential,
    TanhNormal,
    ValueOperator,
)
from torchrl.trainers.helpers.models import ACTIVATIONS


class IQLNetwork(torch.nn.Module):
    
    def __init__(self, env: GymEnv, cfg: DictConfig, device: str) -> None:
        super().__init__()
        # Value network
        # Compose into TensorDictModules
        mlp_module = MLP(
                out_features=1,
                num_cells=cfg.encoder_layers,
                activation_class=ACTIVATIONS[cfg.activation_class],
            )

        # Compose everything into ValueOperator
        value_net = ValueOperator(
            module=mlp_module,
            in_keys=["observation", "desired_goal", "wall_info"] if cfg.use_wall_info else ["observation", "desired_goal"],
            out_keys=["state_value"]
        )

        # Q-network
        mlp_module = MLP(
                out_features=1,
                num_cells=cfg.encoder_layers,
                activation_class=ACTIVATIONS[cfg.activation_class],
            )
        
        
        q_net = ValueOperator(
            module=mlp_module,
            in_keys=["observation", "desired_goal", "wall_info", "action"] if cfg.use_wall_info else ["observation", "desired_goal", "action"],
            out_keys=["state_action_value"]
        )

        # Actor/policy MLP
        actor_mlp =SafeModule(
            module=MLP(
                out_features=2 * env.action_spec.shape[-1],
                num_cells=cfg.encoder_layers,
                activation_class=ACTIVATIONS[cfg.activation_class],
            ),
            in_keys=["observation", "desired_goal", "wall_info"] if cfg.use_wall_info else ["observation", "desired_goal"],
            out_keys=["state_value"]
        )

        # Map MLP output to location and scale parameters (the latter must be positive)
        actor_extractor = SafeModule(
            module=NormalParamExtractor(scale_lb=0.1),
            in_keys=["state_value"],
            out_keys=["loc", "scale"]
        )
        actor_module = SafeSequential(
            actor_mlp,
            actor_extractor
        )

        # Use ProbabilisticActor to map it to the correct action space
        actor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            spec=env.action_spec,
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": env.action_spec.space.low,
                "high": env.action_spec.space.high,
                "tanh_loc": False,
            },
            default_interaction_type=ExplorationType.DETERMINISTIC,
        )

        self.model = torch.nn.ModuleList(
            [
                actor,
                value_net,
                q_net,
            ]
        ).to(device)
        
        # test forward passes & initialize lazy shapes
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            obs = env.reset().to(device)

            obs["action"] = env.action_spec.sample().to(device)
            
            self.actor(obs[None])
            self.value_net(obs[None])
            self.q_net(obs[None])
        
    @property
    def actor(self):
        return self.model[0]
    
    @property
    def value_net(self):
        return self.model[1]
    
    @property
    def q_net(self):
        return self.model[2]