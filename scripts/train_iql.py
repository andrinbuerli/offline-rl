# iql_training.py


from pathlib import Path

import gymnasium_robotics  # pylint: disable=unused-import
import hydra
import minari
import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch.utils.data import DataLoader
from torchrl.data.datasets.minari_data import MinariExperienceReplay
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs import DoubleToFloat, TransformedEnv
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
from torchrl.objectives import IQLLoss, SoftUpdate
from torchrl.record.loggers.wandb import WandbLogger
from torchrl.trainers import Trainer
from torchrl.trainers.helpers.models import ACTIVATIONS
from tqdm.auto import tqdm


@torch.no_grad()
def evaluate_policy(env, policy, cfg, num_eval_episodes=20):
    """Calculate the mean cumulative reward over multiple episodes."""
    episode_rewards = []

    for _ in range(num_eval_episodes):
        eval_td = env.rollout(max_steps=cfg.train.max_episode_steps, policy=policy, auto_cast_to_device=True)
        episode_rewards.append(eval_td["next", "reward"].sum().item())

    return np.mean(episode_rewards)

def collate_fn(batch):
    result = {
        "id": torch.Tensor([x.id for x in batch]),
        "action": torch.stack([torch.as_tensor(x.actions) for x in batch]).float(),
    }
    
    observations = torch.stack([TensorDict(x.observations) for x in batch])
    observations = observations["observation"].float()
    desired_goals = torch.stack([torch.as_tensor(x.desired_goals) for x in batch]).float()
    
    result["observation"] = observations[:, :-1] # remove the last step
    result["desired_goal"] = desired_goals[:, :-1] # remove the last step
        
    # build "next" dict with reward, done, observation
    result["next"] = {}
    
    result["next"]["reward"] = torch.stack([torch.as_tensor(x.rewards) for x in batch]).unsqueeze(-1).float()
    # done as in "terminated" or "truncated"
    result["next"]["done"] = (
        torch.stack([torch.as_tensor(x.terminations) for x in batch]) 
        | torch.stack([torch.as_tensor(x.truncations) for x in batch])
    ).unsqueeze(-1)
    result["next"]["observation"] = observations[:, 1:] # remove the first step
    result["next"]["desired_goal"] = desired_goals[:, 1:] # remove the first step
    
    return TensorDict(result)

@hydra.main(config_path=str((Path(__file__).parent.parent / "configs" / "train").resolve()))
def main(cfg: DictConfig):
    # Initialize Weights & Biases
    wandb.init(entity="andi-mueller-csem-sa", project="offline-rl", mode=cfg.logging.mode, config=dict(cfg))

    # Load dataset and environment
    dataset = minari.load_dataset(cfg.dataset.id)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    base_env = GymEnv(cfg.dataset.env_id, device=cfg.train.device)
    env = TransformedEnv(base_env, DoubleToFloat())
    env.set_seed(cfg.seed)

    # Define models
    obs_dim = cfg.model.obs_dim
    action_dim = cfg.model.action_dim

    # Value network
    value_net = ValueOperator(
        module=MLP(
            out_features=1, 
            num_cells=cfg.model.encoder_layers,
            activation_class=ACTIVATIONS[cfg.model.activation_class],
        ),
        in_keys=["observation", "desired_goal"],
        out_keys=["state_value"],
    )

    # Q-network
    q_net = ValueOperator(
        module=MLP(
            out_features=1, 
            num_cells=cfg.model.encoder_layers,
            activation_class=ACTIVATIONS[cfg.model.activation_class],
        ),
        in_keys=["observation", "desired_goal", "action"],
        out_keys=["state_action_value"],
    )

    # Actor/policy MLP
    actor_mlp = MLP(
            out_features=2 * action_dim, 
            num_cells=cfg.model.encoder_layers,
            activation_class=ACTIVATIONS[cfg.model.activation_class],
        )

    # Map MLP output to location and scale parameters (the latter must be positive)
    actor_extractor = NormalParamExtractor(scale_lb=0.1)
    actor_net = torch.nn.Sequential(actor_mlp, actor_extractor)

    # Specify tensordict inputs and outputs
    actor_module = TensorDictModule(
        actor_net,
        in_keys=["observation", "desired_goal"],
        out_keys=["loc", "scale"]
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

    model = torch.nn.ModuleList(
        [
            actor,
            value_net,
            q_net,
        ]
    ).to(cfg.train.device)
    
    # test forward passes
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        obs = env.reset().to(cfg.train.device)

        obs["action"] = env.action_spec.sample().to(cfg.train.device)
        #obs["observation"] = torch.concat(
        #    [obs["observation"], obs["desired_goal"]], dim=-1
        #)
        
        for net in model:
            net(obs[None])

    # IQL loss
    loss_module = IQLLoss(
        actor_network=model[0],
        value_network=model[1],
        qvalue_network=model[2],
        loss_function="l2",
        temperature=cfg.model.temperature,
        expectile=cfg.model.expectile,
    )
    loss_module.make_value_estimator(gamma=cfg.train.gamma)
    target_net_updater = SoftUpdate(loss_module, tau=0.005)

    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.train.lr)

    loss_logs = []
    eval_reward_logs = []

    step = 0
    
    for epoch in (pbar := tqdm(range(cfg.train.epochs), total=cfg.train.epochs, desc="Training")):
        for batch in dataloader:
            # 2) Compute loss l = L_V + L_Q + L_pi
            loss_dict = loss_module(batch.to(cfg.train.device))
            loss = loss_dict["loss_value"] + loss_dict["loss_qvalue"] + loss_dict["loss_actor"]
            loss_logs.append(loss.item())

            # 3) Backpropagate the gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # Update V(s), Q(a, s), pi(a|s)
            target_net_updater.step()  # Update the target Q-network

            if step % cfg.train.log_every == 0:
                wandb.log(
                    {
                        "loss": loss.item(),
                        "loss_value": loss_dict["loss_value"].item(),
                        "loss_qvalue": loss_dict["loss_qvalue"].item(),
                        "loss_actor": loss_dict["loss_actor"].item(),
                    },
                    step=step,
                )
            step += 1
        
        # Evaluate the policy
        eval_reward_logs.append(evaluate_policy(env, model[0], cfg))
        pbar.set_description(
            f"Loss: {loss_logs[-1]:.1f}, Avg return: {eval_reward_logs[-1]:.1f}"
        )


if __name__ == "__main__":
    main()
