# iql_training.py


from pathlib import Path
from typing import Iterable

import gymnasium_robotics  # pylint: disable=unused-import
import hydra
import minari
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchrl.envs import DoubleToFloat, TransformedEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.objectives import CQLLoss, IQLLoss, SoftUpdate
from tqdm.auto import tqdm

import wandb
from offline_rl.nn.iql import IQLNetwork
from offline_rl.replay_buffer import LocalMinariReplayBuffer


@torch.no_grad()
def evaluate_policy(env, policy, cfg):
    """Calculate the mean cumulative reward over multiple episodes."""
    episode_rewards = []

    policy.eval()
    rng = np.random.default_rng(cfg.seed)
    for _ in range(cfg.train.num_eval_episodes):
        env.reset(seed=int(rng.uniform(0, int(1e10))))
        eval_td = env.rollout(max_steps=cfg.dataset.max_steps_per_episode, policy=policy, auto_cast_to_device=True)
        episode_rewards.append(eval_td["next", "reward"].sum().item())
    policy.train()
    return np.mean(episode_rewards)

@hydra.main(config_path=str((Path(__file__).parent.parent / "configs").resolve()), config_name="train")
def main(cfg: DictConfig):
    # Initialize Weights & Biases    
    print(f"Using W&B mode: {cfg.logging.mode}")
    print(OmegaConf.to_container(cfg))
    wandb.init(entity="andi-mueller-csem-sa", project="offline-rl", mode=cfg.logging.mode, config=OmegaConf.to_container(cfg))

    # Load dataset and environment
    dataset = None
    if isinstance(cfg.dataset.id, Iterable) and not isinstance(cfg.dataset.id, str):
        dataset = [
            minari.load_dataset(dataset_id, download=cfg.dataset.get("download", False))
            for dataset_id in cfg.dataset.id
        ]
        print(f"Loaded datasets {cfg.dataset.id} with {sum([len(x) for x in dataset])} episodes.")
    else:        
        dataset = minari.load_dataset(cfg.dataset.id, download=cfg.dataset.get("download", False))
        print(f"Loaded dataset '{cfg.dataset.id}' with {len(dataset)} episodes.")
    buffer = LocalMinariReplayBuffer(dataset, max_size=cfg.train.replay_buffer_size, load_bsize=cfg.dataset.get("load_bsize", 32))
    
    print(f"Buffer size: {len(buffer)}")
    
    base_env = GymEnv(cfg.env.id, continuing_task=True, reset_target=True, max_episode_steps=cfg.dataset.max_steps_per_episode, device=cfg.model.device)
    env = TransformedEnv(base_env, DoubleToFloat())
    env.set_seed(cfg.seed)

    # Build actor network
    model = IQLNetwork(env, cfg.model, cfg.model.device)
    print(model)

    # print model weights
    print(f"Actor network weights: {sum(p.numel() for p in model.actor.parameters())}")
    print(f"Value network weights: {sum(p.numel() for p in model.value_net.parameters())}")
    print(f"Q-value network weights: {sum(p.numel() for p in model.q_net.parameters())}")
    
    target_net_updater = None
    if cfg.model.type == "iql":
        loss_module = IQLLoss(
            actor_network=model.actor,
            value_network=model.value_net,
            qvalue_network=model.q_net,
            loss_function="l2",
            temperature=cfg.model.temperature,
            expectile=cfg.model.expectile,
            #gamma=cfg.model.gamma,
        )
        loss_module.make_value_estimator(gamma=cfg.model.gamma)
        target_net_updater = SoftUpdate(loss_module, tau=0.005)
        optimizer = torch.optim.AdamW(loss_module.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.num_training_steps)
    elif cfg.model.type == "cql":
        loss_module = CQLLoss(
            actor_network=model.actor,
            qvalue_network=model.q_net,
            #gamma=cfg.model.gamma,
            lagrange_thresh=cfg.model.lagrange_thresh,
            with_lagrange=cfg.model.with_lagrange,
        )
        loss_module.make_value_estimator(gamma=cfg.model.gamma)
        
        optimizer = [
            
            torch.optim.AdamW(model.actor.parameters(), lr=cfg.model.actor_lr, weight_decay=cfg.train.weight_decay),
            torch.optim.AdamW(model.q_net.parameters(), lr=cfg.model.q_lr, weight_decay=cfg.train.weight_decay)
        ]
        
        scheduler = [
            torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=cfg.train.num_training_steps)
            for o in optimizer
        ]

    loss_logs = []
    eval_reward_logs = [np.nan]
    
    for step in (pbar := tqdm(range(cfg.train.num_training_steps), desc="Training", unit="step")):
        batch = buffer.sample(cfg.train.batch_size).to(cfg.model.device)
        # 2) Compute loss l = L_V + L_Q + L_pi
        loss_dict = loss_module(batch)
        
        infos = {}
        if cfg.model.type == "iql":
            loss = loss_dict["loss_qvalue"] + loss_dict["loss_actor"] + loss_dict["loss_value"]
        elif cfg.model.type == "cql":
            loss = (
                    loss_dict["loss_qvalue"] +
                    loss_dict["loss_cql"] +
                    loss_dict["loss_actor"] #+
                    #loss_dict["loss_alpha"] +
                    #loss_dict["loss_alpha_prime"]
                )
            
        for k, v in loss_dict.items():
            infos[k] = v.item()
        
        loss_logs.append(loss.item())

        # 3) Backpropagate the gradients
        if cfg.model.type == "iql":
            optimizer.zero_grad()
            loss.backward()
        elif cfg.model.type == "cql":
            for opt in optimizer:
                opt.zero_grad()
            loss.backward()
        
        # compute gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            loss_module.parameters(),
            cfg.train.grad_norm_clip,
        )
            
        if cfg.model.type == "iql":
            optimizer.step()  # Update V(s), Q(a, s), pi(a|s)
            if cfg.train.use_scheduler:
                scheduler.step()  # Update the learning rate
            if target_net_updater is not None:
                target_net_updater.step()  # Update the target Q-network
        elif cfg.model.type == "cql":
            for opt in optimizer:
                opt.step()
            
            if cfg.train.use_scheduler:
                for sch in scheduler:
                    sch.step()
        

        if step % cfg.train.log_every == 0:
            if cfg.model.type == "iql":
                lr = optimizer.param_groups[0]["lr"]
                alr = optimizer.param_groups[0]["lr"]
                qlr = optimizer.param_groups[0]["lr"]
            elif cfg.model.type == "cql":
                lr = 0
                alr = optimizer[0].param_groups[0]["lr"]
                qlr = optimizer[1].param_groups[0]["lr"]
                
            wandb.log(
                {
                    "loss": loss.item(),
                    "loss_value": loss_dict["loss_value"].item() if "loss_value" in loss_dict else 0,
                    "loss_qvalue": loss_dict["loss_qvalue"].item(),
                    "loss_actor": loss_dict["loss_actor"].item(),
                    "grad_norm": grad_norm,
                    "learning_rate": lr,
                    "actor_learning_rate": alr,
                    "q_learning_rate": qlr,
                    "state_value": batch["state_value"].mean() if "state_value" in batch else 0,
                    "state_action_value": model.q_net(batch)["state_action_value"].mean(),
                    "td_error": batch["td_error"].mean(),
                    **infos,
                },
                step=step,
            )
            
        pbar.set_description(
            f"Step {step} / {cfg.train.num_training_steps}; Loss: {loss_logs[-1]:.1f}, Avg eval return: {eval_reward_logs[-1]:.1f}"
        )
        
        # Evaluate the policy
        if step % cfg.train.eval_every == 0:
            log_and_eval(cfg, env, model, eval_reward_logs, step)
            
    # Final evaluation
    log_and_eval(cfg, env, model, eval_reward_logs, step)

def log_and_eval(cfg, env, model, eval_reward_logs, step):
    eval_reward_logs.append(evaluate_policy(env, model.actor, cfg))
    wandb.log({"eval_reward": eval_reward_logs[-1]}, step=step)
            
    if cfg.train.store_model:
                # store actor 
        torch.save(
                    model.state_dict(),
                    f"model_{cfg.env.id}_latest.pth",
                )
                
                


if __name__ == "__main__":
    main()
