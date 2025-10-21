# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 11:43:06 2025

@author: lenovo
"""

import time
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import heapq
from env import DefectEnv, network, load_data_from_tensor, load_stats

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    cuda: bool = True
    track: bool = False
    capture_video: bool = False
    save_model: bool = True

    total_timesteps: int = 100000
    learning_rate: float = 3e-4
    num_envs: int = 1
    num_steps: int = 256
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 20
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None
    init_logstd: float = -1.0
    final_logstd: float = -1.5
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, init_logstd=0.0, stats=None):
        super().__init__()
        obs_shape = np.array(envs.single_observation_space.shape).prod()
        act_shape = int(np.prod(envs.single_action_space.shape))
        self.act_shape = act_shape
        self.stats=stats
        # critic branches
        self.critic_c_branch = nn.Sequential(
           nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
           nn.LeakyReLU(),
           nn.AdaptiveAvgPool1d(25),
           nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
           nn.LeakyReLU(),
           nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
           nn.LeakyReLU(),
           nn.AdaptiveAvgPool1d(1),
           nn.Flatten(),
           nn.Linear(64, 128),
           nn.LeakyReLU(),
       )
        self.critic_b1_branch = nn.Sequential(
           nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
           nn.LeakyReLU(),
           nn.AdaptiveAvgPool1d(16),
           nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
           nn.LeakyReLU(),
           nn.AdaptiveAvgPool1d(1),
           nn.Flatten(),
           nn.Linear(32, 64),
           nn.LeakyReLU(),
       )
        self.critic_b2_branch = nn.Sequential(
           nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
           nn.LeakyReLU(),
           nn.AdaptiveAvgPool1d(16),
           nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
           nn.LeakyReLU(),
           nn.AdaptiveAvgPool1d(1),
           nn.Flatten(),
           nn.Linear(32, 64),
           nn.LeakyReLU(),
       )
        self.critic_s_branch = nn.Sequential(
           nn.Linear(6, 8),
           nn.LeakyReLU(),
       )
        self.critic_g_branch = nn.Sequential(
           nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
           nn.LeakyReLU(),
           nn.AdaptiveAvgPool2d((8, 8)),
           nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
           nn.LeakyReLU(),
           nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
           nn.LeakyReLU(),
           nn.AdaptiveAvgPool2d((1, 1)),
           nn.Flatten(),
           nn.Linear(64, 128),
           nn.LeakyReLU(),
       )
        self.critic_fc = nn.Sequential(
           nn.Linear(128 + 64 + 64 + 8 + 128, 256),
           nn.LeakyReLU(),
           nn.Linear(256, 64),
           nn.LeakyReLU(),
           nn.Linear(64, 1)
       )
        self.actor_c_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(25),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
        )
        self.actor_b1_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
        )
        self.actor_b2_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
        )
        self.actor_s_branch = nn.Sequential(
            nn.Linear(6, 8),
            nn.LeakyReLU(),
        )
        self.actor_g_branch = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
        )
        self.actor_fc = nn.Sequential(
            nn.Linear(128 + 64 + 64 + 8 + 128, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, act_shape),
        )
        self.actor_logstd = nn.Parameter(torch.ones(act_shape, dtype=torch.float32) * float(init_logstd))

    def extract_features(self, x):
        device = x.device
        batch = x.shape[0]
        C_list, B1_list, B2_list, S_list, G_list = [], [], [], [], []
        for i in range(batch):
            c,b1,b2,s,g = load_data_from_tensor(x[i], device=device, stats=self.stats)
            C_list.append(c)
            B1_list.append(b1)
            B2_list.append(b2)
            S_list.append(s)
            G_list.append(g)
        C = torch.cat(C_list, dim=0).to(device)  
        B1 = torch.cat(B1_list, dim=0).to(device)
        B2 = torch.cat(B2_list, dim=0).to(device)
        S = torch.cat(S_list, dim=0).to(device)
        G = torch.cat(G_list, dim=0).to(device)  
        return C, B1, B2, S, G

    def get_value(self, x):
        C, B1, B2, S, G = self.extract_features(x)
        x_c = self.critic_c_branch(C.unsqueeze(1))
        x_b1 = self.critic_b1_branch(B1.unsqueeze(1))
        x_b2 = self.critic_b2_branch(B2.unsqueeze(1))
        x_s = self.critic_s_branch(S)
        x_g = self.critic_g_branch(G)
        x_combined = torch.cat([x_c, x_b1, x_b2, x_s, x_g], dim=1)
        return self.critic_fc(x_combined)

    def get_action_and_value(self, x, action=None):
        C, B1, B2, S, G = self.extract_features(x)
        x_c = self.actor_c_branch(C.unsqueeze(1))
        x_b1 = self.actor_b1_branch(B1.unsqueeze(1))
        x_b2 = self.actor_b2_branch(B2.unsqueeze(1))
        x_s = self.actor_s_branch(S)
        x_g = self.actor_g_branch(G)
        x_combined = torch.cat([x_c, x_b1, x_b2, x_s, x_g], dim=1)
        action_mean = self.actor_fc(x_combined)
        action_logstd = self.actor_logstd.unsqueeze(0).expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(x)

if __name__ == "__main__":
    seeds=[1,2,3,4]
    for s in seeds:
        t1 = time.time()
        reward_history = []
        avg_reward_history = []
        global_step_history = []
        best_reward=-float('inf')
        topk=100
        top_rewards=[]
        args = tyro.cli(Args)
        args.seed=s
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = max(1, args.total_timesteps // args.batch_size)
        run_name = f"DefectEnv__{args.exp_name}__{args.seed}__{int(time.time())}"
        writer = SummaryWriter(f"runs/{run_name}")
        print(f"TensorBoard logdir: runs/{run_name}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        stats = load_stats('jc_statistics_m.pth', device=device)   
        envs = gym.vector.SyncVectorEnv([lambda: DefectEnv() for _ in range(args.num_envs)])
        assert isinstance(envs.single_action_space, gym.spaces.Box)
        agent = Agent(envs, init_logstd=args.init_logstd,stats=stats).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
        logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
        rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
        dones = torch.zeros((args.num_steps, args.num_envs), device=device)
        values = torch.zeros((args.num_steps, args.num_envs), device=device)
        global_step = 0
        start_time = time.time()
        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
        next_done = torch.zeros(args.num_envs, device=device)
    
        for iteration in range(1, args.num_iterations + 1):
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate
            for step in range(args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done
    
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob
                action_np = action.detach().cpu().numpy()
                next_obs_np, reward_np, terminations, truncations, infos = envs.step(action_np)
                next_done = np.logical_or(terminations, truncations)
    
                rewards[step] = torch.tensor(reward_np, dtype=torch.float32, device=device).view(-1)
                reward_history.extend([float(r) for r in reward_np])
                global_step_history.append(global_step)
                reward_val = float(reward_np[0])       
                obs_val = next_obs_np[0].copy()        
    
                if reward_val > best_reward:
                    best_reward = reward_val
                writer.add_scalar("charts/best_reward", best_reward, global_step)
    
                if len(top_rewards) < topk:
                    heapq.heappush(top_rewards, (reward_val, tuple(obs_val)))
                else:
                    heapq.heappushpop(top_rewards, (reward_val, tuple(obs_val)))
                if len(reward_history) >= 100:
                    avg_reward = float(np.mean(reward_history[-100:]))
                    avg_reward_history.append(avg_reward)
                    writer.add_scalar("charts/avg_reward_100", avg_reward, global_step)
    
                writer.add_scalar("charts/instant_reward", float(reward_np), global_step)
    
                next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
                next_done = torch.tensor(next_done, dtype=torch.float32, device=device)
    
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
    
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards, device=device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    advantages[t] = lastgaelam
                returns = advantages + values
    
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
    
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            approx_kl = torch.tensor(0.0, device=device)
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    mb_obs = b_obs[mb_inds].to(device)
                    mb_actions = b_actions[mb_inds].to(device)
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
                    logratio = newlogprob - b_logprobs[mb_inds].to(device)
                    ratio = logratio.exp()
    
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
    
                    mb_advantages = b_advantages[mb_inds].to(device)
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
    
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds].to(device)) ** 2
                        v_clipped = b_values[mb_inds].to(device) + torch.clamp(
                            newvalue - b_values[mb_inds].to(device),
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns[mb_inds].to(device)) ** 2).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds].to(device)) ** 2).mean()
    
                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
    
                if args.target_kl is not None and approx_kl.item() > args.target_kl:
                    break
    
            progress = iteration / args.num_iterations
    
            if iteration % 5 == 0:
                recent_rewards = np.array(reward_history[-args.batch_size:] if len(reward_history) > args.batch_size else reward_history)
                if len(recent_rewards) > 0:
                    writer.add_scalar("stats/reward_mean", np.mean(recent_rewards), global_step)
                    writer.add_scalar("stats/reward_std", np.std(recent_rewards), global_step)
                    writer.add_scalar("stats/reward_max", np.max(recent_rewards), global_step)
                    writer.add_scalar("stats/reward_min", np.min(recent_rewards), global_step)
    
            explained_var = 1 - np.var(b_returns.cpu().numpy() - b_values.cpu().numpy()) / (np.var(b_returns.cpu().numpy()) + 1e-8)
            
    
        if args.save_model:
            torch.save(agent.state_dict(), f'final_policy_{s}_m.pth')
        # --- 训练结束保存 top-100 ---
        top_rewards_sorted = sorted(top_rewards, key=lambda x: -x[0])
        np.savez(f"top100_rewards_obs_{s}_RL_m.npz",
             rewards=np.array([r for r, _ in top_rewards_sorted]),
             obs=np.array([o for _, o in top_rewards_sorted]))
        print(f"Top-100 rewards and corresponding next_obs saved to top100_rewards_obs_{s}_RL.npz")
        envs.close()
        writer.close()
        t2 = time.time()
        print("Elapsed:", t2 - t1)
