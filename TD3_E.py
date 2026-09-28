import numpy as np
import torch as th
import torch.nn as nn
import copy
from PER_Sumtree import PERMemory


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


def init_layers(layers, init_custom):
    for layer in layers:
        if hasattr(layer, "weight"):
            if init_custom:
                layer.weight.data.uniform_(*hidden_init(layer))
            else:
                nn.init.xavier_uniform_(layer.weight)

        if hasattr(layer, "bias"):
            if layer.bias is not None:
                if init_custom:
                    layer.bias.data.uniform_(*hidden_init(layer))
                else:
                    nn.init.xavier_uniform_(layer.bias)


class Actor(nn.Module):
    def __init__(self, input_size, output_size, layers=[256, 128], init_custom=True):
        super(Actor, self).__init__()
        self.layers = nn.ModuleList()
        self.init_custom = init_custom
        pre_units = input_size

        for L in layers:
            self.layers.append(nn.Linear(pre_units, L))
            self.layers.append(nn.ReLU())
            pre_units = L
        self.final_linear = nn.Linear(pre_units, output_size)
        self.final_activation = nn.Tanh()
        self.reset_parameters()
        return

    def reset_parameters(self):
        init_layers(self.layers, self.init_custom)
        nn.init.uniform_(self.final_linear.weight, -0.003, 0.003)
        nn.init.uniform_(self.final_linear.bias, -0.003, 0.003)
        return

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_linear(x)
        x = self.final_activation(x)
        return x


class Critic(nn.Module):
    def __init__(self, state_size, act_size, output_size=1, layers=[256, 256], init_custom=True):
        super(Critic, self).__init__()
        self.init_custom = init_custom
        self.layers = nn.ModuleList()

        pre_units = state_size + act_size
        for L in layers:
            self.layers.append(nn.Linear(pre_units, L))
            self.layers.append(nn.ReLU())
            pre_units = L

        self.final_linear = nn.Linear(pre_units, output_size)
        self.reset_parameters()
        return

    def reset_parameters(self):
        init_layers(self.layers, self.init_custom)
        nn.init.uniform_(self.final_linear.weight, -0.003, 0.003)
        nn.init.uniform_(self.final_linear.bias, -0.003, 0.003)
        return

    def forward(self, state, action):
        x = th.cat((state, action), dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.final_linear(x)
        return x


class TD3_E_PER:
    """
    Baseline: Ensemble-based uncertainty PER（探针式 ensemble）
    与 TD3_S（一阶差分 U）/ TD3_V（双 Q 分歧 U）的唯一区别: U 的计算方式。

    设计:
    - 新增 N 个独立初始化的小 critic 探针（不参与 TD3 主干训练，仅做不确定性评估器）
    - 探针以主网络的 TD 目标做 MSE 回归，各自独立优化器
    - U = N 个探针对同一 (s,a) 预测的逐样本方差，batch 内 max 归一化到 [0,1]
      （与 TD3_V 的归一化方式一致，保证 (1+U)^PER_BETA 调制强度可比）
    - 复合优先级公式不变: P = |δ|^α / (1+U)^β
    - 主干（actor/双 critic/target 软更新）与 TD3_S/TD3_V 完全一致
    """
    def __init__(self, state_dim, action_dim, max_action, device,
                 discount=0.99, tau=0.005, policy_noise=0.2,
                 noise_clip=0.5, policy_freq=2,
                 n_probes=5, probe_layers=[128, 128]):
        self.device = device
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.actor = Actor(input_size=state_dim, output_size=action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic_1 = Critic(state_size=state_dim, act_size=action_dim).to(device)
        self.critic_target_1 = copy.deepcopy(self.critic_1)
        self.critic_1_optimizer = th.optim.Adam(self.critic_1.parameters(), lr=3e-4)

        self.critic_2 = Critic(state_size=state_dim, act_size=action_dim).to(device)
        self.critic_target_2 = copy.deepcopy(self.critic_2)
        self.critic_2_optimizer = th.optim.Adam(self.critic_2.parameters(), lr=3e-4)

        # ===== 探针式 ensemble（仅用于不确定性估计，不影响主干）=====
        self.n_probes = n_probes
        self.probes = nn.ModuleList([
            Critic(state_size=state_dim, act_size=action_dim, layers=probe_layers).to(device)
            for _ in range(n_probes)
        ])
        self.probe_optimizers = [th.optim.Adam(p.parameters(), lr=3e-4) for p in self.probes]

        self.memory = PERMemory(capacity=int(1e6), engine='torch', device=device, continuous=True)
        self.total_it = 0
        self.RANDOM_WARM_UP = 256

    def select_action(self, state):
        state = th.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.max_action * self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        if len(self.memory) < self.RANDOM_WARM_UP:
            return

        experiences, tree_idxs, IS_weights = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = experiences

        with th.no_grad():
            noise = (th.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.max_action * self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)

            target_Q1 = self.critic_target_1(next_states, next_action)
            target_Q2 = self.critic_target_2(next_states, next_action)
            target_Q = th.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.discount * target_Q

        # ===== Ensemble-based 方法计算 U（逐样本探针方差）=====
        # 用更新前的探针计算当前不确定性（反映采样时刻各探针对该样本的分歧）
        with th.no_grad():
            probe_Qs = th.stack([probe(states, actions) for probe in self.probes], dim=0)  # (N, B, 1)
            U_np = probe_Qs.var(dim=0).cpu().numpy().flatten()
        # batch 内 max 归一化: U ∈ [0,1]，与 TD3_V 保持一致
        U = U_np / (np.max(U_np) + 1e-8)
        # ===== Ensemble-based 方法结束 =====

        current_Q1 = self.critic_1(states, actions)
        current_Q2 = self.critic_2(states, actions)

        residual_1 = current_Q1 - target_Q
        td_errors = th.abs(residual_1).cpu().detach().numpy().flatten()

        self.memory.batch_update(tree_idxs, td_errors, U)

        critic_1_loss = (residual_1.pow(2) * IS_weights).mean()

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        th.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1)
        self.critic_1_optimizer.step()

        residual_2 = current_Q2 - target_Q
        critic_2_loss = (residual_2.pow(2) * IS_weights).mean()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        th.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 1)
        self.critic_2_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic_1(states, self.max_action * self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # ===== 探针训练: 以主网络 TD 目标做 MSE 回归（主干更新完成后进行）=====
        for probe, probe_opt in zip(self.probes, self.probe_optimizers):
            probe_loss = (probe(states, actions) - target_Q).pow(2).mean()
            probe_opt.zero_grad()
            probe_loss.backward()
            probe_opt.step()
