import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from src.core.env import StockTradingEnv

class ActorCritic(nn.Module):
    """
    Actor-Critic 网络架构，包含策略网络(Actor)和价值网络(Critic)
    针对连续动作空间优化
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # 共享的特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor网络 - 输出动作均值
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # 确保权重和为1
        )
        
        # 初始化动作的协方差矩阵 (对角线矩阵)
        self.action_var = nn.Parameter(torch.ones(action_dim) * 0.1)
        
        # Critic网络 - 输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        features = self.feature_layer(state)
        action_mean = self.actor_mean(features)
        state_value = self.critic(features)
        return action_mean, state_value
    
    def get_action_and_value(self, state, action=None):
        """获取动作分布、状态价值和熵"""
        features = self.feature_layer(state)
        action_mean = self.actor_mean(features)
        state_value = self.critic(features)
        
        # 创建协方差矩阵 (对角线)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        
        # 创建多元正态分布
        dist = MultivariateNormal(action_mean, cov_mat)
        
        if action is None:
            # 采样动作并确保权重和为1
            action = dist.sample()
            action = action / action.sum() if action.sum() > 0 else action
        
        action_logprob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, action_logprob, state_value, entropy

class PPOAgent:
    """
    使用PPO算法的投资组合优化代理
    """
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, 
                 k_epochs=4, batch_size=64, device='auto'):
        """
        初始化PPO代理
        
        参数:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            lr: 学习率
            gamma: 折扣因子
            eps_clip: PPO裁剪系数
            k_epochs: 每批数据的训练轮数
            batch_size: 批量大小
            device: 计算设备 ('cpu', 'cuda', 'auto')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        
        # 确定设备 (CPU或GPU)
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 初始化策略网络
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 初始化旧策略 (用于计算重要性采样比率)
        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 训练标志
        self.training = True
        
        # 存储每个episode的状态、动作、奖励等
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.next_states = []
        self.is_terminals = []
        
        # 损失函数
        self.MseLoss = nn.MSELoss()
        
    def select_action(self, state):
        """
        根据当前状态选择动作
        
        参数:
            state: 当前状态
            
        返回:
            action: 选择的动作 (投资组合权重)，限制在[0,1]之间，只做多策略
        """
        # 转换为PyTorch张量
        state = torch.FloatTensor(state).to(self.device)
        
        # 使用旧策略获取动作
        with torch.no_grad():
            features = self.policy_old.feature_layer(state)
            action_mean = self.policy_old.actor_mean(features)
            
            # 在评估模式下，直接使用动作均值作为投资组合权重
            if not self.training:
                return action_mean.cpu().numpy() # action_mean 来自 Softmax，已归一化且在[0,1]范围内
            
            # 创建协方差矩阵 (对角线)
            # 减小方差以控制探索的范围，避免产生过大的随机值
            action_var = self.policy_old.action_var.expand_as(action_mean).clamp(min=1e-6, max=0.01) 
            cov_mat = torch.diag_embed(action_var)
            
            # 创建多元正态分布
            dist = MultivariateNormal(action_mean, cov_mat)
            action_sampled = dist.sample() # 采样动作
            logprob = dist.log_prob(action_sampled)
            
            # 纯做多策略：使用Softmax将采样动作转换为非负值且和为1
            # 这确保了所有权重都在[0,1]之间，并且总和为1
            final_action = torch.nn.functional.softmax(action_sampled, dim=-1)
            
            # 确保没有极小的数值问题
            sum_check = final_action.sum()
            if abs(sum_check - 1.0) > 1e-4:  # 如果和与1相差太远，这通常不应该发生
                # 强制重新归一化
                final_action = final_action / (sum_check + 1e-10)
            
            # 调试检查 - 确认权重都在[0,1]范围内且总和为1
            if torch.any(final_action < 0) or torch.any(final_action > 1.01) or abs(final_action.sum() - 1.0) > 0.01:
                # 只有在确实有问题的情况下才触发断点
                import pdb; pdb.set_trace()

        # 存储原始采样动作的对数概率 (用于PPO更新)  
        if self.training:
            self.logprobs.append(logprob)
        
        return final_action.cpu().numpy()
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储状态转移"""
        self.states.append(torch.FloatTensor(state).to(self.device))
        self.actions.append(torch.FloatTensor(action).to(self.device))
        self.rewards.append(reward)
        self.next_states.append(torch.FloatTensor(next_state).to(self.device))
        self.is_terminals.append(done)
    
    def train(self):
        """训练代理（PPO更新）"""
        if len(self.states) < self.batch_size:
            # 数据不足，跳过训练
            return
        
        # 计算每个时间步的折扣回报
        rewards = []
        discounted_reward = 0
        
        # 从后往前计算GAE (广义优势估计)
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # 转换为张量
        old_states = torch.stack(self.states).detach()
        old_actions = torch.stack(self.actions).detach()
        old_logprobs = torch.stack(self.logprobs).detach()
        
        # 计算当前状态值
        with torch.no_grad():
            old_values = self.policy_old.get_action_and_value(old_states)[2].squeeze()
        
        # 计算优势
        rewards = torch.FloatTensor(rewards).to(self.device)
        advantages = rewards - old_values
        
        # 归一化优势
        if len(advantages) > 1:  # 防止单样本情况
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 优化策略
        for _ in range(self.k_epochs):
            # 对数据进行随机采样
            indices = torch.randperm(len(old_states))
            
            # 批量训练
            for start_idx in range(0, len(indices), self.batch_size):
                # 获取batch索引
                end_idx = min(start_idx + self.batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]
                
                # 获取批量数据
                batch_states = old_states[batch_indices]
                batch_actions = old_actions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = rewards[batch_indices]
                
                # 评估当前策略
                _, new_logprobs, state_values, entropy = self.policy.get_action_and_value(
                    batch_states, batch_actions
                )
                
                # 计算比率 (pi_theta / pi_theta__old)
                ratios = torch.exp(new_logprobs - batch_old_logprobs)
                
                # 裁剪比率，计算不同的目标函数
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * batch_advantages
                
                # 计算损失
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.MseLoss(state_values, batch_returns)
                entropy_loss = -entropy.mean() * 0.01  # 熵正则化
                
                # 总损失
                total_loss = actor_loss + critic_loss + entropy_loss
                
                # 优化
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)  # 梯度裁剪
                self.optimizer.step()
        
        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 清空buffer
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.next_states = []
        self.is_terminals = []
        
    def save(self, directory, name="ppo_agent"):
        """保存模型参数"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        model_path = os.path.join(directory, f"{name}.pth")
        torch.save({
            'policy': self.policy.state_dict(),
            'policy_old': self.policy_old.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, model_path)
        print(f"模型已保存至 {model_path}")
    @classmethod
    def load(self, directory, name="ppo_agent"):
        """加载模型参数"""
        model_path = os.path.join(directory, f"{name}.pth")
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy'])
            self.policy_old.load_state_dict(checkpoint['policy_old'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"模型已从 {model_path} 加载")
            return True
        else:
            print(f"无法加载模型，文件不存在: {model_path}")
            return False
            
# 向后兼容的类名
SimpleRLAgent = PPOAgent