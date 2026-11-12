# Online-FLAME for Fusion 设计

## 1 算法概述

### 1.1 核心思想

Online-FLAME 将 **最大熵强化学习** 与 **单步流匹配** 相结合，通过 Q-reweighted 流匹配目标实现原则性的探索-利用权衡。相比 Online-OFQL，FLAME 的关键改进在于：

1. **Q-Reweighted Flow Matching**：将 Q 值通过重要性权重融入流匹配训练，理论上等价于优化 $Q(s,a) + \alpha H(\pi)$
2. **解耦的密度估计器**：使用增广 ODE 独立计算策略熵，用于 Critic 的熵正则化
3. **原则性的探索机制**：自动平衡探索与利用，无需手工设计探索噪声

### 1.2 理论保证

**命题 1**（Q-Reweighted 目标的一致性）：  
Q-reweighted 流匹配目标的全局最优解与原始最大熵目标 $\max_\theta \mathbb{E}_\pi[Q(s,a) + \alpha H(\pi)]$ 的最优解一致。

**命题 2**（密度估计误差界）：  
单步密度估计的误差为 $\mathcal{E}_{\text{single}} \approx \frac{1}{2}\text{Tr}(\mathcal{J}^2)$，多步估计（$N_{\text{est}}$ 步）将误差降低到 $O(1/N_{\text{est}})$。

---

## 2 算法框架

### 2.1 策略表示

策略表示为**平均速度场** $u_\theta(a_t, t; s)$（注意：与 OFQL 不同，FLAME 只使用单个时间变量 $t$）：

**推理时单步生成**：
$$a = \varepsilon - u_\theta(\varepsilon, t{=}1; s), \quad \varepsilon \sim \mathcal{N}(0, I)$$

**密度估计**（用于熵计算）：
$$\log \pi_\theta(a|s) = \log p_0(\varepsilon) - \int_0^1 \text{Tr}\left(\frac{\partial u_\theta(a_t, t; s)}{\partial a_t}\right) dt$$

其中积分项通过增广 ODE 计算，避免显式求解高维雅可比矩阵。

### 2.2 损失函数

#### Critic 损失（带熵正则化）

使用解耦密度估计器改进 Bellman 目标：

$$y_i = r_i + \gamma \cdot (1 - \text{done}_i) \cdot \left[\min_{j \in \{1,2\}} Q_{\phi'_j}(s'_i, a'_i) + \alpha \cdot H_i\right]$$

其中熵项 $H_i = -\log \pi_{\theta'}(a'_i | s'_i)$ 通过增广 ODE 计算。

完整 Critic 损失：
$$\mathcal{L}_{\text{critic}}(\phi) = \frac{1}{B} \sum_{i=1}^B \sum_{j=1,2} \left(Q_{\phi_j}(s_i, a_i) - y_i\right)^2$$

#### Actor 损失（Q-Reweighted Flow Matching）

$$\mathcal{L}_{\text{actor}}(\theta) = \mathbb{E}_{(s,a) \sim \mathcal{B}} \left[ \exp\left(\frac{Q_\phi(s, a_1)}{\alpha}\right) \cdot \mathbb{E}_{t, \varepsilon} \left[\|u_\theta(a_t, t; s) - u_{\text{target}}\|^2\right] \right]$$

其中：
- $a_1 \sim \pi_\theta(\cdot | s)$ 是策略生成的终点动作
- $a_t = (1-t) \varepsilon + t \cdot a_1$ 是线性插值路径
- $u_{\text{target}} = a_1 - \varepsilon$ 是目标速度
- $\exp(Q/\alpha)$ 是重要性权重，使高 Q 值区域获得更大训练权重

---

## 3 实现细节

### 3.1 变量说明

| 符号 | 含义 | 取值/维度 |
|------|------|-----------|
| $\mathcal{S}, \mathcal{A}$ | 状态空间、动作空间 | $\mathcal{A} \subseteq \mathbb{R}^{d_a}$ |
| $s, s', a, a'$ | 当前/下一 状态/动作 | — |
| $r, \gamma$ | 即时奖励、折扣因子 | $r \in \mathbb{R}, \gamma \in [0,1)$ |
| $\mathcal{B}$ | Replay Buffer | $\{(s, a, r, s', \text{done})\}$，容量 $N$ |
| $\theta, \theta'$ | 策略网络参数、目标网络参数 | EMA 更新 |
| $\phi_i, \phi'_i$ | Critic 参数（$i \in \{1,2\}$）、目标 Critic | EMA 更新 |
| $\rho$ | EMA 衰减率 | $\in (0,1)$，典型值 $0.995$ |
| $\alpha$ | 温度参数（熵权重） | 自适应调节，初始值 $\alpha_0 = 1.0$ |
| $N_{\text{est}}$ | 密度估计的积分步数 | 训练时 $5 \sim 10$，推理时 $1$ |
| $N_{\text{act}}$ | 批量候选动作数（探索） | 典型值 $16 \sim 32$ |
| $B$ | Mini-batch 大小 | 典型值 $256$ |
| $T_{\text{warm}}$ | 热身步数 | 可选，建议 $1000 \sim 5000$ |
| $G$ | UTD ratio（每步环境交互的梯度更新次数） | 典型值 $1$（UTD=1.0）或 $0.2$（1更新/5步） |

### 3.2 网络架构

#### 速度网络 $u_\theta(a_t, t; s)$

```python
class VelocityNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.time_embed = SinusoidalEmbedding(dim=64)
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, a_t, t, s):
        # a_t: [B, d_a], t: [B, 1], s: [B, d_s]
        t_embed = self.time_embed(t)  # [B, 64]
        x = torch.cat([s, a_t, t_embed], dim=-1)
        return self.net(x)  # [B, d_a]
```

#### 增广 ODE 密度估计器

```python
class AugmentedODEEstimator:
    """解耦的密度估计器，用于计算 log π(a|s)"""
    
    def __init__(self, velocity_net, N_est=5):
        self.u_theta = velocity_net
        self.N_est = N_est  # 多步积分的步数
    
    def estimate_log_prob(self, s, a_1):
        """
        计算 log π(a_1 | s)
        
        Args:
            s: 状态 [B, d_s]
            a_1: 终点动作 [B, d_a]
        
        Returns:
            log_prob: [B]
        """
        B = s.shape[0]
        device = s.device
        
        # 初始化：ε ~ N(0, I)
        epsilon = torch.randn_like(a_1)
        log_p0 = -0.5 * (epsilon ** 2).sum(dim=-1) - 0.5 * a_1.shape[-1] * np.log(2 * np.pi)
        
        # 累积散度项（通过增广 ODE）
        div_accum = torch.zeros(B, device=device)
        
        # 多步积分：将 [0, 1] 分为 N_est 个子区间
        dt = 1.0 / self.N_est
        a_t = epsilon.clone()
        
        for i in range(self.N_est):
            t = torch.full((B, 1), i * dt, device=device)
            
            # 计算 Tr(∂u/∂a_t) 使用 Hutchinson's trace estimator
            with torch.enable_grad():
                a_t.requires_grad_(True)
                u_t = self.u_theta(a_t, t, s)
                
                # 随机投影向量（Hutchinson estimator）
                v = torch.randn_like(a_t)
                jvp = torch.autograd.grad(u_t, a_t, v, create_graph=False)[0]
                div_u = (jvp * v).sum(dim=-1)  # [B]
            
            # 累积散度
            div_accum += div_u * dt
            
            # Euler 步进
            a_t = a_t + u_t * dt
            a_t = a_t.detach()
        
        # 最终密度
        log_prob = log_p0 - div_accum
        return log_prob
```

### 3.3 Online-FLAME 伪代码

```
Online-FLAME 训练
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入：
  env        — 在线环境
  N          — Replay Buffer 容量
  T_warm     — 热身步数（可选）
  G          — UTD ratio（每步环境交互的梯度更新次数）
  N_act      — 批量候选动作数（探索）
  N_est      — 密度估计积分步数
  ρ          — EMA 衰减率
  α₀         — 初始温度参数
  B          — mini-batch 大小

初始化：
  随机初始化速度网络参数 θ
  随机初始化 Q 网络参数 φ₁, φ₂
  θ' ← θ，φ'₁ ← φ₁，φ'₂ ← φ₂
  α ← α₀
  B_buf ← ReplayBuffer(capacity=N)
  entropy_estimator ← AugmentedODEEstimator(u_θ', N_est)
  s ← env.reset()

━━━━━━━━━━━━━━ 阶段一：热身（可选） ━━━━━━━━━━━━━━━━━━━
// 目的：预填 Replay Buffer，提供初始探索数据
// 可以跳过此阶段，直接开始在线学习

for step = 1 to T_warm:
    a ← UniformRandom(A_low, A_high)
    s', r, done ← env.step(a)
    B_buf.push((s, a, r, s', done))
    s ← done ? env.reset() : s'

━━━━━━━━━━━━━━ 阶段二：在线学习 ━━━━━━━━━━━━━━━━━━━━━━━━
for each environment step:

    // ══════════════════════════════════════════════════════
    // 步骤 1：批量候选采样 + 动作选择（MaxEnt 探索）
    // ══════════════════════════════════════════════════════
    
    // 1a. 生成 N_act 个候选动作
    candidates = []
    Q_values = []
    for i = 1 to N_act:
        ε_i ← N(0, I)
        a_i = ε_i - u_θ(ε_i, t=1; s)      // 单步生成
        candidates.append(a_i)
        Q_values.append(Q_φ₁(s, a_i))     // 用第一个 Critic 评估
    
    // 1b. 选择最高 Q 值的动作执行
    best_idx = argmax(Q_values)
    a_exec = clip(candidates[best_idx], A_low, A_high)
    
    // 1c. 环境交互
    s', r, done ← env.step(a_exec)
    B_buf.push((s, a_exec, r, s', done))
    s ← done ? env.reset() : s'

    // ══════════════════════════════════════════════════════
    // 步骤 2：梯度更新（G 次）
    // ══════════════════════════════════════════════════════
    for _ = 1 to G:
        
        // 从 Replay Buffer 采样 mini-batch
        {(s_i, a_i, r_i, s'_i, done_i)} ← B_buf.sample(B)
        
        // ── Critic 更新（带熵正则化）──────────────────────
        
        // 1. 为每个 s'_i 生成目标动作
        for each i in batch:
            ε'_i ← N(0, I)
            a'_i = ε'_i - u_θ'(ε'_i, t=1; s'_i)    // 目标策略
        
        // 2. 计算熵项（使用解耦密度估计器）
        log_π' = entropy_estimator.estimate_log_prob(s'_batch, a'_batch)
        H = -log_π'  // [B]
        
        // 3. Bellman 目标（Double Q + 熵正则化）
        Q_target_1 = Q_φ'₁(s'_batch, a'_batch)
        Q_target_2 = Q_φ'₂(s'_batch, a'_batch)
        Q_target = min(Q_target_1, Q_target_2)
        
        y = r_batch + γ * (1 - done_batch) * (Q_target + α * H)
        
        // 4. Critic 损失
        L_critic = (1/B) * Σᵢ [ (Q_φ₁(sᵢ, aᵢ) - yᵢ)² + (Q_φ₂(sᵢ, aᵢ) - yᵢ)² ]
        
        φ₁, φ₂ ← φ₁, φ₂ - ∇_φ L_critic
        
        // ── Actor 更新（Q-Reweighted Flow Matching）──────
        
        L_actor = 0
        for each i in batch:
            // 1. 生成终点动作 a₁
            ε_i ← N(0, I)
            a₁_i = ε_i - u_θ(ε_i, t=1; sᵢ)
            
            // 2. 计算重要性权重
            Q_i = Q_φ₁(sᵢ, a₁_i)  // 使用当前 Critic
            w_i = exp(Q_i / α)
            
            // 3. 流匹配损失（在插值路径上）
            t_i ← Uniform(0, 1)
            a_t_i = (1 - t_i) * ε_i + t_i * a₁_i
            u_pred_i = u_θ(a_t_i, t_i; sᵢ)
            u_target_i = a₁_i - ε_i              // 目标速度
            
            // 4. 加权流匹配损失
            L_actor += w_i * ‖u_pred_i - sg(u_target_i)‖²
        
        L_actor = L_actor / B
        
        θ ← θ - ∇_θ L_actor
        
        // ── 目标网络 EMA 更新 ────────────────────────────
        θ'   ← ρ * θ'   + (1 - ρ) * θ
        φ'₁ ← ρ * φ'₁ + (1 - ρ) * φ₁
        φ'₂ ← ρ * φ'₂ + (1 - ρ) * φ₂
        
        // ── 温度参数自适应（可选）────────────────────────
        // 根据当前策略熵调整 α，保持目标熵水平
        if step % update_alpha_freq == 0:
            current_entropy = entropy_estimator.estimate_entropy(s_batch)
            α ← α * exp(lr_alpha * (target_entropy - current_entropy))
            α ← clip(α, α_min, α_max)
```

### 3.4 推理伪代码

```
Online-FLAME 推理
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

模式 1: FLAME-R（快速推理，NFE=1）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入：状态 s，训练好的速度网络 u_θ

ε ← N(0, I)
a = ε - u_θ(ε, t=1; s)    // 单步生成，不加探索噪声

输出：a

模式 2: FLAME-M（精确推理，带多步密度估计）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入：状态 s，u_θ，Q_φ，N_cand（候选数），N_est（密度估计步数）

// 生成多个候选动作
candidates = []
log_probs = []
for i = 1 to N_cand:
    ε_i ← N(0, I)
    a_i = ε_i - u_θ(ε_i, t=1; s)
    
    // 使用多步密度估计计算 log π(a_i | s)
    log_π_i = estimate_log_prob_multistep(s, a_i, u_θ, N_est)
    
    candidates.append(a_i)
    log_probs.append(log_π_i)

// 选择最大化 Q(s, a) + α * log π(a|s) 的动作
scores = [Q_φ(s, a_i) + α * log_π_i for a_i, log_π_i in zip(candidates, log_probs)]
best_idx = argmax(scores)
a = candidates[best_idx]

输出：a
```

---

## 4 与 Online-OFQL 的对比

### 4.1 核心差异总结

| 维度 | Online-OFQL | Online-FLAME |
|------|-------------|--------------|
| **理论基础** | Flow-Based Consistency (FBC) | Q-Reweighted Flow Matching + MaxEnt |
| **Actor 损失** | $L_{\text{FBC}} - \alpha \cdot Q$ | $\mathbb{E}[\exp(Q/\alpha) \cdot L_{\text{FM}}]$ |
| **时间变量** | $(t, r)$ 两个变量 + Flow Ratio | 单个 $t$ |
| **熵处理** | 隐式（通过 $\varepsilon$ 采样） | 显式（解耦密度估计器） |
| **探索机制** | 外部高斯噪声 $\mathcal{N}(0, \sigma^2)$ | 批量候选采样 + MaxEnt |
| **Critic 目标** | $r + \gamma \min Q'$ | $r + \gamma (\min Q' + \alpha H)$ |
| **理论保证** | FBC 一致性约束 | MaxEnt 最优性证明 + 密度估计误差界 |

### 4.2 优势与劣势

**Online-FLAME 的优势**：
1. ✅ **原则性探索**：MaxEnt 框架提供理论保证，自动平衡探索-利用
2. ✅ **熵信号**：解耦估计器提供准确的策略熵，改进 Critic 训练质量
3. ✅ **稀疏奖励任务**：在需要大量探索的环境中表现更优
4. ✅ **多模态发现**：能够发现多个局部最优策略

**Online-FLAME 的劣势**：
1. ❌ **实现复杂度**：需要增广 ODE 求解器和 Hutchinson trace estimator
2. ❌ **计算成本**：密度估计需要 $N_{\text{est}}$ 次前向传播（训练时）
3. ❌ **超参数敏感**：温度 $\alpha$ 和密度估计步数 $N_{\text{est}}$ 需要仔细调优
4. ❌ **批量候选采样**：在高维动作空间可能低效（需要生成 $N_{\text{act}}$ 个候选）

---

## 5 针对 Fusion 控制任务的适配

### 5.1 任务特点分析

Fusion 等离子体控制的关键挑战：
- **高维连续动作空间**：多个电源/磁场线圈的同时控制
- **复杂动力学**：等离子体响应非线性、多时间尺度
- **稀疏奖励**：稳定维持时间长才有高奖励
- **多模态解**：不同放电模式对应不同最优策略
- **安全约束**：需要避免破裂等危险状态

### 5.2 混合设计建议

针对 Fusion 任务，建议采用 **渐进式混合方案**：

#### 阶段 1：早期训练（探索为主）

使用 FLAME 的 MaxEnt 机制：
```python
config_phase1 = {
    "actor_loss": "Q-reweighted",        # FLAME 风格
    "entropy_estimator": "decoupled",    # N_est=5
    "exploration": "batch_sampling",     # N_act=16
    "alpha": 1.0,                        # 高温度，鼓励探索
    "critic_entropy_reg": True,          # 熵正则化 Bellman 目标
}
```

#### 阶段 2：后期训练（利用为主）

逐渐切换到确定性策略：
```python
config_phase2 = {
    "actor_loss": "Q-reweighted",        # 保持 FLAME 框架
    "entropy_estimator": "decoupled",    # 仍需熵信号稳定 Critic
    "exploration": "epsilon_greedy",     # 减少候选采样开销
    "alpha": 0.1,                        # 降低温度，更确定性
    "alpha_decay": 0.999,                # 逐步退火
}
```

### 5.3 实现优化

#### 优化 1：降低密度估计开销

```python
class AdaptiveDensityEstimator:
    """根据训练阶段自适应调整 N_est"""
    
    def __init__(self, u_theta, N_est_init=5):
        self.u_theta = u_theta
        self.N_est = N_est_init
    
    def set_training_phase(self, progress):
        """
        progress: 训练进度 [0, 1]
        早期：N_est=10（高精度熵估计）
        后期：N_est=3（降低计算成本）
        """
        if progress < 0.3:
            self.N_est = 10
        elif progress < 0.7:
            self.N_est = 5
        else:
            self.N_est = 3
```

#### 优化 2：安全约束下的候选采样

```python
def safe_batch_sampling(s, u_theta, N_act, safety_checker):
    """
    生成满足安全约束的候选动作
    """
    candidates = []
    Q_values = []
    
    max_attempts = N_act * 5  # 防止无限循环
    attempts = 0
    
    while len(candidates) < N_act and attempts < max_attempts:
        ε = torch.randn(action_dim)
        a = ε - u_theta(ε, t=1, s=s)
        
        # 安全检查（例如：避免磁场超限）
        if safety_checker.is_safe(s, a):
            candidates.append(a)
            Q_values.append(Q_phi(s, a))
        
        attempts += 1
    
    if len(candidates) == 0:
        # 回退到安全的默认动作
        return default_safe_action(s)
    
    best_idx = np.argmax(Q_values)
    return candidates[best_idx]
```

#### 优化 3：多尺度时间编码

针对 Fusion 控制的多时间尺度特性：

```python
class MultiScaleTimeEmbedding(nn.Module):
    """多尺度正弦时间编码"""
    
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        # 不同频率捕捉快/慢动力学
        self.freqs = 2.0 ** torch.arange(0, dim // 2) * np.pi
    
    def forward(self, t):
        # t: [B, 1]
        args = t * self.freqs.to(t.device)  # [B, dim//2]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
```

---

## 6 超参数配置建议

### 6.1 网络架构

```python
velocity_net_config = {
    "hidden_dim": 512,           # 增大以捕捉复杂动力学
    "n_layers": 3,               # 深度适中
    "activation": "Mish",        # 平滑激活函数
    "time_embed_dim": 128,       # 更丰富的时间表示
    "layernorm": True,           # 稳定训练
}

critic_config = {
    "hidden_dim": 512,
    "n_layers": 3,
    "activation": "ReLU",
    "ensemble_size": 2,          # Double Q-learning
}
```

### 6.2 训练超参数

```python
training_config = {
    # Replay Buffer
    "buffer_size": 1_000_000,
    "batch_size": 256,
    
    # 学习率
    "lr_actor": 3e-4,
    "lr_critic": 3e-4,
    "lr_schedule": "cosine",     # 从 3e-4 → 3e-5
    
    # 目标网络
    "ema_tau": 0.995,
    
    # 温度参数
    "alpha_init": 1.0,
    "alpha_min": 0.01,
    "alpha_max": 10.0,
    "alpha_lr": 1e-4,            # 自适应调整学习率
    "target_entropy": -action_dim,  # SAC 默认目标熵
    
    # 密度估计
    "N_est_train": 5,            # 训练时
    "N_est_eval": 1,             # 推理时（FLAME-R 模式）
    
    # 探索
    "N_act": 16,                 # 批量候选数
    "warmup_steps": 5000,
    
    # UTD
    "utd_ratio": 1.0,            # 每步环境交互更新 1 次
}
```

---

## 7 实验评估建议

### 7.1 对比基线

1. **Online-OFQL**：当前方案
2. **SAC**：经典 MaxEnt 算法
3. **TD3**：确定性策略基线
4. **Online-FLAME（本方案）**

### 7.2 评估指标

```python
metrics = {
    # 性能指标
    "episode_return": "平均回合奖励",
    "success_rate": "稳定维持成功率",
    "disruption_rate": "破裂率（越低越好）",
    
    # 探索指标
    "policy_entropy": "策略熵（使用密度估计器）",
    "state_coverage": "状态空间覆盖率",
    "action_diversity": "动作多样性",
    
    # 效率指标
    "sample_efficiency": "达到目标性能所需样本数",
    "wall_clock_time": "实际训练时间",
    "inference_time": "单步推理延迟",
}
```

### 7.3 消融实验

```python
ablation_studies = [
    # 1. Q-reweighting 的作用
    {"name": "No_Qreweight", "config": {"use_q_reweighting": False}},
    
    # 2. 解耦熵估计的作用
    {"name": "No_Entropy", "config": {"entropy_regularization": False}},
    
    # 3. 密度估计步数
    {"name": "N_est_1", "config": {"N_est": 1}},
    {"name": "N_est_5", "config": {"N_est": 5}},
    {"name": "N_est_10", "config": {"N_est": 10}},
    
    # 4. 候选采样数
    {"name": "N_act_4", "config": {"N_act": 4}},
    {"name": "N_act_16", "config": {"N_act": 16}},
    {"name": "N_act_32", "config": {"N_act": 32}},
]
```

---

## 8 预期优势与风险

### 8.1 相对 Online-OFQL 的预期改进

✅ **理论层面**：
- MaxEnt 框架提供原则性的探索-利用权衡
- 解耦估计器提供可靠的熵信号
- Q-reweighting 保证与 MaxEnt 目标的一致性

✅ **实验层面**（根据论文结果）：
- 在稀疏奖励任务上样本效率提升 20-50%
- 多模态环境中发现更多高质量策略
- 训练稳定性提升（熵正则化防止 Critic 过估计）

### 8.2 潜在风险

❌ **实现复杂度**：
- 增广 ODE 求解器需要仔细实现（Hutchinson estimator）
- 多个超参数需要调优（$\alpha$, $N_{\text{est}}$, $N_{\text{act}}$）

❌ **计算开销**：
- 密度估计增加 $N_{\text{est}}$ 倍前向传播
- 批量候选采样增加 $N_{\text{act}}$ 倍动作生成

❌ **高维动作空间**：
- 批量候选采样在高维空间可能低效
- 需要更大的 $N_{\text{act}}$ 才能覆盖动作空间

### 8.3 降低风险的策略

1. **渐进式实施**：
   - 先实现基础 FLAME-R（无多步密度估计）
   - 验证 Q-reweighting 有效后再加入解耦估计器

2. **计算优化**：
   - 使用 Hutchinson estimator 避免完整雅可比矩阵
   - 训练后期降低 $N_{\text{est}}$ 和 $N_{\text{act}}$

3. **回退机制**：
   - 如果 MaxEnt 探索过度，提供切换到确定性模式的开关
   - 保留 OFQL 的 Flow Ratio 机制作为备选方案

---

## 9 代码框架示例

```python
class OnlineFLAME:
    """Online-FLAME 算法实现"""
    
    def __init__(self, env, config):
        self.env = env
        self.config = config
        
        # 网络
        self.velocity_net = VelocityNetwork(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            **config["velocity_net"]
        )
        self.critic = TwinCritic(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            **config["critic"]
        )
        
        # 目标网络
        self.velocity_net_target = copy.deepcopy(self.velocity_net)
        self.critic_target = copy.deepcopy(self.critic)
        
        # 解耦密度估计器
        self.entropy_estimator = AugmentedODEEstimator(
            self.velocity_net_target,
            N_est=config["N_est_train"]
        )
        
        # Replay Buffer
        self.buffer = ReplayBuffer(capacity=config["buffer_size"])
        
        # 优化器
        self.opt_actor = torch.optim.Adam(
            self.velocity_net.parameters(),
            lr=config["lr_actor"]
        )
        self.opt_critic = torch.optim.Adam(
            self.critic.parameters(),
            lr=config["lr_critic"]
        )
        
        # 温度参数（可学习）
        self.log_alpha = torch.tensor(
            np.log(config["alpha_init"]),
            requires_grad=True
        )
        self.opt_alpha = torch.optim.Adam(
            [self.log_alpha],
            lr=config["alpha_lr"]
        )
        self.target_entropy = -env.action_space.shape[0]
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def sample_action(self, s, mode="train"):
        """
        采样动作
        mode: "train" 使用批量候选采样
              "eval" 使用单步确定性生成
        """
        if mode == "train":
            return self._batch_candidate_sampling(s, self.config["N_act"])
        else:
            # FLAME-R 模式：单步生成
            with torch.no_grad():
                epsilon = torch.randn(self.env.action_space.shape[0])
                a = epsilon - self.velocity_net(
                    epsilon,
                    torch.ones(1, 1),
                    torch.tensor(s).unsqueeze(0)
                )
                return a.squeeze(0).cpu().numpy()
    
    def _batch_candidate_sampling(self, s, N_act):
        """批量候选采样（MaxEnt 探索）"""
        s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        
        candidates = []
        Q_values = []
        
        with torch.no_grad():
            for _ in range(N_act):
                epsilon = torch.randn(1, self.env.action_space.shape[0])
                a = epsilon - self.velocity_net(
                    epsilon,
                    torch.ones(1, 1),
                    s_tensor
                )
                candidates.append(a)
                Q_values.append(self.critic.Q1(s_tensor, a).item())
        
        best_idx = np.argmax(Q_values)
        return candidates[best_idx].squeeze(0).cpu().numpy()
    
    def update_critic(self, batch):
        """Critic 更新（带熵正则化）"""
        s, a, r, s_next, done = batch
        
        with torch.no_grad():
            # 生成目标动作
            epsilon_next = torch.randn_like(a)
            a_next = epsilon_next - self.velocity_net_target(
                epsilon_next,
                torch.ones(s_next.shape[0], 1),
                s_next
            )
            
            # 计算熵项
            log_prob_next = self.entropy_estimator.estimate_log_prob(
                s_next, a_next
            )
            entropy_next = -log_prob_next
            
            # Bellman 目标
            Q_target_1, Q_target_2 = self.critic_target(s_next, a_next)
            Q_target = torch.min(Q_target_1, Q_target_2)
            y = r + self.config["gamma"] * (1 - done) * (
                Q_target + self.alpha * entropy_next
            )
        
        # Critic 损失
        Q1, Q2 = self.critic(s, a)
        critic_loss = F.mse_loss(Q1, y) + F.mse_loss(Q2, y)
        
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()
        
        return critic_loss.item()
    
    def update_actor(self, batch):
        """Actor 更新（Q-Reweighted Flow Matching）"""
        s, _, _, _, _ = batch
        B = s.shape[0]
        
        actor_loss = 0
        log_probs = []
        
        for i in range(B):
            s_i = s[i:i+1]
            
            # 1. 生成终点动作
            epsilon = torch.randn(1, self.env.action_space.shape[0])
            a_1 = epsilon - self.velocity_net(
                epsilon,
                torch.ones(1, 1),
                s_i
            )
            
            # 2. 计算重要性权重
            with torch.no_grad():
                Q_val = self.critic.Q1(s_i, a_1)
                weight = torch.exp(Q_val / self.alpha)
            
            # 3. 流匹配损失
            t = torch.rand(1, 1)
            a_t = (1 - t) * epsilon + t * a_1
            
            u_pred = self.velocity_net(a_t, t, s_i)
            u_target = (a_1 - epsilon).detach()
            
            # 4. 加权损失
            fm_loss = torch.sum((u_pred - u_target) ** 2)
            actor_loss += weight * fm_loss
            
            # 记录 log π(a|s) 用于温度更新
            with torch.no_grad():
                log_prob = self.entropy_estimator.estimate_log_prob(s_i, a_1)
                log_probs.append(log_prob)
        
        actor_loss = actor_loss / B
        
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()
        
        # 温度参数更新（自适应）
        log_probs_tensor = torch.stack(log_probs)
        entropy = -log_probs_tensor.mean()
        alpha_loss = -self.log_alpha * (entropy - self.target_entropy).detach()
        
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()
        
        return actor_loss.item(), entropy.item()
    
    def update_target_networks(self):
        """EMA 更新目标网络"""
        tau = self.config["ema_tau"]
        
        for target_param, param in zip(
            self.velocity_net_target.parameters(),
            self.velocity_net.parameters()
        ):
            target_param.data.copy_(
                tau * target_param.data + (1 - tau) * param.data
            )
        
        for target_param, param in zip(
            self.critic_target.parameters(),
            self.critic.parameters()
        ):
            target_param.data.copy_(
                tau * target_param.data + (1 - tau) * param.data
            )
    
    def train(self, total_steps):
        """主训练循环"""
        s = self.env.reset()
        episode_return = 0
        
        for step in range(total_steps):
            # 环境交互
            a = self.sample_action(s, mode="train")
            s_next, r, done, _ = self.env.step(a)
            
            self.buffer.push(s, a, r, s_next, float(done))
            episode_return += r
            
            s = s_next if not done else self.env.reset()
            
            if done:
                print(f"Step {step}, Episode Return: {episode_return}")
                episode_return = 0
            
            # 梯度更新
            if len(self.buffer) > self.config["batch_size"]:
                for _ in range(self.config["utd_ratio"]):
                    batch = self.buffer.sample(self.config["batch_size"])
                    
                    critic_loss = self.update_critic(batch)
                    actor_loss, entropy = self.update_actor(batch)
                    self.update_target_networks()
                    
                    if step % 1000 == 0:
                        print(f"  Critic Loss: {critic_loss:.3f}, "
                              f"Actor Loss: {actor_loss:.3f}, "
                              f"Entropy: {entropy:.3f}, "
                              f"Alpha: {self.alpha:.3f}")
```

---

## 10 总结与展望

### 10.1 核心贡献

Online-FLAME 相比 Online-OFQL 的核心改进：

1. **理论完备性**：通过 Q-reweighted 流匹配实现原则性的最大熵强化学习
2. **熵估计**：解耦的密度估计器提供准确的策略熵信号
3. **探索机制**：批量候选采样 + MaxEnt 自动平衡探索与利用
4. **Critic 改进**：熵正则化的 Bellman 目标提高价值估计质量

### 10.2 适用场景

✅ **推荐使用 Online-FLAME**：
- 稀疏奖励环境（需要大量探索）
- 多模态奖励景观（存在多个局部最优）
- 复杂动力学系统（需要精确价值估计）
- 对样本效率要求高于推理速度

⚠️ **谨慎使用 Online-FLAME**：
- 极高维动作空间（批量候选采样低效）
- 对推理延迟要求极苛刻（密度估计有开销）
- 简单确定性任务（MaxEnt 的探索优势不明显）

### 10.3 未来改进方向

1. **计算优化**：
   - 使用神经 ODE 求解器加速密度估计
   - 探索低秩近似减少雅可比矩阵计算

2. **架构创新**：
   - 引入 Transformer 捕捉长期依赖
   - 多模态策略网络（混合专家模型）

3. **理论扩展**：
   - 多步 bootstrapping（n-step returns）
   - 分层强化学习（hierarchical RL）

4. **工程实践**：
   - 分布式训练（Replay Buffer sharding）
   - 在线超参数调优（自适应 $\alpha$, $N_{\text{est}}$）
