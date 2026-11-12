# Online-OFQL for fusion 设计

## 1 OFQL 算法

### 1.1 算法框架

OFQL 将 DQL 的策略建模替换为平均速度参数化 $u_\theta$，其余部分（Critic 结构、actor-critic 框架）保持不变。

**完整损失函数**：

$$\mathcal{L}(\phi) = \mathbb{E}_{(s,a,r,s')\sim\mathcal{D},\, a'\sim\pi_{\theta'}} \left[\left(r + \gamma \min_{i\in\{1,2\}} Q_{\phi'_i}(s', a') - Q_{\phi_i}(s, a)\right)^2\right]$$

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{FBC}}(\theta) - \alpha \cdot \mathbb{E}_{s\sim\mathcal{D},\, a\sim\pi_\theta} \left[Q_\phi(s, a)\right]$$

**单步动作采样**：
- Critic 更新时，目标动作 $a' \sim \pi_{\theta'}$ 通过单步生成
- Actor 更新时，$a \sim \pi_\theta$ 通过单步生成

### 1.2 Flow Ratio

训练时对时间对 $(t, r)$ 的采样引入 **flow ratio** $\lambda$，即 $t = r$ 的概率。当 $t = r$ 时，平均速度退化为瞬时速度，bias 训练更多地学习瞬时速度（有助于 bootstrapping 目标的准确估计）。

默认 flow ratio = 0.5，保留一半样本用于学习平均速度，另一半用于正则化瞬时速度。消融实验表明 flow ratio = 0.5 最优；设为 1（纯流匹配）或 0 均导致性能下降。

## 2 Online版本核心思想

Online-OFQL 将策略表示为**平均速度场** $u_\theta(a_t, r, t; s)$，通过单步端点映射生成动作：

$$a = \varepsilon - u_\theta(\varepsilon,\ r{=}0,\ t{=}1;\ s), \quad \varepsilon \sim \mathcal{N}(0, I)$$

训练分两阶段：
1. **热身阶段**：随机动作填充 replay buffer，不更新网络
2. **在线学习**：每步环境交互后执行 grad_steps 次梯度更新

每次梯度更新包含：
- **Critic 更新**：Double Q-learning + EMA 目标网络
- **Actor 更新**：FBC（Flow-Based Consistency） 损失（MeanFlow Identity）+ 自适应 Q 引导

相比offline版本的改动：
- offline版本使用固定数据集而 online 版本使用 replay buffer 更新梯度（来实现在线学习）。
- 为保证exploration程度探索其他状态，加入了高斯探索噪声（训练后期可减少噪声权重“退火”操作）。
- 为避免训练初期 replay buffer 中数据过少导致过拟合，训练正式开始前先进行 warm-up，先交互一定步数填充 replay buffer。




## 3 实现逻辑

### 3.1 变量说明

| 符号 | 含义 | 取值/维度 |
|------|------|-----------|
| $\mathcal{S}$ | 状态空间 | — |
| $\mathcal{A}$ | 动作空间 | $\subseteq \mathbb{R}^{d_a}$ |
| $s, s'$ | 当前状态、下一状态 | $\in \mathcal{S}$ |
| $a, a'$ | 当前动作、下一动作 | $\in \mathcal{A}$ |
| $r$ | 即时奖励（reward） | $\in \mathbb{R}$ |
| $\gamma$ | 折扣因子 | $\in [0, 1)$ |
| $\mathcal{D}$ | 离线数据集（固定） | $\{(s, a, r, s')\}$ |
| $\mathcal{B}$ | Replay Buffer（动态，Online 专用） | $\{(s, a, r, s', \text{done})\}$ |
| $\theta$ | 策略网络参数（Actor） | — |
| $\theta'$ | 目标策略网络参数（EMA 更新） | — |
| $\phi_i$ | 第 $i$ 个 Q 网络参数（Critic） | $i \in \{1, 2\}$ |
| $\phi'_i$ | 目标 Q 网络参数（EMA 更新） | — |
| $\rho$ | EMA 衰减率（目标网络更新速率） | $\in (0,1)$ |
| $\eta$ | Q 权重调节系数（超参数） | $\in \{0.001, 0.01, 0.1, 0.3, 0.5\}$ |
| $\alpha$ | 自适应 Q 权重（运行时计算） | $= \eta / \mathbb{E}[\|Q_\phi\|]$ |
| $B$ | mini-batch 大小 | 典型值 $256$ |
| $\mathcal{B}$ | Replay Buffer（环形队列） | 最大容量 $N$，存储 $(s, a, r, s', \text{done})$ |
| $N$ | Replay Buffer 容量 | - |
| $T_{\text{warm}}$ | 热身步数（随机策略填充 Buffer） | - |
| $G$ | 每个环境步对应的梯度更新次数 | 基本是1次 |
| $\sigma_{\text{explore}}$ | 探索噪声标准差 | 典型值 $0.1$，可随训练退火 |
| $\text{done}$ | 回合终止标志 | $\in \{0, 1\}$；$1$ 表示回合结束 |
| $\text{clip}(a, A)$ | 动作裁剪到合法范围 | $a \leftarrow \max(A_{\text{low}}, \min(A_{\text{high}}, a))$ |


### 3.2 Online-OFQL 伪代码

```
Online-OFQL 训练
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入：
  env        — 在线环境（支持 reset() 和 step(a)）
  N          — Replay Buffer 容量
  T_warm     — 热身步数
  G          — 每步梯度更新次数
  σ          — 探索噪声标准差
  λ          — Flow Ratio（默认 0.5）
  ρ          — EMA 衰减率
  η          — Q 权重调节超参数
  B          — mini-batch 大小

初始化：
  随机初始化平均速度网络参数 θ
  随机初始化 Q 网络参数 φ₁, φ₂
  θ' ← θ，φ'₁ ← φ₁，φ'₂ ← φ₂
  B_buf ← ReplayBuffer(capacity=N)     // 空 Replay Buffer
  s ← env.reset()

━━━━━━━━━━━━━━ 阶段一：热身（随机探索） ━━━━━━━━━━━━━━━━
// 目的：预填 Replay Buffer，避免初期训练过拟合极少数据
// 此阶段不更新任何网络参数

for step = 1 to T_warm:
    a ← UniformRandom(A_low, A_high)       // 完全随机动作
    s', r, done ← env.step(a)
    B_buf.push((s, a, r, s', done))
    s ← done ? env.reset() : s'

━━━━━━━━━━━━━━ 阶段二：在线学习 ━━━━━━━━━━━━━━━━━━━━━━━━
for each environment step:

    // ══════════════════════════════════════════════════════
    // 步骤 1：与环境交互，收集数据
    // ══════════════════════════════════════════════════════

    // 1a. 用当前策略生成动作（带探索噪声）
    ε ← N(0, I)
    a_det = ε - u_θ(ε, r=0, t=1; s)          // 单步确定性动作
    // u_θ 的随机性来自 ε 的随机采样，但单次采样是确定性的
    // 叠加外部探索噪声以鼓励探索
    a_explore ← N(0, σ²I)
    a = clip(a_det + a_explore, A_low, A_high)
    // clip: 将动作限制在合法范围内

    // 1b. 执行动作，收集转移
    s', r, done ← env.step(a)
    B_buf.push((s, a, r, s', done))
    s ← done ? env.reset() : s'

    // ══════════════════════════════════════════════════════
    // 步骤 2：梯度更新（G 次）
    // ══════════════════════════════════════════════════════
    for _ = 1 to G:

        // 从 Replay Buffer 随机采样 mini-batch
        {(sᵢ, aᵢ, rᵢ, s'ᵢ, doneᵢ)} ← B_buf.sample(B)
        // sᵢ:    当前状态
        // aᵢ:    实际执行的动作（含探索噪声）
        //        与 Offline 不同，aᵢ 来自带噪策略，是近似行为数据
        // rᵢ:    即时奖励
        // s'ᵢ:   下一状态
        // doneᵢ: 回合终止标志

        // ── Critic 更新 ──────────────────────────────────
        for each i in batch:
            ε'ᵢ ← N(0, I)
            a'ᵢ = ε'ᵢ - u_θ'(ε'ᵢ, r=0, t=1; s'ᵢ)   // 目标策略单步生成

        // Bellman 目标
        yᵢ = rᵢ + γ * (1 - doneᵢ) * min_{j=1,2} Q_{φ'_j}(s'ᵢ, a'ᵢ)
        // (1 - doneᵢ): 回合结束时无未来回报

        L_critic(φ) = (1/B) * Σᵢ Σ_{j=1,2} (Q_{φ_j}(sᵢ, aᵢ) - yᵢ)²
        φ₁, φ₂ ← φ₁, φ₂ - ∇_φ L_critic(φ)

        // ── Actor 更新 ────────────────────────────────────

        // FBC 损失（使用 Replay Buffer 中的行为数据）
        // 近期访问分布
        L_FBC = 0
        for each i in batch:
            // 采样时间对 (t, r)
            with probability λ:
                t_i = r_i ← LogitNormal(-0.4, 1.0)
            else:
                t_i, r_i ← LogitNormal(-0.4, 1.0)，保证 r_i ≤ t_i

            εᵢ ← N(0, I)
            a_t_i = (1 - t_i) * aᵢ + t_i * εᵢ     // 线性插值
            v_t_i = εᵢ - aᵢ                         // 条件瞬时速度

            // MeanFlow Identity 计算目标
            Jv_i = JVP(u_θ(a_t_i, r_i, t_i; sᵢ), wrt=a_t_i, vec=v_t_i)
            du_dt_i = ∂u_θ(a_t_i, r_i, t_i; sᵢ) / ∂t_i
            u_tgt_i = v_t_i - (t_i - r_i) * (Jv_i + du_dt_i)

            L_FBC += ‖u_θ(a_t_i, r_i, t_i; sᵢ) - sg(u_tgt_i)‖²

        L_FBC = L_FBC / B

        // Q 引导（单步采样，无 BPTT）
        for each i in batch:
            ε̃ᵢ ← N(0, I)
            ãᵢ = ε̃ᵢ - u_θ(ε̃ᵢ, r=0, t=1; sᵢ)


        // Online 中分母使用 Replay Buffer 的滑动估计值
        α = η / E_{(s,a)~B_buf}[‖Q_φ(s, a)‖]

        // Actor 总损失
        L_actor(θ) = L_FBC(θ) - α * (1/B) * Σᵢ Q_φ(sᵢ, ãᵢ)
        //                      ↑ Q 引导（策略改进）

        θ ← θ - ∇_θ L_actor(θ)

        // EMA 更新目标网络
        θ'   ← ρ * θ'   + (1 - ρ) * θ
        φ'_j ← ρ * φ'_j + (1 - ρ) * φ_j    for j = 1, 2
```

### 3.3 推理伪代码

```
Online-OFQL 推理（评估模式）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入：状态 s，已训练的平均速度网络 u_θ

ε ← N(0, I)
a = ε - u_θ(ε, r=0, t=1; s)    // 单步生成，不加探索噪声

输出：a
```

