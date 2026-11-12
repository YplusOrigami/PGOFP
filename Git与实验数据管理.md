# Git 与实验数据管理

2026-11-13 已在项目根目录初始化本地 Git 仓库，初始分支为 main。没有自动创建远程 GitHub 仓库或上传数据。

## 跟踪范围

| 内容 | 处理方式 |
|---|---|
| 源码、配置、脚本、文档、测试 | 普通 Git |
| outputs 中的 JSONL、CSV、JSON/YAML、日志、图表 | 普通 Git |
| outputs 中的 TensorBoard 事件文件 | 普通 Git |
| artifacts 中冻结先验的权重、归一化器、配置、manifest、预算、校准记录 | 完整纳入普通 Git，包括 .pt 与 .npz |
| outputs 下的 checkpoints 目录（训练 checkpoint、replay 及对应元数据） | 忽略，保留本地/服务器；未来需要版本化时使用 Git LFS 或独立存储 |
| WandB SDK 本地镜像、运行锁、临时写入、绘图缓存 | 忽略 |
| .env、.venv、Python 缓存、macOS ._ 元数据 | 忽略 |

实际规则见 `.gitignore`。本次没有为全部 outputs 二进制文件启用 Git LFS，也没有删除或移动本地 checkpoint。

## 本次实测体积

忽略 macOS `._` 元数据文件，按文件实际字节计：

- outputs 全部实际文件约 3.76 GiB；其中 .pt 与 .npz 合计约 3.55 GiB。
- artifacts 中现有冻结先验共约 200 KiB，四个制备阶段均保留。
- 应用上述规则后，outputs 约 41.55 MiB；整个项目候选跟踪内容约 65 MiB。
- 计划跟踪的最大单文件约 1.06 MiB，没有超过 100 MiB 的文件。

磁盘的 `du` 占用包含外置盘块分配开销及元数据，可能大于上述文件字节总量。Git 压缩后体积也与文件总量不同。

## GitHub 限制

根据 2026-11-13 查询的 GitHub 官方文档：

- 普通 Git 单文件大于 50 MiB 会警告，大于 100 MiB 会拒绝；本次纳入的文件均明显低于该值。[大文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- 单次普通 Git push 的限制为 2 GB；仓库的推荐 on-disk 上限为 10 GB，指压缩后的 .git 目录，并不是所有工作区文件之和的硬限制。[仓库限制](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits)
- Git LFS 对 Free/Pro 当前包含 10 GiB 存储和 10 GiB 下载带宽；额度由仓库所有者共享使用，并非每个仓库单独拥有。二进制文件每个新版本会占用新的存储空间。[LFS 额度与计费](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)

按当前规则，一次初始提交的候选内容只有约 65 MiB，体积不会触及上述上传限制。此结论针对当前文件，不保证后续大量训练历史不会增长。若未来要把 checkpoint/replay 也版本化，应先配置 Git LFS，再加入这些大文件，避免先将其提交到普通 Git 历史。

## 同步与恢复边界

Git clone 可恢复代码、实验数值、TensorBoard 曲线与完整冻结先验。训练恢复所需的 last checkpoint 和 replay 仍需从服务器或其他备份单独取得。数值数据持续变化后，需要再次 git add/commit 才会保存新的版本；加入 Git 不会自动上传到 GitHub。
