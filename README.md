# Biped Robot with SAC (Soft Actor-Critic)

这是一个基于 **PyBullet + Gymnasium + Stable-Baselines3 (SAC)** 的二足机器人环境与强化学习训练项目。  
项目包含：
- 自定义 **二足机器人 URDF 模型**（躯干 + 左右大腿、小腿、脚）
- 封装的 **Gymnasium 环境**（支持步态奖励 shaping）
- 使用 **SAC 算法** 的训练脚本（含 VecNormalize、Monitor、TensorBoard 支持）

## 项目结构

```text
.
├── biped.urdf             # 二足机器人 URDF 模型
├── envs/
│   └── biped_env.py       # 自定义 Gymnasium 环境
├── train_sac.py           # 训练脚本
├── models/                # 训练后保存的模型与 VecNormalize 统计量
├── logs/                  # TensorBoard 日志
└── README.md              # 本文档
