# Biped Robot with SAC (Soft Actor-Critic)

这是一个基于 **PyBullet + Gymnasium + Stable-Baselines3 (SAC)** 的二足机器人环境与强化学习训练项目。  
项目包含：
- 自定义 **二足机器人 URDF 模型**（躯干 + 左右大腿、小腿、脚）
- 封装的 **Gymnasium 环境**（支持步态奖励 shaping）
- 使用 **SAC 算法** 的训练脚本（含 VecNormalize、Monitor、TensorBoard 支持）

## 项目结构

```
.
├── biped.urdf             # 二足机器人 URDF 模型
├── envs/
│   └── biped_env.py       # 自定义 Gymnasium 环境
├── train_sac.py           # 训练脚本
├── models/                # 训练后保存的模型与 VecNormalize 统计量
├── logs/                  # TensorBoard 日志
└── README.md              # 本文档
```
## 主要特点：
- **自定义机器人模型**：包含躯干、左右大腿、小腿和脚，支持多关节控制。  
- **Gymnasium 接口**：观测空间涵盖机器人姿态、高度、关节角度与速度，动作空间为关节角度增量控制。  
- **奖励函数设计**：结合前向速度、直线保持、稳定性、动作平滑与生存奖励，帮助机器人学习自然步态。  
- **强化学习训练**：采用 Soft Actor-Critic (SAC) 算法，支持自动温度调节、经验回放、归一化处理。  
- **可视化与调试**：支持 TensorBoard 日志与 PyBullet GUI 渲染，便于训练过程监控与结果展示。  

## 应用场景：
- 二足机器人运动控制研究  
- 强化学习在机器人控制中的应用实验  
- 教学与课程实验（机器人学、强化学习相关课程）
