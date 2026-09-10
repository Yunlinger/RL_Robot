# SG90 十舵机二足机器人强化学习

这是一个可在 macOS 上运行的 PyBullet + Gymnasium + Stable-Baselines3 项目，用 SAC 学习小型二足机器人的步态。执行器模型按 SG90/MG90S 类 9g 舵机做了保守的力矩、速度和动作限幅。

当前设计目标是先做轻量实物，再做 sim-to-real。普通 SG90 的扭矩会受电压、温度、安装角度、电池电量和机械摩擦影响，仿真参数必须在实物完成后重新标定。

## 机器人设计

每条腿 5 个自由度，共 10 个舵机：

| 关节 | 作用 |
| --- | --- |
| `hip_roll` | 左右方向的重心控制 |
| `hip_pitch` | 大腿前后摆动 |
| `knee` | 膝关节弯曲 |
| `ankle_pitch` | 脚掌前后调平 |
| `ankle_roll` | 脚掌左右调平 |

参考参数集中在 [robot_config.py](robot_config.py)：大腿和小腿各约 55 mm，髋部间距约 60 mm，标称总质量约 250 g，单关节假设最大约 0.08 N·m、速度约 6 rad/s。仿真使用 500 Hz 物理步进、50 Hz 控制频率。策略输出 10 个关节的有界残差，基础步态由 [envs/gait.py](envs/gait.py) 生成。

![参考机器人姿态](docs/reference_robot.png)

![参考步态](docs/reference_gait.gif)

## macOS 安装

建议 Python 3.10–3.12：

```bash
cd /Users/lvshaofei/Documents/ChatGPT/robotRL
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

检查 PyTorch 是否能使用 Apple MPS：

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
PY
```

这个项目的 MLP 很小，默认使用 CPU 通常更稳定。需要时可以显式指定 `--device mps`。

## 先检查和预览

完整回归测试：

```bash
python -m unittest discover -s tests -v
```

预览零残差参考步态并导出 GIF：

```bash
python scripts/preview.py --gif docs/my_reference.gif
```

直接打开 PyBullet GUI：

```bash
python scripts/preview.py
```

## 训练

先跑小实验确认环境：

```bash
python train_sac.py \
  --steps 20000 \
  --eval-freq 5000 \
  --eval-episodes 3 \
  --run-dir runs/smoke
```

正式训练建议至少从 400,000 步开始：

```bash
python train_sac.py \
  --steps 400000 \
  --eval-freq 10000 \
  --eval-episodes 5 \
  --run-dir runs/walk_sg90
```

Apple Silicon 上确认 MPS 可用后：

```bash
python train_sac.py --device mps --steps 400000 --run-dir runs/walk_sg90_mps
```

加入质量、摩擦、舵机强度、延迟和 IMU 噪声随机化，提高迁移鲁棒性：

```bash
python train_sac.py \
  --steps 800000 \
  --randomize \
  --eval-freq 20000 \
  --run-dir runs/walk_sg90_randomized
```

每个运行目录会保存 `final/model.zip`、`final/vecnormalize.pkl`、`final/replay_buffer.pkl`、`final/config.json`、`final/evaluation.json`，以及中间 checkpoint、`evaluations.jsonl` 和 `best.json`。

## 续训和评估

使用带 replay buffer 的 bundle 续训：

```bash
python train_sac.py \
  --resume runs/walk_sg90/final \
  --steps 400000 \
  --run-dir runs/walk_sg90_resume
```

无 GUI 评估：

```bash
python test.py \
  --model runs/walk_sg90/final \
  --episodes 10 \
  --output runs/walk_sg90/evaluation_10.json
```

GUI 回放：

```bash
python test.py --model runs/walk_sg90/final --episodes 1 --gui
```

加 `--require-walking` 可以让命令在没有任何成功步行 episode 时返回退出码 1。评估的 `walking_success` 同时检查未摔倒、平均速度、横向漂移和抬脚/触地次数，不能只看 reward。

## 实物制作建议

四舵机模型缺少左右重心控制，建议按当前十舵机结构制作。舵机使用独立 5 V 电源，额定电流建议至少 3 A；舵机电源地必须和主控 GND 共地，不要从开发板的小电流 5 V 接口给十个舵机供电。

首次上电时把舵机动作限制在中位附近，逐个确认零位、方向和机械干涉。实物建议增加 IMU（陀螺仪 + 加速度计）；普通 9g 舵机通常没有可用的位置反馈，因此策略输入按 IMU、步态相位、速度指令和舵机指令历史设计。连接真实舵机前，要测量整机质量、站立电流、单腿抬脚力矩和低电压下的舵机速度。

修改尺寸或质量后重新生成 URDF 并测试：

```bash
python scripts/build_urdf.py
python -m unittest discover -s tests -v
```

## 代码结构

| 文件 | 用途 |
| --- | --- |
| [robot_config.py](robot_config.py) | 尺寸、质量、力矩和速度假设 |
| [scripts/build_urdf.py](scripts/build_urdf.py) | 从配置生成十舵机 URDF |
| [biped.urdf](biped.urdf) | PyBullet 模型 |
| [envs/gait.py](envs/gait.py) | 参考步态和腿部逆运动学 |
| [envs/biped_env.py](envs/biped_env.py) | 环境、接触、奖励和跌倒判定 |
| [training.py](training.py) | bundle、VecNormalize 和评估 |
| [train_sac.py](train_sac.py) | 训练、评估 checkpoint、最终保存 |
| [test.py](test.py) | 独立评估和 GUI 回放 |
| [tests/](tests/) | 环境和训练烟测 |

## 验证结论

本版本在 macOS 上已通过 9 项自动检查，包括 Gymnasium/SB3 环境契约、随机种子、独立物理客户端、舵机限幅、跌倒与超时区分、随机化 reset、渲染、模型保存/恢复和 replay buffer 续训。零残差参考步态可在仿真中连续运行 12 秒，约前进 0.28 m 且不摔倒。SAC 短训练可以正常更新网络并恢复 bundle；正式步态需要按训练曲线和 `evaluation.json` 判断。

## Git

本地提交：

```bash
git status
git diff --check
git add README.md biped.urdf envs robot_config.py scripts tests training.py train_sac.py test.py requirements.txt .gitignore docs
git commit -m "Rebuild biped training pipeline for ten SG90 servos"
git log --oneline -1
```

`runs/`、`models/`、`logs/` 和 Python 缓存已加入 `.gitignore`。提交只更新本地 Git；确认无误后再执行：

```bash
git push origin main
```
