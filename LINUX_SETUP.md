# Linux 开发环境

配置日期：2026-10-31。项目从 Windows 分区复制，原目录保留。

- Linux 项目：`/home/melin/Projects/Wheel_Leg`
- 原项目：`/media/melin/35F95F8C4A48DCED/ds&codex&agr/Wheel_Leg`
- Python 环境：项目内 `.venv-linux`，使用 Python 3.12。
- GPU：NVIDIA GeForce RTX 5080，约 16 GB 显存。

复制保留 `.git`、源码、模型及 `RL/experiments` 实验数据；跳过 Windows
`.venv`。两份项目是独立副本，之后不会自动同步。

## 打开项目和激活环境

```bash
cd /home/melin/Projects/Wheel_Leg
source .venv-linux/bin/activate
unset PYTHONPATH
python -c "import sys; print(sys.executable)"
```

最后一条应显示 `/home/melin/Projects/Wheel_Leg/.venv-linux/bin/python`。
VS Code 中打开这个项目文件夹，再通过 `Python: Select Interpreter` 选择该路径。
不要选择 Windows 分区内的 `.venv/Scripts/python.exe`。

当前系统的终端会通过 `PYTHONPATH` 加入 `/opt/ros/jazzy`。上面的 `unset`
只移除当前终端的额外 Python 搜索路径，避免把系统 ROS 包混入本项目依赖检查，
不会删除 ROS 或修改全局配置。需要运行 ROS 时请使用另外的终端。
不想修改当前终端变量时，也可以在 Python 命令前加 `env -u PYTHONPATH`。

## 重建依赖

以下命令用于需要重新建立环境时；已经配置好的环境无需反复安装。

```bash
cd /home/melin/Projects/Wheel_Leg
python3 -m venv .venv-linux
.venv-linux/bin/python -m pip install --upgrade pip
.venv-linux/bin/python -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
.venv-linux/bin/python -m pip install -r requirements-linux.txt
env -u PYTHONPATH .venv-linux/bin/python -m pip check
```

`requirements-linux.txt` 固定主要依赖，`requirements-linux-lock.txt` 记录验证环境的完整依赖版本。
安装 PyTorch 使用官方 CUDA 12.8 软件源。其他依赖使用 PyPI。
运行预编译 PyTorch 不需要另外安装系统 CUDA Toolkit。

本机 `nvidia-smi` 显示驱动 595.99.02、CUDA 13.2；后者是驱动支持的 CUDA
版本信息，不要求 PyTorch 使用相同版本。此处使用支持 Blackwell 的 CUDA 12.8
PyTorch 构建，不修改或降级系统 NVIDIA 驱动。参考
[PyTorch 官方 Blackwell / CUDA 12.8 说明](https://pytorch.org/blog/pytorch-2-7/)。

## 运行现有程序

原 LQR/VMC 控制器需要桌面显示，且在 `LQR_VMC` 目录启动：

```bash
cd /home/melin/Projects/Wheel_Leg/LQR_VMC
../.venv-linux/bin/python robot_control_main.py
```

直接策略的无窗口短评估（不更新或保存权重）：

```bash
cd /home/melin/Projects/Wheel_Leg
.venv-linux/bin/python RL/v3_direct/direct_navigation.py --evaluate --checkpoint RL/v3_direct/data/results/direct_imitation_candidate.pt --target-x 0.25 --target-y 0 --seconds 2
```

要显示上述策略的仿真窗口，可增加 `--viewer`。该直接策略仍是原有实验模型，
能运行不代表已经学会可靠到点停稳。

2026-10-31 已按用户要求移除项目内的 6 个独立自动测试脚本。整理时在项目外临时运行原有 16 项检查，全部通过。当前入口见 `RL/README.md`。

## 工作范围

本次配置安装 rsl_rl，尚未将现有机器人环境接入其训练接口，尚未实现新的
PD 动作接口，也没有开始新一轮训练。已有训练脚本不会因安装了 CUDA 版 PyTorch
而自动把模型移到 GPU；现有普通 MuJoCo 仿真也不会自动成为 GPU 并行仿真。

`handoff.md` 是较早的教学交接。当前训练状态以 `RL/docs/TRAINING_LOG.md` 和
`RL/docs/TRAINING_CAMPAIGN_20261029.md` 为准。已有研究报告中的 Windows 绝对路径
是历史记录，不是本机启动路径。

## 本次验证结果

- Python 3.12.3、PyTorch 2.11.0+cu128、MuJoCo 3.15.0、rsl-rl-lib 5.5.2。
- RTX 5080 上实际执行策略网络前向和反向计算，结果与梯度有限。
- 原机器人模型加载、物理步进、EGL 离屏渲染通过。
- 桌面 GLFW 上下文创建通过（隐藏窗口），未完整测试键盘/手柄交互。
- 基线权重加载、推理通过；无窗口运行 2 秒保持站立，尚未停稳。
- 清除外部 ROS PYTHONPATH 后，全部 16 项现有回归测试通过，pip check 通过。
- 332 个 RL 文件在验证前与原目录逐一校验一致，原基线权重 SHA256 不变。
- 本次只配置环境，没有开始新的策略训练，没有提交 Git。

机器可读记录：`RL/archive/data/results/linux_environment_20261031.json`。

Linux 副本的 Git 本地设置使用 `core.autocrlf=input`，识别原 Windows 文件的 CRLF
换行，避免无意义的整文件差异；原项目 Git 配置不变。已有 14 个被跟踪的缓存
差异随副本保留，没有清理。
