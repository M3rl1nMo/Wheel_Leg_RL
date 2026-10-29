# Wheeled Bipedal Robot Simulation & Control

本项目是一个基于 **MuJoCo** 物理引擎的轮足双足机器人仿真与控制框架。机器人采用五连杆腿部机构 + 轮毂电机的构型，结合 **LQR**（线性二次型调节器）进行轮式倒立摆平衡控制，以及 **VMC**（虚拟模型控制）进行腿长与姿态控制，并支持通过 Xbox 手柄或键盘进行实时遥控。

在原有 LQR + VMC 控制器基础上，本仓库新增了 **PyTorch 策略网络、强化学习教学实验、LQR 模仿学习及网络轨迹数据聚合训练**。原控制器仍可单独运行；学习控制代码位于 `RL/`。

**当前成果：** 保持单隐藏层 `20 → 32 → 6`、SiLU 激活函数，通过 LQR 动作监督和三轮轨迹数据补充，策略网络在 5 个固定测试姿态和额外 10 个小幅倾斜姿态下均完成了 **3 秒不摔倒**。这是短时仿真验证，尚未验证长期稳定、较大扰动、行走和实机控制。详细过程见 [训练日志](RL/TRAINING_LOG.md)。

## 目录结构

```text
wheel_bipedal_robot/
├── LQR_VMC/                        # [核心] LQR + VMC 联合主控制代码
│   ├── robot_control_main.py       #   主程序入口，封装左右腿控制函数，含离地保护逻辑
│   ├── lqr_update.py               #   LQR 控制器：12维状态更新、多项式实时计算 K 矩阵
│   ├── vmc_cal.py                  #   五连杆正运动学 + VMC 雅可比力矩映射
│   ├── roll_leg_control.py         #   横滚平衡控制器（PID 差模分配左右腿长）
│   ├── ground_detection.py         #   单腿离地检测（基于虚拟支持力阈值）
│   ├── mjcf/                       #   MuJoCo 模型文件
│   │   ├── robot.xml               #     机器人 URDF/MJCF 模型
│   │   └── obstacles.xml           #     障碍物场景文件
│   ├── param_cal/                  #   机器人动力学参数计算
│   │   └── robot_param_cal.py      #     从 XML 提取质量/惯量/半径等参数
│   ├── remote/                     #   遥控输入模块
│   │   └── xbox_controller.py      #     封装 PygameController 类（键盘 + Xbox 手柄）
│   ├── robot_mujoco_sensor_feedback/  # 传感器反馈处理
│   │   └── robot_info_update.py    #     统一读取 6 个关节 + IMU 数据
│   └── util/                       #   工具类
│       └── climb_traj.py           #     线性速度爬坡滤波器
│
├── robot_vmc_test/                 # [验证] VMC 五连杆运动学独立验证
│   ├── vmc_verification.py         #   位置环 + 逆运动学分析，验证零点标定与腿长控制
│   └── robot_pos_ctrl_for_vmc.xml  #   位置控制模式的专用模型文件
│
├── robot_feedback_test/            # [验证] 传感器反馈测试
│   ├── plot_sensors.py             #   实时绘制电机位置/速度/IMU 欧拉角的遥测工具
│   ├── quat_euler_trans.py         #   四元数 ↔ 欧拉角转换测试
│   └── robot.xml                   #   MuJoCo 模型文件
│
├── RL/                             # [新增] 学习控制与训练实验
│   ├── networks.py                 #   策略网络和价值网络
│   ├── observations.py             #   20 维观测及固定尺度缩放
│   ├── observe_robot.py            #   单步 Actor–Critic 教学实验
│   ├── actor_critic_demo.py        #   手写数据的梯度更新演示
│   ├── checkpoints.py             #   奖励训练存档保存与恢复
│   ├── training_metrics.py        #   回合 CSV 记录和滑动平均
│   ├── lqr_imitation.py           #   LQR 数据采集、模仿训练和策略评估
│   ├── lqr_dagger.py              #   网络轨迹上的 LQR 标注与再训练
│   ├── TRAINING_LOG.md            #   实验过程、结果和局限
│   ├── results/                   #   已提交的策略模型和验证报告
│   └── experiments/               #   本地生成的数据和存档，不纳入 Git
│
├── cal_lqr_k.m                 #   MATLAB LQR 整定与多项式拟合，生成 POLY_COEFFICIENT
└── README.md
```

## 环境依赖

原 LQR + VMC 项目基于 **Python 3.10.12+、Ubuntu 22.04** 开发。新增学习控制实验已在 **Windows、Python 3.12、CPU** 环境运行，无须为了这些实验切换到 Linux。

原控制器依赖：

```bash
pip install mujoco
pip install numpy
pip install scipy
pip install pygame
pip install matplotlib   # 用于遥测绘图（可选）
```

学习控制模块额外依赖 `torch`。建议在项目根目录创建独立虚拟环境。Windows PowerShell 示例：

```powershell
py -3.12 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install mujoco numpy scipy pygame torch
# 仅需要传感器绘图时安装
& '.\.venv\Scripts\python.exe' -m pip install matplotlib
```

已有 `.venv` 时可跳过创建步骤。下方 Windows 命令直接使用虚拟环境解释器，无须激活环境。实验记录中的版本为 PyTorch `2.14.1+cpu`、MuJoCo `3.15.0`；不同依赖版本可能影响数值结果，安装命令未锁定版本。

## 快速开始

### 1. 运行主仿真程序

主程序位于 `LQR_VMC` 目录：

```bash
cd LQR_VMC
python robot_control_main.py
```

### 2. 操控说明

程序启动后会打开 MuJoCo 查看器和一个 Pygame 控制面板。

| 操作                     | 键盘                    | Xbox 手柄                    |
|-------------------------|-------------------------|------------------------------|
| 前进 / 后退              | ↑ / ↓ 方向键           | 左摇杆上下                     |
| 左转 / 右转              | ← / → 方向键           | 右摇杆左右                     |
| 增加 / 减少目标腿长       | + / -                   | RB / LB（肩键）               |
| 暂停 / 继续仿真           | 空格键                  | —                            |
| 开启 / 关闭离地检测保护    | 回车键 (Enter)          | 左摇杆按键                      |

**离地检测保护**：系统会实时估计足端虚拟支持力。当检测到某条腿离地时，自动将该侧轮毂力矩清零、髋关节切换为空中阻尼姿态维持模式，防止因轮子悬空误驱动导致姿态失控。

## 新增工作：策略网络与 LQR 模仿学习

### 直接查看已训练策略

仓库包含本次验证模型，无须先训练。在**项目根目录**运行：

```powershell
& '.\.venv\Scripts\python.exe' '.\RL\lqr_imitation.py' --evaluate --viewer --checkpoint '.\RL\results\lqr_dagger_policy.pt'
```

- `--evaluate`：加载策略并评估，不更新参数。
- `--viewer`：打开 MuJoCo 窗口；去掉此参数可无窗口运行。
- `--checkpoint`：指定本次提交的策略文件。

该演示由网络独立输出 `tanh(mean) × torque_limits`，不随机采样动作。LQR 在脚本内部可以计算标签，但其输出不会下发到电机。每次最多运行 3 秒，摔倒则提前结束，不提供原控制器的键盘/手柄遥控。

### 网络输入与输出

策略网络使用一个 32 单元隐藏层，结构为 `20 → 32 → 6`，隐藏层激活函数为 SiLU。20 个输入由以下部分组成：

| 观测内容 | 数量 |
| --- | ---: |
| 俯仰角、横滚角 | 2 |
| 三轴角速度 | 3 |
| 机身高度 | 1 |
| 三轴线速度 | 3 |
| 四个腿部电机关节角度 | 4 |
| 六个电机角速度 | 6 |
| 回合剩余时间比例 | 1 |

输入顺序和尺度由 `observations.py` 统一定义。六个动作依次对应右腿 1、右腿 2、左腿 1、左腿 2、右轮和左轮。经过 `tanh` 与力矩上限缩放后，腿部范围为 ±1.8 N·m，轮子为 ±1.5 N·m。

### 训练过程与结果

1. **单步奖励训练实验：** 根据存活奖励和摔倒惩罚更新策略、价值网络，曾出现长期训练退步；目前未验证其能稳定站立。奖励训练的 `gamma` 已恢复为 `1.0`。
2. **LQR 模仿学习：** 由 LQR 控制机器人，记录“当前观测—LQR 力矩”，直接最小化网络动作与目标动作的误差。采集 19500 条训练数据、6000 条验证数据，默认姿态存活时间由 0.410 秒提升至 1.834 秒。
3. **网络轨迹数据聚合：** 网络实际控制机器人，LQR 沿相同轨迹计算目标动作。追加这些状态和标签后再训练，共进行三轮，新增 51310 条训练数据。

| 阶段 | 累计训练样本 | 固定 5 个测试姿态完成 3 秒 |
| --- | ---: | ---: |
| 首次模仿训练 | 19500 | 0/5 |
| 数据补充第 1 轮 | 32295 | 3/5 |
| 数据补充第 2 轮 | 51310 | 5/5 |
| 数据补充第 3 轮 | 70810 | 5/5 |

最终模型重新加载后，在另外 10 个小幅倾斜姿态下也全部完成 3 秒。结果表明：补充网络实际访问的状态，有助于改善独立控制表现；无需先增加隐藏层即可取得本次进展。但有限测试不代表任意状态下都能站稳。

本次模仿和数据聚合训练**不使用奖励、gamma 或价值网络**，训练目标是 LQR 动作。它们与 `observe_robot.py` 中的 Actor–Critic 实验是不同的训练入口。

### 从头复现训练

在项目根目录依次运行：

```powershell
& '.\.venv\Scripts\python.exe' '.\RL\lqr_imitation.py'
& '.\.venv\Scripts\python.exe' '.\RL\lqr_dagger.py'
```

第一次命令采集 LQR 数据并训练 120 遍。第二次命令从首次模仿模型开始，执行三轮采集和再训练，每轮训练 80 遍。训练完成后会保存模型、数据集和验证报告。

- 初次模仿结果位于 `RL/experiments/lqr_imitation_silu/`。已有 `policy.pt` 时脚本会拒绝覆盖；再次从头训练前请先另存旧结果。
- 数据聚合结果位于 `RL/experiments/lqr_dagger_时间戳/`，各轮模型分别保存为 `policy_round_1.pt` 至 `policy_round_3.pt`。
- 评估新模型时，将演示命令中的 `--checkpoint` 改为相应模型路径。
- `observe_robot.py` 不会自动加载这些模仿模型；直接运行它会进入独立的奖励训练实验。

本次结果文件：

- [训练日志与结论](RL/TRAINING_LOG.md)
- [最终策略模型](RL/results/lqr_dagger_policy.pt)
- [三轮训练与固定姿态评估报告](RL/results/lqr_dagger_report.json)
- [额外 10 个姿态的评估报告](RL/results/additional_evaluation.json)

虚拟环境、完整训练数据及历史实验存档保留在本地，不随 Git 提交。报告中的历史机器路径仅用于记录来源，运行随仓库提供的模型不依赖这些路径。

## 修改为自定义同构型机器人

`LQR_VMC/param_cal/robot_param_cal.py` 从 MuJoCo XML 模型中自动提取机器人动力学参数。运行：

```bash
cd LQR_VMC
python param_cal/robot_param_cal.py
```

参数会保存到 `LQR_VMC/param_cal/output/robot_params.npz`，推荐在 VS Code 中安装 **NPZ Viewer** 扩展直接查看：

![NPZ Viewer](images/npzviewer.png)

### 修改模型并更新 LQR 增益

修改机器人模型后，需依次完成以下步骤使控制器适配新参数：

1. **修改 `robot.xml`** — 调整 `LQR_VMC/mjcf/robot.xml` 中各部件的几何尺寸与质量。
2. **提取新参数** — 运行 `python param_cal/robot_param_cal.py`，从 `param_cal/output/robot_params.npz` 获取更新后的 `torso_mass`、`pendulum_mass`、`wheel_mass`、`wheel_radius` 等值。
3. **填入 MATLAB 脚本** — 将新参数写入项目根目录的 `cal_lqr_k.m` 的 `R`、`mw`、`mp`、`M` 等变量。
4. **运行 MATLAB** — 执行 `cal_lqr_k.m`，会对 0.09~0.17 m 腿长范围逐点做 LQR 离散整定，再对 12 维 K 矩阵每个元素做三次多项式拟合，输出格式化的 `POLY_COEFFICIENT`。
5. **更新 Python 控制器** — 将 MATLAB 输出的 `POLY_COEFFICIENT` 替换到 `lqr_update.py` 的同名变量中。
6. **同步连杆长度** — 如修改了 `robot.xml` 中连杆几何，需同步更新 `vmc_cal.py` 中 `five_links_param.__init__()` 的 `l_1`~`l_5`（见下方五连杆参数表）。

### 参数计算方法

以下表格列出 `robot_param_cal.py` 中每个参数的计算逻辑及对应公式。

| 参数 | 符号 | 计算方法 | 公式 |
|------|------|----------|------|
| 机体质量 | `torso_mass` | MuJoCo 将焊死刚体自动累加至父 body，取 `torso` + 4 个限位块 body 质量之和 | M_torso = m_body[torso] + Σ m_body[limit_i] |
| 驱动轮半径 | `wheel_radius` | 读取 `right_wheel_geom` 的 geom 尺寸第一分量（圆柱半径） | R = geom_size[right_wheel_geom][0] |
| 驱动轮质量 | `wheel_mass` | 读取 `right_wheel` body 的动力学质量 | m_wheel = body_mass[right_wheel] |
| 驱动轮转动惯量 | `wheel_inertia` | 按**实心圆柱绕中心轴旋转**模型 | I_zz = ½ · m_wheel · R² |
| 机体转动惯量 | `torso_inertia` | MuJoCo 内部计算的 `torso` body 本体主轴惯量，返回 `[I_xx, I_yy, I_zz]` | I_yy = body_inertia[torso] |
| 摆杆质量 | `pendulum_mass` | 左右腿共 8 段连杆 body 质量求和 | m_p = Σ body_mass[leg_i] (i=1..8) |
| 质点偏移 | `com_offset` | 取 IMU site 相对 torso body 坐标系 Z 轴偏移的绝对值 | δ_z = \|site_pos[imu][2]\| |
| 摆杆转动惯量 | — | 独立函数 `calc_pendulum_inertia(m_p, L₀)`，按**细长杆绕端点旋转**模型，随虚拟腿长 L₀ 实时变化 | I_p = ⅓ · m_p · L₀² |

> **注**：前 7 项在脚本运行后保存至 `robot_params.npz`；第 8 项摆杆转动惯量由 `calc_pendulum_inertia()` 独立函数在 LQR 控制循环中根据当前腿长 L₀ 动态调用。

### 五连杆几何参数

以下参数硬编码在 `vmc_cal.py` 的 `five_links_param.__init__()` 中，修改 `robot.xml` 的连杆长度时需同步更新。

| 参数 | 符号 | 值 (m) | 说明 |
|------|------|--------|------|
| 上连杆 1 | `l_1` | 0.075 | 髋关节 → 膝关节（近端） |
| 下连杆 2 | `l_2` | 0.14 | 膝关节 → 足端（近端） |
| 下连杆 3 | `l_3` | 0.14 | 膝关节 → 足端（远端） |
| 上连杆 4 | `l_4` | 0.075 | 髋关节 → 膝关节（远端） |
| 固定连杆 | `l_5` | 0.08 | 髋关节间距 |

## 其他文件夹简单介绍（可选）
1. 传感器(电机编码器 IMU)可视化

```bash
cd robot_feedback_test
python plot_sensors.py
```

实时绘制 6 个关节电机的位置、速度以及 IMU 欧拉角变化曲线；
查看机体的姿态可以将robot_feedback_test/robot.xml中的freejoint取消注释，固定在天上输入quat进行查看

2. 初期运动学验证

```bash
cd robot_vmc_test
python vmc_verification.py
```

验证五连杆正/逆运动学、电机零点标定和腿长控制范围。

## TODO

- [ ] 验证学习策略的长期站立表现
- [ ] 验证初始姿态变化和外力扰动下的稳定性
- [ ] 在保留平衡能力的前提下研究奖励训练与行走任务
- [ ] 实现打滑检测
- [ ] 进行真机验证
