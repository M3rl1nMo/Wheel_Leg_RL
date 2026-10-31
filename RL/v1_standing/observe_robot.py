
# 支持从项目根目录直接运行此文件。
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if p.name == "RL").parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
from pathlib import Path
import sys
CURRENT_FILE =  Path(__file__).resolve()
RL_DIR = _PROJECT_ROOT / "RL" / "v1_standing" / "data"
PROJECT_ROOT = _PROJECT_ROOT
CONTROL_DIR = PROJECT_ROOT / "LQR_VMC"

sys.path.insert(0, str(CONTROL_DIR))

import time
import math
import mujoco
import mujoco.viewer
import numpy as np

#神经网络部分
import torch
from RL.v1_standing.networks import PolicyNetwork, ValueNetwork
from RL.common.checkpoints import save_checkpoint, load_latest_checkpoint
from RL.common.training_metrics import EpisodeMetrics

from robot_mujoco_sensor_feedback.robot_info_update import robot_info_update
from lqr_update import lqr_controller
from vmc_cal import five_links_param

from remote.xbox_controller import PygameController 
from util.climb_traj import VelocityRampFilter 
from roll_leg_control import RollLegController
from ground_detection import detect_leg_ground_status

class PIDController:
    def __init__(self, kp, ki, kd, max_out, max_i, feedforward=0.0):
        self.kp = kp; self.ki = ki; self.kd = kd; self.max_out = max_out; self.max_i = max_i; self.feedforward = feedforward
        self.integral = 0.0; self.prev_error = 0.0
    def compute(self, current, target):
        error = target - current; self.integral += error
        self.integral = np.clip(self.integral, -self.max_i, self.max_i)
        derivative = (error - self.prev_error); self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative + self.feedforward
        return np.clip(output, -self.max_out, self.max_out)


# ==================== 封装的轮腿控制函数 ====================

def control_left_leg(info, five_links, lqr, yaw_pid, L0_pid, pitch, gyro_pitch, gyro_yaw, 
                     target_w, target_length, common_motor_pos, common_motor_speed, wheel_radius,
                     is_airborne_last_step, ground_detection_enabled, leg_tp_comp=0.0):
    """
    左腿 LQR + VMC 核心控制计算与条件离地保护机制
    :param ground_detection_enabled: 是否开启离地检测（即 teleop.control_enabled）
    """
    motor1_angle = info['pos']['left1'] 
    motor2_angle = info['pos']['left2']

    # 1. 运动学与状态更新
    five_links.forward_kinematics_cal(1, motor1_angle, motor2_angle, pitch, gyro_pitch)
    lqr.states_update(1, five_links, pitch, gyro_pitch, common_motor_pos, common_motor_speed, wheel_radius)
    lqr.calc_k_matrix_from_poly(five_links.L_0)

    # 2. VMC 虚拟力基本计算
    F_0_ctrl = L0_pid.compute(five_links.L_0, target_length)
    F_0 = (4 / math.cos(five_links.theta) if math.cos(five_links.theta) != 0 else 0) + F_0_ctrl
    F_0 = np.clip(F_0, -100.0, 100.0)

    # 预计算常规状况下的 Tp
    T_p = (lqr.K[1] * (lqr.theta - 0.0)
           + lqr.K[3] * (lqr.d_theta - 0.0)
           + lqr.K[5] * (lqr.x_b - lqr.x_b_set)
           + lqr.K[7] * (lqr.v_b_whole - lqr.v_set)
           + lqr.K[9] * (lqr.phi - 0.0)
           + lqr.K[11] * (lqr.d_phi - 0.0))

    # 3. 核心条件判定：只有在 Enter 激活后，才执行离地保护状态切换
    if ground_detection_enabled and (detect_leg_ground_status(five_links, F_0, T_p) == 1 or is_airborne_last_step):
        # 离地保护状态：轮子力矩清零，髋关节切换为空中阻尼姿态维持
        left_wheel_T = 0.0
        T_p = lqr.K[1] * (lqr.theta - 0.0) + lqr.K[3] * (lqr.d_theta - 0.0)
        T_p += leg_tp_comp
        flight_flag = 1
    else:
        # 正常控制状态（未开启 Enter 或者是处于着地状态）：正常的自平衡控制
        yaw_ctrl = yaw_pid.compute(gyro_yaw, target_w)
        wheel_T = (lqr.K[0] * (lqr.theta - 0.0)
                   + lqr.K[2] * (lqr.d_theta - 0.0)
                   + lqr.K[4] * (lqr.x_b - lqr.x_b_set)
                   + lqr.K[6] * (lqr.v_b_whole - lqr.v_set)
                   + lqr.K[8] * (lqr.phi - 0.0)
                   + lqr.K[10] * (lqr.d_phi - 0.0))
        
        left_wheel_T = -wheel_T - yaw_ctrl
        left_wheel_T = np.clip(left_wheel_T, -1.5, 1.5)
        flight_flag = 0

    # 4. 最终 VMC 力矩映射
    torque_set_0, torque_set_1 = five_links.VMC_torque_cal(F_0, T_p)
    
    return -torque_set_0, -torque_set_1, left_wheel_T, flight_flag


def control_right_leg(info, five_links, lqr, yaw_pid, L0_pid, pitch, gyro_pitch, gyro_yaw, 
                      target_w, target_length, common_motor_pos, common_motor_speed, wheel_radius,
                      is_airborne_last_step, ground_detection_enabled, leg_tp_comp=0.0):
    """
    右腿 LQR + VMC 核心控制计算与条件离地保护机制
    """
    motor1_angle = info['pos']['right1']
    motor2_angle = info['pos']['right2']
    
    # 1. 运动学与状态更新
    five_links.forward_kinematics_cal(0, motor1_angle, motor2_angle, pitch, gyro_pitch)
    lqr.states_update(0, five_links, pitch, gyro_pitch, -common_motor_pos, -common_motor_speed, wheel_radius)
    lqr.calc_k_matrix_from_poly(five_links.L_0)

    # 2. VMC 虚拟力基本计算
    F_0_ctrl = L0_pid.compute(five_links.L_0, target_length)
    F_0 = (4 / math.cos(five_links.theta) if math.cos(five_links.theta) != 0 else 0) + F_0_ctrl
    F_0 = np.clip(F_0, -100.0, 100.0)

    # 预计算常规状况下的 Tp
    T_p = (lqr.K[1] * (lqr.theta - 0.0)
           + lqr.K[3] * (lqr.d_theta - 0.0)
           + lqr.K[5] * (lqr.x_b - lqr.x_b_set)
           + lqr.K[7] * (lqr.v_b_whole - lqr.v_set)
           + lqr.K[9] * (lqr.phi - 0.0)
           + lqr.K[11] * (lqr.d_phi - 0.0))
    
    # 3. 核心条件判定
    if ground_detection_enabled and (detect_leg_ground_status(five_links, F_0, T_p) == 1 or is_airborne_last_step):
        right_wheel_T = 0.0
        T_p = lqr.K[1] * (lqr.theta - 0.0) + lqr.K[3] * (lqr.d_theta - 0.0)
        T_p += leg_tp_comp
        flight_flag = 1
    else:
        yaw_ctrl = yaw_pid.compute(gyro_yaw, target_w)
        wheel_T = (lqr.K[0] * (lqr.theta - 0.0)
                   + lqr.K[2] * (lqr.d_theta - 0.0)
                   + lqr.K[4] * (lqr.x_b - lqr.x_b_set)
                   + lqr.K[6] * (lqr.v_b_whole - lqr.v_set)
                   + lqr.K[8] * (lqr.phi - 0.0)
                   + lqr.K[10] * (lqr.d_phi - 0.0))
                   
        right_wheel_T = wheel_T - yaw_ctrl
        right_wheel_T = np.clip(right_wheel_T, -1.5, 1.5)
        flight_flag = 0

    torque_set_0, torque_set_1 = five_links.VMC_torque_cal(F_0, T_p)
    
    return torque_set_0, torque_set_1, right_wheel_T, flight_flag


# ============================================================

if __name__ == "__main__":
    dt = 0.002
    wheel_radius = 0.06
    
    target_length_left = 0.1
    target_length_right = 0.1
    roll = 0.0
    current_v = 0.0 
    
    # 状态标志：记录整车上一周期是否触发了空中保护状态
    robot_is_airborne = False
    
    five_links_right = five_links_param(dt=dt)
    five_links_left = five_links_param(dt=dt)
    lqr_right = lqr_controller(dt=dt)
    lqr_left = lqr_controller(dt=dt)
    
    left_yaw_pid = PIDController(0.5, 0.0, 1.0, 2.0, 0.0, feedforward=0.0)
    right_yaw_pid = PIDController(0.5, 0.0, 1.0, 2.0, 0.0, feedforward=0.0)
    left_L0_pid = PIDController(700.0, 0.0, 5.0, 150.0, 0.0, feedforward=0.0)
    right_L0_pid = PIDController(700.0, 0.0, 5.0, 150.0, 0.0, feedforward=0.0)
    
    model_path = CONTROL_DIR / "mjcf" / "robot.xml"
    m = mujoco.MjModel.from_xml_path(str(model_path))
    torso_id = m.body("torso").id
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)  # 初始化传感器数据
    policy = PolicyNetwork()
    value_net = ValueNetwork()

    policy.eval()
    value_net.train()

    #创建价值网络
    value_optimizer = torch.optim.SGD(
        value_net.parameters(),
        lr = 0.001
    )
    #创建策略网络
    policy_optimizer = torch.optim.SGD(
        policy.parameters(),
        lr = 0.0001
    )

    # 只限制日志频率；网络仍然在每个物理步更新。
    log_interval = 0.1  # 单位：仿真秒
    last_print_time = -log_interval
    episode_reward = 0.0
    episode_index = 1
    completed_episodes = 0
    total_updates = 0
    stop_reason = "window_closed"
    max_episodes = 10
    episode_time_limit = 3.0  # 每回合最多运行的仿真秒数
    control_mode = "policy"
    # 每次启动额外训练 max_episodes 回合；累计计数从旧文件恢复。
    resume_training = True  # 改为 False 可显式从头训练，不删除旧文件
    previous_completed_episodes = 0
    initial_total_updates = 0
    resumed_from = None
    if control_mode == "policy" and resume_training:
        expected_config = {
            "algorithm": "one_step_actor_critic",
            "observation_dim": 4,
            "action_dim": 6,
            "observation_order": ["negative_pitch", "negative_gyro_pitch",
                                  "leftwheel_velocity", "rightwheel_velocity"],
            "action_transform": "tanh_then_scale",
            "torque_limits": [1.8, 1.8, 1.8, 1.8, 1.5, 1.5],
            "std": 0.2,
            "gamma": 0.99,
            "timestep": float(m.opt.timestep),
            "fall_angle": 0.8,
            "min_torso_height": 0.12,
            "fall_reward": -5.0,
            "alive_reward": float(m.opt.timestep),
            "episode_time_limit": episode_time_limit,
        }
        restored = load_latest_checkpoint(
            RL_DIR / "checkpoints", policy, value_net,
            policy_optimizer, value_optimizer, expected_config
        )
        if restored is None:
            print("[初始化] 没有已保存的模型，本次从头训练。")
        else:
            checkpoint_path, saved_info = restored
            resumed_from = str(checkpoint_path)
            previous_completed_episodes = saved_info["completed_episodes"]
            total_updates = saved_info["total_updates"]
            initial_total_updates = total_updates
            print(f"[恢复] 已加载：{checkpoint_path}")
            print(
                f"[恢复] 累计完成回合={previous_completed_episodes}"
                f" | 累计更新={total_updates} | 本次再训练={max_episodes}回合"
            )
            print("[恢复] 机器人从初始状态开始；不恢复上次中途的物理状态。")
    elif control_mode == "policy":
        print("[初始化] 已关闭恢复，本次从头训练。")
    # 旧存档没有历史记录时，从本次第一个完整回合开始统计。
    if control_mode == "policy":
        history = saved_info.get("recent_episode_durations", []) if resumed_from else []
        metrics = EpisodeMetrics(RL_DIR / "training_logs", history, resumed_from)
        print(f"[记录] 每回合数据保存至：{metrics.path}")

    print(f"[启动] 模式={control_mode} | 日志间隔={log_interval:.2f}仿真秒")
    print("每条记录对应一个物理步；箭头表示同一观测/动作在参数更新前后的数值。")
   

    teleop = PygameController()
    v_filter = VelocityRampFilter(dt=dt, accel_rate=0.8, decel_rate=1.4)
    roll_balancer = RollLegController(kp=1.0, ki=0.0, kd=0.0, max_out=0.14, min_leg=0.10, max_leg=0.14)
    
    try:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running():
                step_start = time.time()
            
                teleop.update(
                    current_v=current_v, 
                    target_length_left=target_length_left, 
                    target_length_right=target_length_right, 
                    roll=roll, 
                    viewer=viewer
                )
            
                if teleop.paused:
                    viewer.sync()
                    time_until_next_step = dt - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
                    continue

                # 获取统一传感器信息
                info = robot_info_update(d)

                            # 每个仿真循环都读取观测并计算动作
                obs = torch.tensor(
                    [[
                        -info["euler"]["pitch"],
                        -info["gyro_pitch"],
                        info["vel"]["leftwheel"],
                        info["vel"]["rightwheel"],
                    ]],
                    dtype=torch.float32
                )

                with torch.no_grad():
                    mean = policy(obs)

                    std = torch.full_like(mean, 0.2)
                    distribution = torch.distributions.Normal(mean, std)
                    raw_action = distribution.sample()

                    bounded_action = torch.tanh(raw_action)

                    torque_limits = torch.tensor(
                        [1.8, 1.8, 1.8, 1.8, 1.5, 1.5],
                        dtype=torch.float32
                    )

                    motor_torques = bounded_action * torque_limits

                # 去掉批次维度，得到六个执行器对应的一维数组
                torque_array = motor_torques.detach().cpu().numpy()[0]

                pitch = -info['euler']['pitch']
                yaw = info['euler']['yaw']
                roll = -info['euler']['roll']

                # 摔倒中断
                # fall_angle = 0.8
                # fallen = (
                #     abs(pitch) > fall_angle
                #     or abs(roll) > fall_angle
                # )
                # #奖励系统
                # if control_mode == "policy":
                #     # 初始状态还没有执行动作，不计算动作结果的奖励
                #     if d.time > 0.0:
                #         if fallen:
                #             reward = -5.0
                #         else:
                #             reward = float(m.opt.timestep)
                #         episode_reward += reward

                #     if fallen:
                #         print("检测到摔倒")
                #         print("俯仰角：", pitch)
                #         print("横滚角：", roll)
                #         print("回合累计奖励：", episode_reward)
                #         break

            
                gyro_pitch = -info['gyro_pitch']
                gyro_yaw = info['gyro_yaw']  
                gyro_roll = -info['gyro_roll']
            
                # 使用封装后的 Roll 平衡控制器
                target_length_left, target_length_right = roll_balancer.compute_leg_lengths(
                    current_roll=roll,
                    target_roll=0.0,
                    base_length=teleop.target_length
                )

                # --- 共模提取 ---
                motor_speed_left = info['vel']['leftwheel']
                motor_speed_right = info['vel']['rightwheel']
                motor_pos_left = info['pos']['leftwheel']
                motor_pos_right = info['pos']['rightwheel']
            
                common_motor_speed = (motor_speed_left - motor_speed_right) / 2.0
                common_motor_pos = (motor_pos_left - motor_pos_right) / 2.0

                # ---------------- 离地保护中的：位移、速度期望清零 ----------------
                # 只有在 Enter 开关激活 (teleop.control_enabled 为 True) 并且检测到空中状态时才重置
                if teleop.control_enabled and robot_is_airborne:
                    lqr_left.x_b_set = (common_motor_pos * wheel_radius)
                    lqr_right.x_b_set = (common_motor_pos * wheel_radius)
                    current_v = 0.0
                    teleop.target_w = 0.0  
                else:
                    # 正常平衡状态（无论是否按 Enter 键，都可以通过爬坡滤波器正常运动平衡）
                    current_v = v_filter.update(current_v=current_v, target_v=teleop.target_v)
                    if not (current_v == 0.0 and teleop.target_v == 0):
                        lqr_left.x_b_set = (common_motor_pos * wheel_radius)
                        lqr_right.x_b_set = (common_motor_pos * wheel_radius) 
            
                lqr_left.v_set = current_v
                lqr_right.v_set = current_v

                # ---------------- 调用左腿控制函数（传入检测使能锁） ----------------
                ctrl_left_0, ctrl_left_1, left_wheel_T, left_flight = control_left_leg(
                    info, five_links_left, lqr_left, left_yaw_pid, left_L0_pid,
                    pitch, gyro_pitch, gyro_yaw, teleop.target_w, target_length_left,
                    common_motor_pos, common_motor_speed, wheel_radius,
                    is_airborne_last_step=robot_is_airborne,
                    ground_detection_enabled=teleop.control_enabled, # 按 Enter 开启保护
                    leg_tp_comp=0.0
                )

                # ---------------- 调用右腿控制函数（传入检测使能锁） ----------------
                ctrl_right_0, ctrl_right_1, right_wheel_T, right_flight = control_right_leg(
                    info, five_links_right, lqr_right, right_yaw_pid, right_L0_pid,
                    pitch, gyro_pitch, gyro_yaw, teleop.target_w, target_length_right,
                    common_motor_pos, common_motor_speed, wheel_radius,
                    is_airborne_last_step=robot_is_airborne,
                    ground_detection_enabled=teleop.control_enabled, # 按 Enter 开启保护
                    leg_tp_comp=0.0
                )

                # ---------------- 整体离地状态判断汇总 ----------------
                if teleop.control_enabled:
                    robot_is_airborne = (left_flight == 1) or (right_flight == 1)
                else:
                    robot_is_airborne = False

                # ---------------- 整体执行执行器下发（任何时候都保持自平衡控制输出） ----------------
                # 不受 teleop.control_enabled 的直接拦截，任何时候力矩均可以下发
                if control_mode == "policy":
                    d.ctrl[:] = torque_array

                elif control_mode == "lqr":
                    d.ctrl[0] = ctrl_right_0
                    d.ctrl[1] = ctrl_right_1
                    d.ctrl[2] = ctrl_left_0
                    d.ctrl[3] = ctrl_left_1
                    d.ctrl[4] = right_wheel_T
                    d.ctrl[5] = left_wheel_T

                else:
                    raise ValueError("control_mode 必须是 lqr 或 policy")     

                mujoco.mj_step(m, d)
                viewer.sync()

                mujoco.mj_forward(m, d)
                next_info = robot_info_update(d)
                next_pitch = -next_info["euler"]["pitch"]
                next_roll = -next_info["euler"]["roll"]
                torso_height = float(d.xpos[torso_id, 2])
                min_torso_height = 0.12

                next_obs = torch.tensor(
                    [[
                        -next_info["euler"]["pitch"],
                        -next_info["gyro_pitch"],
                        next_info["vel"]["leftwheel"],
                        next_info["vel"]["rightwheel"],
                    ]],
                    dtype=torch.float32
                )

                fallen =(
                    abs(next_pitch) > 0.8 
                    or abs(next_roll) > 0.8
                    or torso_height < min_torso_height
                )

                if control_mode == "policy":
                    if fallen:
                        reward = -5.0
                    else:
                        reward = float(m.opt.timestep)

                    episode_reward += reward
                    gamma = 0.99

                    with torch.no_grad():
                        if fallen:
                            target = torch.tensor(
                                [[reward]],
                                dtype=torch.float32
                            )
                        else:
                            next_value = value_net(next_obs)
                            target = reward + gamma * next_value
                    #计算梯度反向传播
                    value = value_net(obs)
                    advantage = (target - value).detach()
                    value_loss = ((value - target) ** 2).mean()

                    value_optimizer.zero_grad()
                    value_loss.backward()
                    value_optimizer.step()


                    #计算对数密度
                    mean_for_update = policy(obs)
                    std_for_update = torch.full_like(mean_for_update, 0.2)
                    distribution_for_update = torch.distributions.Normal(mean_for_update, 
                                                                         std_for_update)
                    #计算log_prob
                    log_prob_each = distribution_for_update.log_prob(raw_action.detach())
                    log_prob = log_prob_each.sum(dim=-1)
                    advantage_for_policy = advantage.squeeze(-1)

                    policy_loss = -(
                        log_prob * advantage_for_policy
                    ).mean()

                    policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_optimizer.step()
                    total_updates += 1

                    # 同一步的指标集中输出；摔倒/超时最后一步必定打印。
                    if fallen or d.time >= episode_time_limit or d.time - last_print_time >= log_interval:
                        last_print_time = d.time

                        # 固定同一个旧观测和已执行样本，检查更新前后变化。
                        # 这些前向计算只用于日志，不产生新的动作，也不更新参数。
                        with torch.no_grad():
                            updated_value = value_net(obs)
                            updated_mean = policy(obs)
                            updated_distribution = torch.distributions.Normal(
                                updated_mean, std_for_update
                            )
                            updated_log_prob = updated_distribution.log_prob(
                                raw_action.detach()
                            ).sum(dim=-1)

                        # f 字符串把花括号中的数值放入文本；.4f 保留四位小数。
                        print(
                            f"[训练 {episode_index}/{max_episodes}] 时间={d.time:.3f}s | 本步奖励={reward:+.4f}"
                            f" | 累计奖励={episode_reward:+.4f}"
                            f" | 优势={advantage.item():+.4f}"
                        )
                        print(
                            f"  价值预测：{value.item():+.4f} -> {updated_value.item():+.4f}"
                            f" | 目标={target.item():+.4f} | 价值损失={value_loss.item():.6f}"
                        )
                        print(
                            f"  原始动作联合log_prob：{log_prob.item():+.4f}"
                            f" -> {updated_log_prob.item():+.4f}"
                            f" | 策略损失={policy_loss.item():+.6f}"
                        )
                        print(
                            f"  执行后姿态：俯仰={next_pitch:+.4f}rad"
                            f" | 横滚={next_roll:+.4f}rad | 机身高度={torso_height:.4f}m"
                        )

                    # 最后一步先完成评分和网络更新，再结束/重置回合。
                    if fallen or d.time >= episode_time_limit:
                        reasons = []
                        if fallen:
                            if abs(next_pitch) > 0.8:
                                reasons.append("俯仰角超限")
                            if abs(next_roll) > 0.8:
                                reasons.append("横滚角超限")
                            if torso_height < min_torso_height:
                                reasons.append("机身高度过低")
                            reason_text = "摔倒：" + "、".join(reasons)
                        else:
                            reason_text = "达到测试时间上限"

                        print(
                            f"\n[回合结束] 第 {episode_index}/{max_episodes} 回合"
                            f" | {reason_text} | 时长={d.time:.3f}s"
                            f" | 累计奖励={episode_reward:+.4f}"
                        )

                        completed_episodes += 1
                        average_duration, sample_count = metrics.record(
                            previous_completed_episodes + completed_episodes,
                            episode_index, float(d.time), float(episode_reward),
                            reason_text, total_updates,
                        )
                        print(
                            f"[统计] 最近 {sample_count} 回合平均存活时间="
                            f"{average_duration:.3f}s（最多统计10回合）"
                        )
                        if episode_index >= max_episodes:
                            stop_reason = "episode_limit"
                            print("本次训练结束，正在准备保存网络。")
                            break

                        episode_index += 1
                        mujoco.mj_resetData(m, d)
                        mujoco.mj_forward(m, d)

                        episode_reward = 0.0
                        last_print_time = -log_interval
                        current_v = 0.0
                        roll = 0.0
                        robot_is_airborne = False
                        target_length_left = 0.1
                        target_length_right = 0.1

                        # 原控制器仍参与计算，因此清除它们的回合历史状态。
                        # 不重新创建 policy、value_net 或两个优化器。
                        five_links_left = five_links_param(dt=dt)
                        five_links_right = five_links_param(dt=dt)
                        lqr_left = lqr_controller(dt=dt)
                        lqr_right = lqr_controller(dt=dt)
                        for controller in (
                            left_yaw_pid, right_yaw_pid,
                            left_L0_pid, right_L0_pid, roll_balancer,
                        ):
                            controller.integral = 0.0
                            controller.prev_error = 0.0

                        print(f"[回合开始] 第 {episode_index}/{max_episodes} 回合")
                        viewer.sync()
                        continue

                elif d.time - last_print_time >= log_interval:
                    last_print_time = d.time
                    print(
                        f"[LQR] 时间={d.time:.3f}s | 俯仰={next_pitch:+.4f}rad"
                        f" | 横滚={next_roll:+.4f}rad | 机身高度={torso_height:.4f}m"
                    )

                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print("\n[停止] 收到 Ctrl+C，准备保存当前网络。")
    except Exception:
        stop_reason = "error"
        raise
    finally:
        # LQR 模式或未完成任何训练更新时，不保存随机初始化的网络。
        if control_mode == "policy" and total_updates > initial_total_updates:
            training_info = {
                "algorithm": "one_step_actor_critic",
                "observation_dim": 4,
                "action_dim": 6,
                "observation_order": ["negative_pitch", "negative_gyro_pitch",
                                      "leftwheel_velocity", "rightwheel_velocity"],
                "action_transform": "tanh_then_scale",
                "torque_limits": [1.8, 1.8, 1.8, 1.8, 1.5, 1.5],
                "std": 0.2,
                "gamma": 0.99,
                "timestep": float(m.opt.timestep),
                "fall_angle": 0.8,
                "min_torso_height": 0.12,
                "fall_reward": -5.0,
                "alive_reward": float(m.opt.timestep),
                "episode_time_limit": episode_time_limit,
                "max_episodes": max_episodes,
                "completed_episodes": previous_completed_episodes + completed_episodes,
                "session_completed_episodes": completed_episodes,
                "session_updates": total_updates - initial_total_updates,
                "resumed_from": resumed_from,
                "recent_episode_durations": list(metrics.durations),
                "metrics_file": str(metrics.path),
                "episode_index": episode_index,
                "total_updates": total_updates,
                "episode_reward": float(episode_reward),
                "simulation_time": float(d.time),
                "stop_reason": stop_reason,
                "torch_version": str(torch.__version__),
                "mujoco_version": str(mujoco.__version__),
            }
            saved_path = save_checkpoint(
                RL_DIR / "checkpoints", policy, value_net,
                policy_optimizer, value_optimizer, training_info
            )
            print(f"[保存] 已写入：{saved_path}")
            print(
                f"[保存] 本次完成回合={completed_episodes}"
                f"，累计完成回合={previous_completed_episodes + completed_episodes}"
                f"，累计更新次数={total_updates}"
            )
            print("[提示] 下次启动默认加载最新保存文件，继续训练。")
