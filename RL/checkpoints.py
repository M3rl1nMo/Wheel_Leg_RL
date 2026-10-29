"""保存及恢复网络、优化器和训练计数。"""
from pathlib import Path
import time
import torch


def save_checkpoint(directory, policy, value_net, policy_optimizer,
                    value_optimizer, training_info):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # 每次保存使用不同名称，保留先前实验，不覆盖旧模型。
    filename = f"checkpoint_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}.pt"
    path = directory / filename
    temporary_path = path.with_suffix(".tmp")
    checkpoint = {
        "format_version": 1,
        "policy_state_dict": policy.state_dict(),
        "value_state_dict": value_net.state_dict(),
        "policy_optimizer_state_dict": policy_optimizer.state_dict(),
        "value_optimizer_state_dict": value_optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "training_info": training_info,
    }
    # 完整写入后再改为 .pt，避免把未写完的文件当成有效模型。
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)
    return path


def load_latest_checkpoint(directory, policy, value_net, policy_optimizer,
                           value_optimizer, expected_config):
    files = list(Path(directory).glob("checkpoint_*.pt"))
    if not files:
        return None
    # 只考虑完整的 .pt 文件；不读取保存过程中遗留的 .tmp。
    path = max(files, key=lambda item: (item.stat().st_mtime_ns, item.name))
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError(f"不支持的模型文件格式：{path}")
    info = checkpoint["training_info"]
    for key, expected in expected_config.items():
        if info.get(key) != expected:
            raise ValueError(
                f"模型配置不匹配：{key}，保存值={info.get(key)!r}，"
                f"当前值={expected!r}。文件：{path}"
            )
    for key in ("completed_episodes", "total_updates"):
        if type(info.get(key)) is not int or info[key] < 0:
            raise ValueError(f"无效的训练计数：{key}，文件：{path}")
    for key in ("policy_state_dict", "value_state_dict"):
        for name, tensor in checkpoint[key].items():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"模型参数包含非有限数值：{key}/{name}，文件：{path}")

    policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    value_net.load_state_dict(checkpoint["value_state_dict"], strict=True)
    policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
    value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    return path, info
