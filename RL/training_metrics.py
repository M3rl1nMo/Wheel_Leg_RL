"""每个完整回合写入 CSV，并统计最近最多 10 回合的平均存活时间。"""
from collections import deque
import csv
from pathlib import Path
import time


class EpisodeMetrics:
    def __init__(self, directory, history=(), resumed_from=None):
        # deque 的 maxlen 限制最多保留 10 条，加入新数据时自动移除最早一条。
        self.durations = deque(history, maxlen=10)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"episodes_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}.csv"
        self.resumed_from = resumed_from or ""
        # utf-8-sig 让 Excel 能正确识别中文；newline 避免 Windows 出现多余空行。
        with self.path.open("x", encoding="utf-8-sig", newline="") as file:
            csv.writer(file).writerow([
                "累计回合", "本次回合", "存活时间_秒", "累计奖励", "结束原因",
                "累计更新次数", "最近回合数", "最近平均存活时间_秒", "恢复来源",
            ])

    def record(self, episode, session_episode, duration, reward, reason, updates):
        self.durations.append(duration)
        count = len(self.durations)
        average = sum(self.durations) / count
        # 每回合追加一行并关闭文件，训练中途停止也保留已经写入的回合。
        with self.path.open("a", encoding="utf-8-sig", newline="") as file:
            csv.writer(file).writerow([
                episode, session_episode, duration, reward, reason, updates,
                count, average, self.resumed_from,
            ])
        return average, count
