import os
import re
from collections import defaultdict

log_dir = "/Users/telle/Desktop/ogc2025/logs"
summary = {}

for i in range(1, 11):
    fname = f"prob{i}_log.txt"
    path = os.path.join(log_dir, fname)

    demand_counts = defaultdict(int)
    port_counts = defaultdict(int)

    with open(path) as f:
        for line in f:
            m = re.search(r"Unloading path for demand (\d+) at node .*?Rehandling", line)
            if m:
                d_id = int(m.group(1))
                demand_counts[d_id] += 1

            m2 = re.search(r"Port (\d+)", line)
            if m2:
                cur_port = int(m2.group(1))

        # 총 rehandling 횟수
        total = sum(demand_counts.values())
        summary[f"prob{i}"] = {
            "total_rehandling": total,
            "demand_ids": list(demand_counts.keys()),
            "count_per_demand": dict(demand_counts),
        }

# 결과 요약
for prob, info in summary.items():
    print(f"[{prob}]")
    print(f"  - 재적재 횟수: {info['total_rehandling']}")
    print(f"  - 재적재된 demand ID: {info['demand_ids']}")
    print(f"  - demand별 횟수: {info['count_per_demand']}")
    print()