import re


def calculate_cache_rate(path):
    new_total = 0
    cached_total = 0
    num_pattern = re.compile(r'"new": (\d+).*?"cached": (\d+)')

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # 使用正则快速提取数值（比完整JSON解析快3-5倍）
            match = num_pattern.search(line)
            if match:
                new_total += int(match.group(1))
                cached_total += int(match.group(2))

    total = new_total + cached_total
    print("total", total)
    return cached_total / total if total else 0.0


print(
    calculate_cache_rate(
        "/u/cjia/sglang-common/sglang/cache_lora_benchmark/trace/acc.txt"
    )
)
