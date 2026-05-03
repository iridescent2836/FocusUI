import re
import pandas as pd
import argparse

def parse_log_file(file_path):
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        log_text = f.read()

    # 按照换行符分割日志
    lines = log_text.strip().split('\n')

    # 定义正则表达式匹配各个字段
    log_pattern = re.compile(
        r'\[(.*?)\](\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO - FocusUI - (.*): (.*)'
    )

    parsed_data = []
    current_record = {}

    seen = []

    for line in lines:
        match = log_pattern.search(line)
        if match:
            timestamp = match.group(2)
            metric_name = match.group(3).strip()
            value_str = match.group(4).strip()
            # 提取数值并转换为浮点数（去掉单位 s 或 seconds）
            value_str = value_str.replace('seconds', '').replace('second', '').replace('s', '').replace('econd', '').strip()
            try:
                value = float(value_str)
            except ValueError:
                num_match = re.search(r'\d+\.?\d*', value_str)
                if num_match:
                    value = float(num_match.group(0))
                else:
                    continue
            if metric_name not in seen:
                if len(seen) == 0:
                    current_record['timestamp'] = timestamp
                seen.append(metric_name)
                current_record[metric_name] = value
            else:
                parsed_data.append(current_record.copy())
                current_record = {}
                seen = []

    # 补充最后一个未添加的完整组
    if current_record and all(k in current_record for k in ['image_preprocess_time', 'visual_encoding_time', 'generation_time']):
        parsed_data.append(current_record)

    # 转化为 DataFrame，并计算总时间（三个阶段相加）
    df = pd.DataFrame(parsed_data)

    return df

if __name__ == "__main__":
    # 使用 argparse 添加参数解析，方便指定文件路径
    parser = argparse.ArgumentParser(description="Parse FocusUI log files.")
    parser.add_argument("file", type=str, nargs='?', default="focusui-2026-05-02.log", help="Path to the log file (default: app.log)")
    args = parser.parse_args()

    try:
        df_result = parse_log_file(args.file)
        print(f"--- 成功读取并解析文件：{args.file} ---")
        print(df_result.to_string(index=True))

        # 保存为 CSV
        output_csv = "parsed_log.csv"
        df_result.to_csv(output_csv, index=False)
        print(f"\n解析结果已成功保存到 {output_csv}")
    except FileNotFoundError:
        print(f"错误：未找到文件 {args.file}，请检查路径是否正确。")