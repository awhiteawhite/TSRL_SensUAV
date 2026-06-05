import pandas as pd
from pathlib import Path
from datetime import datetime
import os
import argparse


def preprocess_dataset():
    """
    预处理数据集：按天划分数据，并在每一天内按时间顺序排序
    （原始逻辑，仅处理杭州，覆盖写 daily/ 文件夹）
    """
    script_dir = Path(__file__).parent
    input_file = script_dir / 'hangzhou_region0_101_MayToJuly.csv'
    output_dir = script_dir / 'daily'

    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    print(f"正在读取数据集: {input_file}")
    df = pd.read_csv(input_file)

    print(f"原始数据集包含 {len(df)} 条记录")
    print(f"列名: {list(df.columns)}")

    if 'accept_time' not in df.columns:
        raise ValueError("数据集中没有 'accept_time' 列")

    print("正在解析时间字段...")
    df['accept_datetime'] = pd.to_datetime(df['accept_time'], format='%Y/%m/%d %H:%M')
    df['date'] = df['accept_datetime'].dt.date

    output_dir.mkdir(exist_ok=True)
    print(f"输出目录: {output_dir}")

    grouped = df.groupby('date')
    total_files = 0
    total_records = 0

    for date, group in grouped:
        sorted_group = group.sort_values('accept_datetime').reset_index(drop=True)

        filename = f"{date}.csv"
        output_path = output_dir / filename

        sorted_group.drop(['accept_datetime', 'date'], axis=1).to_csv(output_path, index=False)

        print(f"保存 {filename}: {len(sorted_group)} 条记录")
        total_files += 1
        total_records += len(sorted_group)

    print("\n预处理完成!")
    print(f"总共生成 {total_files} 个文件")
    print(f"总记录数: {total_records}")

    original_count = len(df)
    if total_records == original_count:
        print("[OK] 数据完整性验证通过")
    else:
        print(f"[WARNING] 记录数不匹配，原数据 {original_count} 条，处理后 {total_records} 条")


def merge_dataset(input_csv: str):
    """
    将任意城市的订单 CSV 合并写入 datasets/daily/ 目录。
    已有 daily 文件时按 order_id 去重、按 accept_time 排序后写回，
    保证不同城市的数据共存于同一天文件中（靠 region_id 列区分）。

    用法:
        python datasets/preprocess_dataset.py --merge shanghai_region87_May_to_July.csv
    """
    script_dir = Path(__file__).parent
    input_file = script_dir / input_csv
    output_dir = script_dir / 'daily'

    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    print(f"[merge] 正在读取: {input_file}")
    new_df = pd.read_csv(input_file)
    print(f"[merge] 新数据共 {len(new_df)} 条记录, 列: {list(new_df.columns)}")

    if 'accept_time' not in new_df.columns:
        raise ValueError("数据集中没有 'accept_time' 列")

    new_df['accept_datetime'] = pd.to_datetime(new_df['accept_time'], format='%Y/%m/%d %H:%M')
    new_df['date'] = new_df['accept_datetime'].dt.date

    output_dir.mkdir(exist_ok=True)

    grouped = new_df.groupby('date')
    total_new = 0
    total_merged = 0

    for date, group in grouped:
        filename = f"{date}.csv"
        output_path = output_dir / filename

        new_rows = group.drop(['accept_datetime', 'date'], axis=1)

        if output_path.exists():
            existing = pd.read_csv(output_path)
            combined = pd.concat([existing, new_rows], ignore_index=True)
            combined = combined.drop_duplicates(subset=['order_id'], keep='last')
        else:
            combined = new_rows

        combined['_sort_key'] = pd.to_datetime(combined['accept_time'], format='%Y/%m/%d %H:%M')
        combined = combined.sort_values('_sort_key').reset_index(drop=True)
        combined = combined.drop(columns=['_sort_key'])

        combined.to_csv(output_path, index=False)

        n_new = len(new_rows)
        n_total = len(combined)
        print(f"[merge] {filename}: +{n_new} 新行 -> 合计 {n_total} 行")
        total_new += n_new
        total_merged += n_total

    print(f"\n[merge] 完成! 新增 {total_new} 条记录, 涉及 {len(grouped)} 天, 合计 {total_merged} 条")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预处理订单数据集到 daily/ 目录")
    parser.add_argument(
        "--merge",
        type=str,
        default=None,
        help="合并模式：指定要合并的 CSV 文件名（位于 datasets/ 下），"
             "会将数据 append 到已有的 daily/ 文件中而非覆盖。"
             "例: --merge shanghai_region87_May_to_July.csv",
    )
    args = parser.parse_args()

    if args.merge:
        merge_dataset(args.merge)
    else:
        preprocess_dataset()