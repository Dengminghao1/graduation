import pandas as pd
import os


def simple_merge_and_clean(directory, output_name="merged_clean.csv"):
    """
    简化版：合并CSV并删除NaN行
    """
    # 获取所有CSV文件
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    print(f"找到 {len(files)} 个CSV文件")

    # 合并所有文件
    dfs = []
    for file in files:
        df = pd.read_csv(os.path.join(directory, file))
        # df['source_file'] = file  # 标记来源文件
        dfs.append(df)

    # 合并
    merged = pd.concat(dfs, ignore_index=True)
    print(f"合并后总行数: {len(merged)}")

    # 查找attention列
    for col in merged.columns:
        if 'attention' in col.lower():
            # 删除NaN行
            before = len(merged)
            cleaned = merged.dropna(subset=[col])
            after = len(cleaned)

            print(f"删除前: {before} 行")
            print(f"删除NaN后: {after} 行")
            print(f"删除 {before - after} 行")

            # 保存
            output_path = os.path.join(directory, output_name)
            cleaned.to_csv(output_path, index=False)
            print(f"已保存到: {output_path}")
            return cleaned

    print("未找到attention列")
    return merged


# 使用示例
if __name__ == "__main__":
    # 修改为你的目录路径
    simple_merge_and_clean(r"E:\数据\20231229 计算机网络考试数据汇总\第1组\视频\2021214387_周婉婷\total\align_face_eeg_second", r"D:\GraduationProject\demo1\output\align_face_eeg_second.csv")