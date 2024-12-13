import pandas as pd
import os


def find_highly_correlated_groups(corr_matrix, threshold=0.90):
    n = corr_matrix.shape[0]
    visited = [False] * n
    groups = []

    def dfs(i, group):
        visited[i] = True
        group.append(i)
        for j in range(n):
            if not visited[j] and abs(corr_matrix.iloc[i, j]) > threshold:
                dfs(j, group)

    for i in range(n):
        if not visited[i]:
            group = []
            dfs(i, group)
            if len(group) > 1:
                groups.append(group)

    return groups


def process_csv(filename):
    # 读取CSV文件
    print(filename)
    data = pd.read_csv(filename)

    # 计算相关性矩阵
    corr_matrix = data.corr()

    # 找到相关性高于阈值的属性组
    groups = find_highly_correlated_groups(corr_matrix)

    # 创建输出目录
    output_dir = "high_correlation_groups"
    os.makedirs(output_dir, exist_ok=True)

    # 将相关性高的属性组分别写入不同的CSV文件
    for idx, group in enumerate(groups):
        columns = corr_matrix.columns[group]
        data[columns].to_csv(os.path.join(output_dir, f"group_{idx + 1}.csv"), index=False)
        print(f"Saved group {idx + 1} with columns: {', '.join(columns)}")


if __name__ == '__main__':
    # 使用示例
    filename = "0x0301#ADC1.csv"  # 替换为你的CSV文件名
    process_csv(filename)
