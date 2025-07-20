import pandas as pd

# 读取txt文件中的内容
with open('UCF51Categories_34.txt', 'r') as f:
    categories_to_keep = set(line.strip() for line in f.readlines())

# 读取csv文件
df = pd.read_csv('ucf51checked.csv')

# 筛选csv文件的category列，如果category不在txt文件中，则删除该行
filtered_df = df[df['category'].isin(categories_to_keep)]

# 保存筛选后的结果到新的csv文件
filtered_df.to_csv('ucf51checked_34.csv', index=False)

print("筛选后的CSV文件已生成。")
