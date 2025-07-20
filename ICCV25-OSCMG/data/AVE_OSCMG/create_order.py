import h5py

# 读取原始 h5 文件
input_file = 'data/AVE-ECCV18-master/data/train_order.h5'
output_file_14 = 'train_order_14class.h5'
output_file_21 = 'train_order_21class.h5'

# 打开输入文件
with h5py.File(input_file, 'r') as f:
    # 假设数据存储在 'order' 数据集中
    sample_order = f['order'][:]
    
    # 前 14 类数据 (假设 1857 条数据对应 14 类)
    data_14 = sample_order[:1857]
    
    # 前 21 类数据 (假设 2628 条数据对应 21 类)
    data_21 = sample_order[:2628]

    # 保存前 14 类数据到新的 h5 文件
    with h5py.File(output_file_14, 'w') as f_out_14:
        f_out_14.create_dataset('order', data=data_14)  # 使用 'order' 作为数据集名称
    
    # 保存前 21 类数据到新的 h5 文件
    with h5py.File(output_file_21, 'w') as f_out_21:
        f_out_21.create_dataset('order', data=data_21)  # 使用 'order' 作为数据集名称

print(f"前 14 类数据保存到 {output_file_14}")
print(f"前 21 类数据保存到 {output_file_21}")
