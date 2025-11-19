# preprocess.py
import pandas as pd
import os
import glob


def preprocess_dataset(dataset_name):
    base_path = f'dataset/{dataset_name}/'

    print(f"处理数据集: {dataset_name}")
    print(f"目录内容: {os.listdir(base_path)}")

    # 针对不同数据集的文件命名约定
    file_mapping = {
        'pheme': {
            'train': f'{dataset_name}_train.csv',
            'test': f'{dataset_name}_test.csv'
        },
        'twitter': {
            'train': 'train_posts_clean.csv',  # Twitter数据集的特殊命名
            'test': 'test_posts.csv'
        },
        'weibo': {
            'train': f'{dataset_name}_train.csv',
            'test': f'{dataset_name}_test.csv'
        }
    }

    # 获取对应的文件名
    train_file = file_mapping.get(dataset_name, {}).get('train', f'{dataset_name}_train.csv')
    test_file = file_mapping.get(dataset_name, {}).get('test', f'{dataset_name}_test.csv')

    train_csv = base_path + train_file
    test_csv = base_path + test_file
    gcn_train_csv = base_path + 'dataforGCN_train.csv'
    gcn_test_csv = base_path + 'dataforGCN_test.csv'

    # 检查文件是否存在
    if not os.path.exists(train_csv):
        print(f"错误: 未找到训练文件 {train_csv}")
        # 尝试查找其他可能的训练文件
        csv_files = [f for f in os.listdir(base_path) if f.endswith('.csv')]
        train_candidates = [f for f in csv_files if 'train' in f.lower() and 'clean' in f.lower()]
        if train_candidates:
            train_csv = base_path + train_candidates[0]
            print(f"尝试使用文件: {train_csv}")
        else:
            return False

    if not os.path.exists(test_csv):
        print(f"错误: 未找到测试文件 {test_csv}")
        # 尝试查找其他可能的测试文件
        csv_files = [f for f in os.listdir(base_path) if f.endswith('.csv')]
        test_candidates = [f for f in csv_files if 'test' in f.lower()]
        if test_candidates:
            test_csv = base_path + test_candidates[0]
            print(f"尝试使用文件: {test_csv}")
        else:
            # 如果没有测试文件，从训练数据中分割
            print("未找到测试文件，将从训练数据中分割...")
            test_csv = None

    try:
        # 读取训练数据
        train_data = pd.read_csv(train_csv)
        print(f"训练数据形状: {train_data.shape}")
        print(f"训练数据列名: {list(train_data.columns)}")

        # 读取或创建测试数据
        if test_csv and os.path.exists(test_csv):
            test_data = pd.read_csv(test_csv)
            print(f"测试数据形状: {test_data.shape}")
            print(f"测试数据列名: {list(test_data.columns)}")
        else:
            # 从训练数据中分割测试集
            split_ratio = 0.2  # 20%作为测试集
            split_index = int(len(train_data) * (1 - split_ratio))
            test_data = train_data[split_index:]
            train_data = train_data[:split_index]
            print(f"从训练数据中分割: 训练集 {len(train_data)} 条, 测试集 {len(test_data)} 条")

        # 数据清洗和列名标准化
        # 删除可能的索引列
        if 'Unnamed: 0' in train_data.columns:
            train_data = train_data.drop(columns=['Unnamed: 0'])
            test_data = test_data.drop(columns=['Unnamed: 0']) if 'Unnamed: 0' in test_data.columns else test_data

        # 确保有label列
        if 'label' not in train_data.columns:
            # 尝试查找标签列
            label_candidates = [col for col in train_data.columns if 'label' in col.lower() or 'class' in col.lower()]
            if label_candidates:
                train_data = train_data.rename(columns={label_candidates[0]: 'label'})
                test_data = test_data.rename(columns={label_candidates[0]: 'label'}) if test_csv else test_data
            else:
                print("警告: 未找到标签列，使用默认值")
                train_data['label'] = 0
                if test_csv:
                    test_data['label'] = 0

        # 保存为GCN所需的格式
        train_data.to_csv(gcn_train_csv, index=False)
        test_data.to_csv(gcn_test_csv, index=False)

        print(f"成功生成: {gcn_train_csv}")
        print(f"成功生成: {gcn_test_csv}")
        return True

    except Exception as e:
        print(f"处理数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dataset_structure(dataset_name):
    """检查数据集结构并提供详细信息"""
    base_path = f'dataset/{dataset_name}/'

    print(f"\n{'=' * 50}")
    print(f"检查 {dataset_name} 数据集")
    print(f"{'=' * 50}")
    print(f"目录: {base_path}")

    if not os.path.exists(base_path):
        print("错误: 数据集目录不存在")
        return

    files = os.listdir(base_path)
    print(f"文件列表: {files}")

    # 特殊处理Twitter数据集
    if dataset_name == 'twitter':
        check_twitter_dataset_structure()
        return

    # 检查关键文件
    key_files = {
        '训练数据': f'{dataset_name}_train.csv',
        '测试数据': f'{dataset_name}_test.csv',
        '图像目录': f'{dataset_name}_image/',
        '伪标签': f'{dataset_name}_analysis_results.csv'
    }

    for desc, filename in key_files.items():
        full_path = base_path + filename
        exists = os.path.exists(full_path)
        status = "存在" if exists else "缺失"
        print(f"{desc}: {filename} - {status}")

    # 如果有训练文件，显示其结构
    train_file = base_path + f'{dataset_name}_train.csv'
    if os.path.exists(train_file):
        try:
            sample_data = pd.read_csv(train_file, nrows=5)
            print(f"\n训练数据样例:")
            print(sample_data.head())
            print(f"列名: {list(sample_data.columns)}")
        except Exception as e:
            print(f"读取训练文件出错: {e}")


def check_twitter_dataset_structure():
    """专门检查Twitter数据集结构"""
    base_path = 'dataset/twitter/'
    print(f"\n=== 详细检查Twitter数据集 ===")

    if not os.path.exists(base_path):
        print("Twitter数据集目录不存在")
        return

    files = os.listdir(base_path)
    print(f"所有文件: {files}")

    # 检查每个CSV文件的结构
    csv_files = [f for f in files if f.endswith('.csv')]
    for csv_file in csv_files:
        try:
            file_path = base_path + csv_file
            data = pd.read_csv(file_path, nrows=5)
            print(f"\n文件: {csv_file}")
            print(f"形状: {data.shape}")
            print(f"列名: {list(data.columns)}")
            print(f"前5行:\n{data.head()}")
        except Exception as e:
            print(f"读取文件 {csv_file} 出错: {e}")

    # 检查目录结构
    dirs = [f for f in files if os.path.isdir(base_path + f)]
    print(f"\n子目录: {dirs}")
    for dir_name in dirs:
        dir_path = base_path + dir_name
        dir_files = os.listdir(dir_path)[:5]  # 只显示前5个文件
        print(f"目录 {dir_name} 中的文件(前5个): {dir_files}")

if __name__ == "__main__":
    # 检查所有数据集结构
    datasets = ['pheme', 'twitter', 'weibo']
    for dataset in datasets:
        check_dataset_structure(dataset)

    # 执行预处理
    for dataset in datasets:
        success = preprocess_dataset(dataset)
        if success:
            print(f"{dataset} 数据集预处理成功!")
        else:
            print(f"{dataset} 数据集预处理失败!")