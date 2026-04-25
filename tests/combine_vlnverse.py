import gzip
import json
import os

vlnverse_original_dir = 'data/vlnverse/raw_data/vlnverse/final_splits_with_distance_formal'
new_dir = 'data/vlnverse/raw_data/vlnverse/final_splits_with_distance_formal/total'

splits = ['train', 'val_seen', 'val_unseen', 'test']
types = ['coarse', 'fine']

# 映射 split 名称到文件名
split_to_filename = {
    'train': 'train',
    'val_seen': 'val_seen',  # val_seen 对应 val.json.gz
    'val_unseen': 'val_unseen',
    'test': 'test',
}


def load_json_gz(filepath):
    """加载 gzip 压缩的 JSON 文件"""
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        return json.load(f)


def save_json_gz(data, filepath):
    """保存数据到 gzip 压缩的 JSON 文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def combine_episodes(coarse_data, fine_data):
    """合并 coarse 和 fine 的 episodes"""
    combined_episodes = []

    # 添加 coarse episodes
    if 'episodes' in coarse_data:
        combined_episodes.extend(coarse_data['episodes'])

    # 添加 fine episodes
    if 'episodes' in fine_data:
        combined_episodes.extend(fine_data['episodes'])

    return combined_episodes


def process_split(split):
    """处理单个 split，合并 coarse 和 fine 数据"""
    filename = split_to_filename[split]

    coarse_file = os.path.join(vlnverse_original_dir,'coarse', f'{split}', f'{filename}.json.gz')
    fine_file = os.path.join(vlnverse_original_dir,'fine', f'{split}', f'{filename}.json.gz')

    # 检查文件是否存在
    if not os.path.exists(coarse_file):
        print(f"警告: {coarse_file} 不存在，跳过")
        return

    if not os.path.exists(fine_file):
        print(f"警告: {fine_file} 不存在，跳过")
        return

    print(f"处理 {split}...")

    # 加载数据
    print(f"  加载 {coarse_file}...")
    coarse_data = load_json_gz(coarse_file)
    print(f"  加载 {fine_file}...")
    fine_data = load_json_gz(fine_file)

    # 合并 episodes
    combined_episodes = combine_episodes(coarse_data, fine_data)

    # 创建输出数据结构
    output_data = {'episodes': combined_episodes}

    # 如果有 instruction_vocab，需要合并（如果两个文件都有的话）
    if 'instruction_vocab' in coarse_data or 'instruction_vocab' in fine_data:
        vocab_set = set()
        if 'instruction_vocab' in coarse_data:
            vocab_set.update(coarse_data['instruction_vocab'])
        if 'instruction_vocab' in fine_data:
            vocab_set.update(fine_data['instruction_vocab'])
        output_data['instruction_vocab'] = sorted(list(vocab_set))

    # 保存到新目录
    output_dir = os.path.join(new_dir, split)
    output_file = os.path.join(output_dir, f'{split}.json.gz')

    print(f"  保存到 {output_file}...")
    save_json_gz(output_data, output_file)

    print(
        f"  完成: 合并了 {len(coarse_data.get('episodes', []))} 个 coarse episodes 和 {len(fine_data.get('episodes', []))} 个 fine episodes，共 {len(combined_episodes)} 个 episodes"
    )


def main():
    """主函数"""
    print(f"开始合并数据...")
    print(f"源目录: {vlnverse_original_dir}")
    print(f"目标目录: {new_dir}")
    print()

    for split in splits:
        try:
            process_split(split)
            print()
        except Exception as e:
            print(f"处理 {split} 时出错: {e}")
            print()

    print("所有 splits 处理完成！")


if __name__ == '__main__':
    main()
