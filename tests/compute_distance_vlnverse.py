import os, sys
import json
import gzip
import numpy as np
import copy

from vlnverse.utils.glove_embedding import InstructionEmbedding

splits = ['train', 'val_seen', 'val_unseen', 'test']
types = ['coarse', 'fine']
# types = ['fine']
instruction_types = ['formal', 'natural', 'casual']
target_instruction_type = 'formal'

target_dir = 'data/vlnverse/raw_data/vlnverse/final_splits_with_distance_formal'
os.makedirs(target_dir, exist_ok=True)

# glove
source_folder = "data/datasets"
glove_embedder = InstructionEmbedding(source_folder=source_folder)
embedding_output_path = os.path.join(target_dir, "embeddings.json.gz")


# target_scan = 'kujiale_0294'
target_scan = '-1'

exclude_scene = ['kujiale_0157', 'kujiale_0145', 'kujiale_0143', 'kujiale_0134']

    
# 累计所有episode的距离
# for data_type in types:
#     for split in splits:
#         path_file = f'data/vlnverse/raw_data/vlnverse/final_splits/{data_type}/{split}/{split}.json.gz'
#         with gzip.open(path_file, 'rt', encoding='utf-8') as f:
#             data = json.load(f)
        
#         new_data = {'episodes': []}
#         new_episodes = []
#         for episode in data['episodes']:
#             if target_scan != '-1':
#                 if episode['scan'] != target_scan:
#                     continue
                    
#             # if episode['scan'] in exclude_scene:
#             #     print(f'[exclude] split: {split}, data_type: {data_type}, scan: {episode["scan"]}')
            
#             # 1. process distance
#             distance = 0
#             for i in range(len(episode['reference_path']) - 1):
#                 distance += np.linalg.norm(np.array(episode['reference_path'][i]) - np.array(episode['reference_path'][i + 1]))
#             episode['info']['geodesic_distance'] = distance
            
#             # 2. process instruction
#             if isinstance(episode['instruction']['instruction_text'], dict):
#                 instruction_count = 0
#                 for instruction_type, instruction_text in episode['instruction']['instruction_text'].items():
#                     if target_instruction_type != '-1':
#                         if instruction_type != target_instruction_type:
#                             continue
#                     new_episode = copy.deepcopy(episode)
#                     new_episode['instruction']['instruction_text'] = instruction_text
#                     tokens, vocab, embeddings = glove_embedder.embedding(instruction_text)
#                     new_episode['instruction']['instruction_tokens'] = tokens
#                     # new_episode['instruction']['instruction_tokens'] = episode['instruction']['instruction_tokens']
#                     new_episode['instruction']['instruction_type'] = instruction_type
#                     new_episode['episode_id'] = f'{episode["episode_id"]}_{instruction_type}'
#                     new_episode['trajectory_id'] = f'{episode["trajectory_id"]}_{instruction_type}'
#                     new_episodes.append(new_episode)
#                     instruction_count += 1
#             else:
#                 new_episodes.append(episode)
#         new_data['episodes'] = new_episodes
            
#         # save
#         os.makedirs(os.path.join(target_dir, data_type, split), exist_ok=True)
#         with gzip.open(os.path.join(target_dir, data_type, split, f'{split}.json.gz'), 'wt', encoding='utf-8') as f:
#             json.dump(new_data, f)
#         print(f'{split} {data_type} distance computed and saved in {os.path.join(target_dir, data_type, split, f"{split}.json.gz")}')

# save embeddings (for original cma & seq2seq)
# 根据最终 vocab 生成并保存 embeddings 到 embedding_output_path
# 形如：[PAD, UNK, word_0, word_1, ...]
# base_vocab = glove_embedder.base_vocab
# vectors = glove_embedder.vectors
# word_list = base_vocab["word_list"][2:]  # 跳过 PAD 和 UNK
# new_vectors = [vectors[w] for w in word_list if w in vectors]

# # 计算 UNK 与 PAD
# if len(new_vectors) == 0:
#     # 兜底：如无可用词向量，生成一个 50 维零向量（与默认 GloVe 50d 对齐）
#     pad_vector = [0.0] * 50
#     unk_vector = [0.0] * 50
# else:
#     unk_vector = list(np.mean(new_vectors, axis=0))
#     pad_vector = list(np.zeros(len(unk_vector)))

# embedding_vectors = [pad_vector] + [unk_vector] + new_vectors

# with gzip.open(embedding_output_path, 'wt', encoding='utf-8') as f:
#     json.dump(embedding_vectors, f)
# print(f'Embeddings saved to {embedding_output_path}')

# 统计每个type下每个split的数据分布
for data_type in types:
    for split in splits:
        path_file = f'data/vlnverse/raw_data/vlnverse/final_splits_with_distance_formal/{data_type}/{split}/{split}.json.gz'
        with gzip.open(path_file, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        print(f'{split} {data_type} has {len(data["episodes"])} episodes')
        