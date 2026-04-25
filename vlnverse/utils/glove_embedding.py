'''
使用GloVe词向量更新词汇表
'''

import gzip
import json
import os
import re
import string
import time
from collections import Counter
from pathlib import Path

import emoji
import numpy as np
from nltk.tokenize import word_tokenize


class TickTimer:
    def __init__(self, precision=3):
        self.cur_time = time.time()
        self.precision = precision

    def __call__(self, info=""):
        print("Time cost [{}] {}".format(info, round(time.time() - self.cur_time, self.precision)))
        self.cur_time = time.time()

    def reset(self):
        self.cur_time = time.time()


def pad_list(tokens, pad_num=200, pad_index=0):
    if len(tokens) >= pad_num:
        return tokens[:pad_num]
    else:
        tokens = tokens + [pad_index] * (pad_num - len(tokens))
        return tokens


def sentence_preprocess(s):
    s = emoji.demojize(s, delimiters=(" ", " "))
    mapping = {}
    for v in string.punctuation:
        mapping[v] = " " + v + " "
    mapping["—"] = " — "
    mapping["–"] = " – "
    trans = str.maketrans(mapping)
    s = s.translate(trans)
    return s


def remove_chat_format(s):
    pattern = r"\n?\d\.|^-\s|\n-\s|.*?:"
    return re.sub(pattern, "", s)


def build_vocab(instructions, old_vocab, min_count=1, vectors=[]):
    old_word_set = old_vocab["word_list"][2:]
    all_words = [w for s in instructions for w in word_tokenize(s) if w in vectors]
    word_cnt = []
    count = Counter(all_words)
    for word, num in count.most_common():
        if num >= min_count:
            word_cnt.append(word)
        else:
            break
    new_word_set = set(word_cnt)
    old_word_set = set(old_word_set)
    new_word_set = old_word_set.union(new_word_set)
    new_word_list = ["<pad>", "<unk>"] + sorted(list(new_word_set))
    print("Old vocabulary size: {}".format(len(old_word_set)))
    print("New vocabulary size: {}".format(len(new_word_list)))
    new_vocab = {
        "word_list": new_word_list,
        "word2idx_dict": {v: k for k, v in enumerate(new_word_list)},
        "itos": new_word_list,
        "stoi": {v: k for k, v in enumerate(new_word_list)},
        "num_vocab": len(new_word_list),
        "UNK_INDEX": 1,
        "PAD_INDEX": 0,
    }
    return new_vocab


def joint_vocab(vocab1, vocab2):
    word_set1 = set(vocab1["word_list"][2:])
    word_set2 = set(vocab2["word_list"][2:])
    print(len(word_set1) + 2, len(word_set2) + 2)
    new_word_list = ["<pad>", "<unk>"] + sorted(list(word_set1.union(word_set2)))
    print("Old vocabulary size: {} {}".format(len(word_set1) + 2, len(word_set2) + 2))
    print("New vocabulary size: {}".format(len(new_word_list)))
    new_vocab = {
        "word_list": new_word_list,
        "word2idx_dict": {v: k for k, v in enumerate(new_word_list)},
        "itos": new_word_list,
        "stoi": {v: k for k, v in enumerate(new_word_list)},
        "num_vocab": len(new_word_list),
        "UNK_INDEX": 1,
        "PAD_INDEX": 0,
    }
    return new_vocab


def joint_two_splits(source_folder, split1, split2, digitize_split2=False, offset=200000, new_split=None):
    # split1 must contain VLN-CE style episode id and trajectory id, that is, be numeric
    if new_split is None:
        new_split = "joint_" + split1.replace("joint_", "") + "_" + split2.replace("joint_", "")
    with gzip.open(source_folder / split1 / (split1 + ".json.gz"), "r") as f:
        data1 = json.load(f)
    with gzip.open(source_folder / split2 / (split2 + ".json.gz"), "r") as f:
        data2 = json.load(f)
    if digitize_split2:
        for i in range(len(data2["episodes"])):
            data2["episodes"][i]["episode_id"] = i
            data2["episodes"][i]["trajectory_id"] = i
    ep_ids1 = [v["episode_id"] for v in data1["episodes"]]
    traj_ids1 = [v["trajectory_id"] for v in data1["episodes"]]
    for v in ep_ids1:
        assert isinstance(v, int)
    for v in traj_ids1:
        assert isinstance(v, int)
    ep_ids2 = [v["episode_id"] for v in data2["episodes"]]

    if len(set(ep_ids1).intersection(set(ep_ids2))):
        max_ep_id = max(ep_ids1)
        max_traj_id = max(traj_ids1)

        for i in range(len(data2["episodes"])):
            if isinstance(data2["episodes"][i]["episode_id"], int):
                data2["episodes"][i]["episode_id"] = data2["episodes"][i]["episode_id"] + max_ep_id + 200000
            # elif data2["episodes"][i]["episode_id"].isdigit():
            #     data2["episodes"][i]["episode_id"] = (
            #         int(data2["episodes"][i]["episode_id"]) + max_ep_id + 200000
            #     )

            if isinstance(data2["episodes"][i]["trajectory_id"], int):
                data2["episodes"][i]["trajectory_id"] = data2["episodes"][i]["trajectory_id"] + max_traj_id + 200000
            # elif  data2["episodes"][i]["trajectory_id"].isdigit():
            #     data2["episodes"][i]["trajectory_id"] = (
            #         int(data2["episodes"][i]["trajectory_id"]) + max_traj_id + 200000
            #     )
    new_episodes = data1["episodes"] + data2["episodes"]
    new_vocab = joint_vocab(data1["instruction_vocab"], data2["instruction_vocab"])
    new_data = {"episodes": new_episodes, "instruction_vocab": new_vocab}
    for i in range(len(new_data["episodes"])):
        inst = new_data["episodes"][i]["instruction"]["instruction_text"]
        tokens = word_tokenize(sentence_preprocess(inst.lower()))
        tokens = [new_vocab["stoi"].get(v, new_vocab["UNK_INDEX"]) for v in tokens]
        tokens = pad_list(tokens)
        new_data["episodes"][i]["instruction"]["instruction_tokens"] = tokens
    os.makedirs(source_folder / new_split, exist_ok=True)
    with gzip.open(source_folder / new_split / (new_split + ".json.gz"), "wt") as f:
        json.dump(new_data, f)
    return new_data, new_split


def unify_vocab(source_folder, splits, vectors={}):
    # splits = [v for v in os.listdir(source_folder) if os.path.isdir(source_folder / v)]
    vocabs = []
    for split in splits:
        with gzip.open(source_folder / split / (split + ".json.gz"), "r") as f:
            data = json.load(f)
        vocabs.append(data["instruction_vocab"])
    new_vocab = vocabs[0]
    for i in range(1, len(vocabs)):
        new_vocab = joint_vocab(new_vocab, vocabs[i])
    print("Unify {} with vocab size {}".format(splits, len(new_vocab["word_list"])))
    for split in splits:
        with gzip.open(source_folder / split / (split + ".json.gz"), "r") as f:
            new_data = json.load(f)
        new_data["instruction_vocab"] = new_vocab
        for i in range(len(new_data["episodes"])):
            inst = new_data["episodes"][i]["instruction"]["instruction_text"]
            tokens = word_tokenize(sentence_preprocess(inst.lower()))
            tokens = [new_vocab["stoi"].get(v, new_vocab["UNK_INDEX"]) for v in tokens]
            tokens = pad_list(tokens)
            new_data["episodes"][i]["instruction"]["instruction_tokens"] = tokens
        with gzip.open(source_folder / split / (split + ".json.gz"), "wt") as f:
            json.dump(new_data, f)
    if vectors:
        word_list = new_vocab["word_list"][2:]
        new_vectors = [vectors[w] for w in word_list]
        unk_vector = list(np.mean(new_vectors, axis=0))
        pad_vector = list(np.zeros(len(unk_vector)))
        new_vectors = [pad_vector] + [unk_vector] + new_vectors
        with gzip.open(source_folder / "embeddings.json.gz", "wt") as f:
            json.dump(new_vectors, f)


def update_vocab(source_folder, splits=["train", "test", "val_seen", "val_unseen"]):
    for split in splits:
        with gzip.open(source_folder / split / (split + ".json.gz"), "r") as f:
            data = json.load(f)
        vocab = data["instruction_vocab"]
        instructions = [v["instruction"]["instruction_text"] for v in data["episodes"]]
        with open("data/glove/glove.6B.50d.txt", "r") as f:
            vectors = {}
            for line in f:
                vals = line.rstrip().split(" ")
                vectors[vals[0]] = [float(x) for x in vals[1:]]
        new_vocab = build_vocab(instructions, vocab, min_count=1, vectors=vectors)
        data["instruction_vocab"] = new_vocab
        with gzip.open(source_folder / split / (split + ".json.gz"), "wt") as f:
            json.dump(data, f)


class InstructionEmbedding:
    def __init__(self, source_folder, glove_path="data/glove/glove.6B.50d.txt"):
        """
        初始化类
        source_folder: 存放原始数据的文件夹路径
        glove_path: GloVe词向量文件路径
        """
        self.source_folder = Path(source_folder)
        self.glove_path = glove_path
        self.vectors = self._load_glove_vectors()
        self.base_vocab = self._load_base_vocab()

    def _load_glove_vectors(self):
        """加载GloVe词向量"""
        vectors = {}
        with open(self.glove_path, "r") as f:
            for line in f:
                vals = line.rstrip().split(" ")
                vectors[vals[0]] = [float(x) for x in vals[1:]]
        return vectors

    def _load_base_vocab(self):
        """加载基础词汇表"""
        with gzip.open(self.source_folder / "train" / "train.json.gz", "r") as f:
            data = json.load(f)
        return data["instruction_vocab"]

    def process_new_instructions(self, instructions, output_dir):
        """
        处理新的instructions并保存结果
        instructions: 新instruction列表
        output_path: 输出文件路径
        """
        start_time = time.time()
        # 构建新的词汇表
        new_vocab = build_vocab(instructions=instructions, old_vocab=self.base_vocab, min_count=1, vectors=self.vectors)

        # 处理instructions的tokens
        processed_instructions = []
        for inst in instructions:
            tokens = word_tokenize(sentence_preprocess(inst.lower()))
            token_ids = [new_vocab["stoi"].get(v, new_vocab["UNK_INDEX"]) for v in tokens]
            token_ids = pad_list(token_ids)
            processed_instructions.append({"instruction_text": inst, "instruction_tokens": token_ids})

        # 生成词向量
        word_list = new_vocab["word_list"][2:]  # 跳过PAD和UNK
        new_vectors = [self.vectors[w] for w in word_list if w in self.vectors]
        unk_vector = list(np.mean(new_vectors, axis=0))
        pad_vector = list(np.zeros(len(unk_vector)))
        embedding_vectors = [pad_vector] + [unk_vector] + new_vectors

        # 保存结果
        os.makedirs(output_dir, exist_ok=True)

        result = {"instruction_vocab": new_vocab, "instructions": processed_instructions}

        instr_path = os.path.join(output_dir, "processed_instructions.json.gz")
        emb_path = os.path.join(output_dir, "processed_embeddings.json.gz")

        with gzip.open(instr_path, "wt") as f:
            json.dump(result, f)

        with gzip.open(emb_path, "wt") as f:
            json.dump(embedding_vectors, f)

        print(f"Processed instructions saved to {instr_path}")
        print(f"Processed embeddings saved to {emb_path}")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        return result

    def embedding(self, instruction):
        """
        将输入的指令token化并生成embedding
        instruction: 输入的指令字符串
        返回: (token_ids, updated_vocab, updated_embeddings)
        """
        # 预处理指令
        processed_inst = sentence_preprocess(instruction.lower())
        tokens = word_tokenize(processed_inst)

        # 检查是否有新词
        new_words = [w for w in tokens if w not in self.base_vocab["word_list"]]

        if new_words:
            # 如果有新词，更新词汇表和embedding
            new_vocab = build_vocab(
                instructions=[instruction], old_vocab=self.base_vocab, min_count=1, vectors=self.vectors
            )

            # 更新词向量
            word_list = new_vocab["word_list"][2:]
            new_vectors = [self.vectors[w] for w in word_list if w in self.vectors]
            unk_vector = list(np.mean(new_vectors, axis=0))
            pad_vector = list(np.zeros(len(unk_vector)))
            updated_embeddings = [pad_vector] + [unk_vector] + new_vectors

            # 更新基础词汇表
            self.base_vocab = new_vocab

            # 生成token ids
            token_ids = [new_vocab["stoi"].get(v, new_vocab["UNK_INDEX"]) for v in tokens]
            token_ids = pad_list(token_ids)

            return token_ids, new_vocab, updated_embeddings
        else:
            # 如果没有新词，直接使用现有词汇表
            token_ids = [self.base_vocab["stoi"].get(v, self.base_vocab["UNK_INDEX"]) for v in tokens]
            token_ids = pad_list(token_ids)

            return token_ids, self.base_vocab, None


if __name__ == "__main__":
    # 初始化
    dataset_dir = "data/datasets/R2R_VLNCE_v1-3_preprocessed"
    output_dir = "data/datasets/real_deploy"
    embedding_output_path = os.path.join(dataset_dir, "update_embeddings.json.gz")

    embedder = InstructionEmbedding(source_folder=dataset_dir)

    # 处理新的instructions
    new_instructions = ["turn left and walk to the door", "go straight ahead until you reach the kitchen"]
    # result = embedder.process_new_instructions(
    #     instructions=new_instructions,
    #     output_dir=output_dir
    # )

    # 读取train.json.gz
    with gzip.open(os.path.join(dataset_dir, "train", "train.json.gz"), "r") as f:
        data = json.load(f)
    instructions = [v["instruction"]["instruction_text"] for v in data["episodes"]]

    token_ids, vocab, embeddings = embedder.embedding(instructions[0])
    print(token_ids)
    print(vocab)
    print(embeddings)
