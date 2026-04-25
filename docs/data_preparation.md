# Data Preparation

VLNverse releases **four HuggingFace datasets** of its own, plus three external dependencies (robot embodiments, R2R preprocessed vocab, GloVe vectors) that the baselines reuse. Total disk: **~500 GB**. We strongly recommend downloading the two large VLNverse datasets to a scratch disk and symlinking them into `data/`; the smaller pieces go directly into `data/`.

## 1. Download summary

### VLNverse datasets

| Component       | HF repo                                                                                       | Size    | Target path under `data/`                                |
| --------------- | --------------------------------------------------------------------------------------------- | ------- | -------------------------------------------------------- |
| Envs            | [`Eyz/VLNVerse_scene`](https://huggingface.co/datasets/Eyz/VLNVerse_scene)                    | 312 GB  | `data/scene_data/vlnverse/` (symlink to scratch disk)    |
| Pre-built data  | [`Eyz/VLNVerse_data`](https://huggingface.co/datasets/Eyz/VLNVerse_data)                      | 179 GB  | `data/vlnverse/` (symlink — provides `raw_data/`, `traj_data/`) |
| Scene Graph     | [`Eyz/SceneSummary`](https://huggingface.co/datasets/Eyz/SceneSummary)                        | 37 MB   | `data/vlnverse/scene_graph/` (zip — unzip after download)|
| Room Meta       | [`Eyz/SceneMeta`](https://huggingface.co/datasets/Eyz/SceneMeta)                              | 2 MB    | `data/vlnverse/room_meta/` (zip — unzip after download)  |

### External dependencies


| Component        | Source                                                                                                          | Size    | Target path under `data/`                         |
| ---------------- | --------------------------------------------------------------------------------------------------------------- | ------- | ------------------------------------------------- |
| Embodiments      | [`InternRobotics/Embodiments`](https://huggingface.co/datasets/InternRobotics/Embodiments)                  | ~80 MB  | `data/Embodiments/`                               |
| R2R preprocessed | [Google Drive](https://drive.google.com/file/d/1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr/view)                          | 500 MB  | `data/datasets/R2R_VLNCE_v1-3_preprocessed/`      |
| GloVe 6B 50-d    | [Stanford NLP](http://nlp.stanford.edu/data/glove.6B.zip)                                                       | 164 MB  | `data/glove/glove.6B.50d.txt`                     |

## 2. Recommended layout

The two large datasets (Envs 312 GB, Pre-built 179 GB) are typically too big for the same disk that hosts the repo. We use two placeholders for the actual storage locations:

- `$VLNVERSE_DATA_ROOT` — scratch disk holding the **Pre-built** dataset (`raw_data/`, `traj_data/`) plus the unzipped `scene_graph/` and `room_meta/`.
- `$SCENE_ROOT` — separate location holding the **USD scene files** from the Envs dataset.

The Embodiments dataset is small (~80 MB) and gets downloaded directly into `data/Embodiments/` — no symlink needed.

After downloads, create the symlinks for the two large datasets from the repo root:

```bash
ln -s $VLNVERSE_DATA_ROOT  data/vlnverse
ln -s $SCENE_ROOT          data/scene_data/vlnverse
```

The resulting `data/` tree:

```
data/
├── Embodiments/                                               # InternRobotics/Embodiments (real dir, ~80 MB)
├── vlnverse            -> $VLNVERSE_DATA_ROOT                 # raw + traj + scene_graph + room_meta
│   ├── raw_data/
│   │   ├── final_splits/                                      # input to process_final_splits.py
│   │   ├── vlnverse/                                          # extended-vocab outputs (after preprocess)
│   │   └── vlnverse_r2r/                                      # r2r-vocab outputs (after preprocess)
│   ├── traj_data/
│   │   └── vlnverse/                                          # LeRobot trajectory features
│   ├── scene_graph/                                           # SceneSummary (unzipped)
│   └── room_meta/                                             # SceneMeta (unzipped)
├── scene_data/
│   └── vlnverse        -> $SCENE_ROOT                         # USD scene files (kujiale_*)
├── datasets/
│   └── R2R_VLNCE_v1-3_preprocessed/                           # base R2R vocab + embeddings
└── glove/
    └── glove.6B.50d.txt                                       # GloVe vectors
```

## 3. Step-by-step download

### 3.1 Pick scratch locations

```bash
export VLNVERSE_DATA_ROOT=/mnt/scratch/vlnverse_data   # ~200 GB needed
export SCENE_ROOT=/mnt/scratch/vlnverse_scenes         # ~315 GB needed
mkdir -p $VLNVERSE_DATA_ROOT $SCENE_ROOT
```

### 3.2 HuggingFace login (once)

```bash
pip install -U huggingface_hub
huggingface-cli login   # paste your token; stored at ~/.cache/huggingface/token
```

### 3.3 Download the four VLNverse HF datasets

```bash
# Pre-built data (179 GB) → unpacks raw_data/, traj_data/
huggingface-cli download Eyz/VLNVerse_data \
    --repo-type dataset \
    --local-dir $VLNVERSE_DATA_ROOT

# USD scenes (312 GB)
huggingface-cli download Eyz/VLNVerse_scene \
    --repo-type dataset \
    --local-dir $SCENE_ROOT

# Scene Graph (37 MB) — arrives zipped, unzip in place
huggingface-cli download Eyz/SceneSummary \
    --repo-type dataset \
    --local-dir $VLNVERSE_DATA_ROOT/scene_graph
unzip -q $VLNVERSE_DATA_ROOT/scene_graph/*.zip -d $VLNVERSE_DATA_ROOT/scene_graph/

# Room Meta (2 MB) — arrives zipped, unzip in place
huggingface-cli download Eyz/SceneMeta \
    --repo-type dataset \
    --local-dir $VLNVERSE_DATA_ROOT/room_meta
unzip -q $VLNVERSE_DATA_ROOT/room_meta/*.zip -d $VLNVERSE_DATA_ROOT/room_meta/
```

### 3.4 Symlink the two large datasets into `data/`

From the repo root:

```bash
ln -s $VLNVERSE_DATA_ROOT  data/vlnverse
mkdir -p data/scene_data
ln -s $SCENE_ROOT          data/scene_data/vlnverse
```

### 3.5 External dependencies

These three downloads are not part of the VLNverse release — they're reused from upstream projects (InternRobotics, VLN-CE, Stanford NLP).

```bash
# Embodiments — robot URDF/USD assets, from InternRobotics (~80 MB)
huggingface-cli download InternRobotics/Embodiments \
    --repo-type dataset \
    --local-dir data/Embodiments

# R2R preprocessed vocab + embeddings — Google Drive (~500 MB)
pip install gdown
mkdir -p data/datasets
gdown 1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr -O data/datasets/R2R_VLNCE_v1-3_preprocessed.zip
unzip -q data/datasets/R2R_VLNCE_v1-3_preprocessed.zip -d data/datasets/

# GloVe 6B 50-d — Stanford NLP (~164 MB)
mkdir -p data/glove
wget http://nlp.stanford.edu/data/glove.6B.zip -O /tmp/glove.6B.zip
unzip -p /tmp/glove.6B.zip glove.6B.50d.txt > data/glove/glove.6B.50d.txt
```

## 4. Verify

Run from the repo root — every line should print `OK`:

```bash
for p in \
  data/Embodiments \
  data/vlnverse/raw_data/final_splits \
  data/vlnverse/traj_data/vlnverse \
  data/vlnverse/scene_graph \
  data/vlnverse/room_meta \
  data/scene_data/vlnverse \
  data/datasets/R2R_VLNCE_v1-3_preprocessed/embeddings.json.gz \
  data/glove/glove.6B.50d.txt; do
  [ -e "$p" ] && echo "OK   $p" || echo "MISS $p"
done
```

## 5. Preprocess

Once the data is in place, generate the tokenised splits + vocab embeddings:

```bash
python scripts/process_final_splits.py --vocab extend   # recommended
# or
python scripts/process_final_splits.py --vocab r2r      # strict R2R 2504-token vocab
```

See the [Preprocess section in the main README](../README.md#preprocess) for what each mode produces.

## 6. Checkpoints

The baselines depend on two pretrained encoder checkpoints. **Both are required to run training and evaluation** — the configs hard-code these paths and the code will fail at startup without them.

- **DDPPO ResNet50** initializes the depth encoder (`VlnResnetDepthEncoder`) used by **every baseline in this repo** (CMA, Seq2Seq, RDP, and their `_clip` / `_vlnverse` variants).
- **LongCLIP-B** provides both the **text tokenizer and the text encoder** for variants whose instruction encoder is CLIP-based. In the CMA family this is `cma_clip`: it tokenises the instruction with `longclip.tokenize` and encodes it with the LongCLIP transformer (instead of the GloVe + LSTM path used by the other variants).

| Checkpoint                       | Used by                                                       | Role                                                | Target path                                                                  | Source                                                                                                                            |
| -------------------------------- | ------------------------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| DDPPO ResNet50                   | every baseline (CMA, Seq2Seq, RDP — all variants)             | depth-encoder backbone init (PointNav-pretrained)   | `checkpoints/ddppo-models/gibson-4plus-mp3d-train-val-test-resnet50.pth`     | [Habitat-DDPPO](https://dl.fbaipublicfiles.com/habitat/data/checkpoints/ddppo/gibson-4plus-mp3d-train-val-test-resnet50.pth)      |
| LongCLIP-B                       | CLIP-instruction variants (e.g. `cma_clip`, `rdp_*_clip`)     | instruction tokeniser + text encoder                | `checkpoints/clip-long/longclip-B.pt`                                        | [`BeichenZhang/LongCLIP-B`](https://huggingface.co/BeichenZhang/LongCLIP-B/resolve/main/longclip-B.pt)                            |

Download:

```bash
# DDPPO depth-encoder weights (~270 MB)
mkdir -p checkpoints/ddppo-models
wget https://dl.fbaipublicfiles.com/habitat/data/checkpoints/ddppo/gibson-4plus-mp3d-train-val-test-resnet50.pth \
    -O checkpoints/ddppo-models/gibson-4plus-mp3d-train-val-test-resnet50.pth

# LongCLIP-B weights (~1.7 GB)
mkdir -p checkpoints/clip-long
wget https://huggingface.co/BeichenZhang/LongCLIP-B/resolve/main/longclip-B.pt \
    -O checkpoints/clip-long/longclip-B.pt
```

> **Note.** The LongCLIP submodule (`vlnverse/model/basemodel/LongCLIP/`) ships only the *code*; the `.pt` weights are not part of the submodule and must be placed at the repo-root `checkpoints/clip-long/` path above — the configs do not look inside the submodule.

After downloading, the `checkpoints/` tree should look like:

```
checkpoints/
├── ddppo-models/
│   └── gibson-4plus-mp3d-train-val-test-resnet50.pth
└── clip-long/
    └── longclip-B.pt
```

Training runs add their own subdirectories under `checkpoints/<run_name>/` — the two folders above are pretrained inputs, not training outputs.
