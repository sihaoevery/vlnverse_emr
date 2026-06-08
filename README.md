<div align="center">

# VLNverse

**Baselines for the VLNverse benchmark — [ECCV 2026 EMR Workshop Challenge](https://emr-workshop.github.io/)**

[![Paper](https://img.shields.io/badge/arXiv-2512.19021-b31b1b.svg)](https://arxiv.org/abs/2512.19021)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://sihaoevery.github.io/vlnverse/)
[![Workshop](https://img.shields.io/badge/ECCV%202026-EMR%20Workshop-7B68EE)](https://emr-workshop.github.io/)
[![Challenge](https://img.shields.io/badge/Challenge-Codabench-blue)](https://www.codabench.org/competitions/17009/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

```
  ┌──────────────────────┐      ┌────────────────┐      ┌─────────────────┐      ┌──────────────┐
  │  natural-language    │ ───▶ │  VLN agent     │ ───▶ │  trajectory in  │ ───▶ │  NE · SR ·   │
  │  instruction         │      │                │      │  a 3D scene     │      │  SPL · OSR   │
  └──────────────────────┘      └────────────────┘      └─────────────────┘      └──────────────┘
       task definition              this repo                 simulator                metrics
```

## What is VLNverse

VLNverse is a large-scale, extensible benchmark for **V**ersatile, **E**mbodied, **R**ealistic **S**imulation and **E**valuation of vision-language navigation. It unifies previously fragmented navigation tasks — classic VLN, Object-Goal, and Visual-Reference navigation — under a single toolkit, with full-kinematics agents and a physics-grounded simulator. The paper is at [arXiv:2512.19021](https://arxiv.org/abs/2512.19021); see the [project page](https://sihaoevery.github.io/vlnverse/) for dataset statistics and qualitative results.


## Challenge submission

**This repository** provides reference baselines and the training / evaluation pipeline for the VLNverse Challenge at the [ECCV 2026 EMR Workshop](https://emr-workshop.github.io/).

The challenge is hosted on **[Codabench](https://www.codabench.org/competitions/17009/)**. Run the evaluator on a `challenge` split and upload the resulting `submission_<granularity>_challenge.json.gz.zip` (see [Output by split type](#output-by-split-type)) to the matching competition phase.

> Key dates (see workshop site for the canonical schedule):
> - Paper submission: **July 12, 2026**
> - Challenge deadline: **July 31, 2026**

## Quickstart

1. **Clone** the repo with submodules:
   ```bash
   git clone --recursive https://github.com/sihaoevery/vlnverse_emr.git && cd vlnverse_emr
   ```
2. **Set up the environment** — Isaac Sim 4.5.0 + PyTorch 2.5.1 + project requirements. See [Installation](#installation).
3. **Download data and checkpoints** — ~500 GB total. See [`docs/data_preparation.md`](docs/data_preparation.md).
4. **Preprocess, train, and evaluate:**
   ```bash
   python scripts/process_final_splits.py --vocab extend
   bash scripts/train/start_train.sh --model cma_clip --name my_first_run
   bash scripts/eval/start_eval_chunked.sh \
       --config scripts/eval/configs/h1_cma_clip_cfg_vlnverse_coarse.py
   ```

## Installation

VLNverse uses **Isaac Sim 4.5.0** for continuous physics-based evaluation and **PyTorch 2.5.1 (CUDA 11.8)** for training. The main setup has four steps, plus an optional editable install:

### 1. Download Isaac Sim 4.5.0

Download the standalone zip from the [official Isaac Sim 4.5.0 page](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html) and unzip it to any location — we'll refer to this path as `ISAACSIM_ROOT`.


### 2. Create the conda environment

```bash
conda create -n vlnverse python=3.10 libxcb=1.14 -y
conda activate vlnverse
```

### 3. Install InternUtopia and link Isaac Sim

```bash
pip install internutopia

# Interactive: paste your $ISAACSIM_ROOT path when prompted.
python -m internutopia.setup_conda_pypi

# Reactivate so the new env vars take effect.
conda deactivate && conda activate vlnverse

cd $ISAACSIM_ROOT
source setup_conda_env.sh
```

### 4. Install PyTorch and project requirements

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements/isaac_requirements.txt
pip install -r requirements/train.txt
pip install -r requirements/eval.txt
```

### 5. (Optional) Editable install

```bash
pip install -e .
```

Skip this if you only run the bundled launchers (`scripts/train/start_train.sh`, `scripts/eval/start_eval_one_gpu.sh`, `scripts/eval/start_eval_chunked.sh`) — they patch `sys.path` themselves. Run it if you want to `import vlnverse` from your own scripts or notebooks outside this repo.

> **Note.** `gradio` (latest, unpinned) in `requirements/isaac_requirements.txt` can clash with the pinned `fastapi==0.110.0` / `starlette==0.36.3`. If you don't need the demo UI (`scripts/eval/vln_gradio_backend.py`), comment that line out before installing.

## Data & Checkpoints

### Download

VLNverse data spans four HuggingFace datasets plus a few small external dependencies (robot embodiments, R2R preprocessed vocab, GloVe vectors) — **~500 GB total**. The baselines additionally require two pretrained encoder checkpoints: a **DDPPO ResNet50** depth encoder and **LongCLIP-B** for CLIP-based models. See [`docs/data_preparation.md`](docs/data_preparation.md) for the full guide (multi-disk symlinks, step-by-step commands, checkpoint downloads, verification).

| Dataset | Contents | Link |
|---|---|---|
| **Envs**           | USD scene files (~312 GB)           | [Hugging Face](https://huggingface.co/datasets/Eyz/VLNVerse_scene) |
| **Pre-built Data** | Training & evaluation data (~179 GB)| [Hugging Face](https://huggingface.co/datasets/Eyz/VLNVerse_data)  |
| **Scene Graph**    | Object relationships (~37 MB)       | [Hugging Face](https://huggingface.co/datasets/Eyz/SceneSummary)   |
| **Room Meta**      | Scene metadata (~2 MB)              | [Hugging Face](https://huggingface.co/datasets/Eyz/SceneMeta)      |

Pretrained checkpoints — both must be in place before training:

| Checkpoint            | Used by                            | Role                                          | Target path                                                                |
|-----------------------|------------------------------------|-----------------------------------------------|----------------------------------------------------------------------------|
| **DDPPO ResNet50**    | all baselines      | depth-encoder                  | `checkpoints/ddppo-models/gibson-4plus-mp3d-train-val-test-resnet50.pth`   |
| **LongCLIP-B**        | CLIP-based baselines               | instruction tokenization      | `checkpoints/clip-long/longclip-B.pt`                                      |

### Preprocess

The preprocessing script tokenises instructions and builds the vocabulary + embedding matrix. It has two modes:

```bash
# Strict R2R vocab (2504 tokens, out-of-vocab → <unk>).
python scripts/process_final_splits.py --vocab r2r

# Extended vocab: R2R prefix (2504) + GloVe-covered new VLNverse words.
python scripts/process_final_splits.py --vocab extend
```

Outputs:

```
data/vlnverse/raw_data/
├── vlnverse_r2r/              # --vocab r2r
│   ├── {coarse,fine}/{train,val_seen,val_unseen,test,challenge}/*.json.gz
│   ├── mixed_splits/…
│   └── embeddings.json.gz     # 2504 × 50
└── vlnverse/                  # --vocab extend
    ├── {coarse,fine}/{train,val_seen,val_unseen,test,challenge}/*.json.gz
    ├── mixed_splits/…
    └── embeddings.json.gz     # (2504 + new) × 50
```

The `--vocab extend` mode requires `data/glove/glove.6B.50d.txt` (GloVe 50-d vectors). Both modes read base R2R vocab/embeddings from `data/datasets/R2R_VLNCE_v1-3_preprocessed/`.

#### The `challenge` split

`challenge` is the subset used for the evaluation-server leaderboard: 150 episodes per granularity, a scene-stratified, representative sample of `test` (all 53 scenes covered; coarse and fine share no trajectory). When `scripts/challenge_subset.txt` (the published list of trajectory ids) is present, `process_final_splits.py` builds it automatically — slicing those ids straight out of the already-tokenized `test` split, so it carries the same tokens/vocab and, like `test`, has no ground truth. To evaluate on it, list `'challenge'` in a config's `split_data_types` (the eval `*_vlnverse_{coarse,fine}.py` configs already do); the dataloader resolves it to `{coarse,fine}/challenge/challenge.json.gz` by convention. Running with no GT writes the submission files (see [Output by split type](#output-by-split-type)); upload the zip — `submission_<granularity>_challenge.json.gz.zip` — to the evaluation server.

## Training

All baselines use the same launcher; GPU count depends on the model:

```bash
bash scripts/train/start_train.sh --model <key> --name <run_name>
```

Checkpoints land in `checkpoints/<run_name>/ckpts/`, TensorBoard logs in `checkpoints/<run_name>/tensorboard/`.

### CMA family

| `--model` key    | Text encoder            | Vocabulary                | Training data                                      | Config file                                    |
| ---------------- | ----------------------- | ------------------------- | -------------------------------------------------- | ---------------------------------------------- |
| `cma`            | GloVe (50-d) + LSTM     | R2R 2504 (OOV → `<unk>`)  | `data/vlnverse/raw_data/vlnverse_r2r/mixed_splits` | `scripts/train/configs/cma.py`                 |
| `cma_vlnverse`   | GloVe (50-d) + LSTM     | VLNverse extended         | `data/vlnverse/raw_data/vlnverse/mixed_splits`     | `scripts/train/configs/cma_vlnverse.py`        |
| `cma_clip`       | LongCLIP text encoder   | — (CLIP tokeniser)        | `data/vlnverse/raw_data/vlnverse/mixed_splits`     | `scripts/train/configs/cma_clip_vlnverse.py`   |

### Seq2Seq family

| `--model` key    | Text encoder            | Vocabulary                | Training data                                      | Config file                                          |
| ---------------- | ----------------------- | ------------------------- | -------------------------------------------------- | ---------------------------------------------------- |
| `seq2seq`        | GloVe (50-d) + LSTM     | R2R 2504 (OOV → `<unk>`)  | `data/vlnverse/raw_data/vlnverse_r2r/mixed_splits` | `scripts/train/configs/seq2seq.py`                   |
| `seq2seq_clip`   | LongCLIP text encoder   | — (CLIP tokeniser)        | `data/vlnverse/raw_data/vlnverse/mixed_splits`     | `scripts/train/configs/seq2seq_clip_vlnverse.py`     |

### RDP family

| `--model` key    | Text encoder            | Vocabulary                | Training data                                      | Config file                                          |
| ---------------- | ----------------------- | ------------------------- | -------------------------------------------------- | ---------------------------------------------------- |
| `rdp_vlnverse`   | LongCLIP text encoder   | — (CLIP tokeniser)        | `data/vlnverse/raw_data/vlnverse/mixed_splits`     | `scripts/train/configs/rdp_vlnverse.py`              |

RDP uses 4 GPUs by default (see `CUDA_VISIBLE_DEVICES` in `start_train.sh`); the other families use 1.

Hyperparameters (epochs, batch size, learning rate, eval cadence) live in the config files above — edit them directly to tune a run.

## Evaluation

A launcher starts the agent server (`vlnverse/agent/utils/server.py`) and the evaluator (`scripts/eval/eval.py`). To evaluate a specific checkpoint, edit `agent.ckpt_path` in your config file to point at `checkpoints/<run_name>/ckpts/checkpoint-<step>`.

There are two launchers:

```bash
# Recommended for any full-split / unattended run (sidesteps the Isaac Sim memory stall):
bash scripts/eval/start_eval_chunked.sh \
    --config scripts/eval/configs/h1_<model>_cfg_vlnverse_<granularity>.py

# Simpler alternative for quick / short runs:
bash scripts/eval/start_eval_one_gpu.sh \
    --config scripts/eval/configs/h1_<model>_cfg_vlnverse_<granularity>.py
```

They handle Isaac Sim's mid-run memory stall differently, as explained below.

### Handling Isaac Sim stalls: two launchers

Isaac Sim can **stall mid-run** during long evaluations: a scene's static colliders are not
freed on `env.reset`, so host RAM climbs every episode until one physics step hangs — OOM on
low-RAM boxes, an unrecoverable CPU-spin on high-RAM ones. Both launchers below detect this
and restart automatically; both resume via LMDB, so completed episodes are never re-run.
They differ in **what triggers the restart**.

**`start_eval_one_gpu.sh` — fixed inactivity timer.** A watchdog polls the eval log; if it
stays silent for `DEADLOCK_THRESHOLD` seconds the run is killed and restarted (i.e. *after*
a stall). Both knobs are plain variables at the top of the script.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `DEADLOCK_THRESHOLD` | `6 * 60` (6 min) | log-silence that counts as a stall → restart |
| `MONITOR_INTERVAL` | `60` (s) | how often the watchdog checks |

The timer must exceed your cold-start time, or the run keeps restarting before it begins. The
`6 min` default is tuned for an RTX 4090 — a cold RtPso shader compile (first run on a fresh
cache) can leave the log silent for several minutes, and slower GPUs take longer, so raise
`DEADLOCK_THRESHOLD` to match your own hardware's startup time.

**`start_eval_chunked.sh` — RSS memory budget (recommended for full / unattended runs).**
Rather than wait for a stall, the evaluator exits cleanly *before* one, as soon as this
process's host RSS crosses a budget; a fresh process then resumes. This adapts to scene size
(the leak is larger for bigger scenes) and stops cleanly on "No more episodes". Two timers
are backstops for a rare mid-episode stall that outruns the budget.

| Hyperparameter (env var) | Default | Meaning |
|---|---|---|
| `VLN_RSS_BUDGET_MIB` | `24000`, capped to free RAM | restart once process RSS exceeds this — primary, scene-adaptive trigger |
| `VLN_INACTIVITY_TIMEOUT` | `420` (s) | kill + resume if stdout is silent this long |
| `VLN_CHUNK_HARDCAP` | `21600` (s, 6 h) | per-chunk wall-clock backstop |

The `24000` default is tuned for our setup (**128 GB RAM, RTX 4090**, where the stall appears
around ~28 GB RSS); set `VLN_RSS_BUDGET_MIB` to suit your own RAM — lower for more safety
margin, higher for fewer Isaac reboots.

`start_eval_one_gpu.sh` ignores the `VLN_*` variables; with `VLN_RSS_BUDGET_MIB` unset the
chunked logic is a no-op, so the evaluator behaves exactly as before.

Available eval configs:

- `scripts/eval/configs/h1_cma_cfg.py` — CMA on the R2R-vocab splits (MP3D scenes)
- `scripts/eval/configs/h1_cma_clip_cfg_vlnverse_{coarse,fine}.py` — CLIP-CMA on VLNverse
- `scripts/eval/configs/h1_seq2seq_cfg.py` — Seq2Seq on the R2R-vocab splits (MP3D scenes)
- `scripts/eval/configs/h1_seq2seq_cfg_vlnverse_{coarse,fine}.py` — Seq2Seq-CLIP on VLNverse
- `scripts/eval/configs/h1_rdp_cfg_vlnverse_{coarse,fine,fine_train}.py` — RDP on VLNverse (`_fine_train` evaluates on the train split as a sanity check)
- `scripts/eval/configs/h1_rdp_cfg_vlnverse_fine_parallel.py` — experimental multi-env RDP eval (`env_num=2, proc_num=2`); may deadlock in Isaac Sim, so prefer the single-env configs.

All outputs land under `logs/<task_name>/`.

### Output by split type

Whether a split has ground truth (`reference_path` in its JSON) drives what the evaluator can compute on-the-fly. The submission JSON is always written so the predicted trajectory can be scored offline against held-out GT. Each save writes a timestamped `submission_<granularity>_<split>_<ts>.json.gz` **and** a stable-named `submission_<granularity>_<split>.json.gz.zip` (a no-recompression zip of the latest json.gz; unzipping recovers it byte-for-byte). **Upload the `.zip` to the [evaluation server](https://www.codabench.org/competitions/17009/).**

| Split                       | GT in JSON | `<dataset_type>_result.json` fields                 | `submission_<granularity>_<split>_<ts>.json.gz` (+ `.zip`) |
|-----------------------------|------------|-----------------------------------------------------|----------------------------------------------------|
| `val_seen` / `val_unseen`   | ✅ yes     | `Count`, `TL`, `FR`, `StR`, **`NE`, `OS`, `SR`, `SPL`** | predicted trajectory per episode                   |
| `test`                      | ❌ no      | `Count`, `TL`, `FR`, `StR` + `note` pointing to submission | predicted trajectory per episode (score offline)   |
| malformed (mixed GT)        | ⚠️ partial | `Count`, `TL`, `FR`, `StR` + `note: malformed split`  | predicted trajectory; `log.warning` at startup      |

The submission JSON mirrors the GT episode schema — `episode_id` / `trajectory_id` / `scan` / `scene_id` / `start_position` / `start_rotation` / `reference_path` (predicted trajectory) / `goals.position` (stop point) / `goals.radius` (success distance) / `info.geodesic_distance = -1` — so it can be scored offline against the released GT using the same `NE` / `OS` / `SR` / `SPL` definitions as `val_seen` / `val_unseen`, without re-running the simulator. It's written after every episode termination via atomic rename, so partial results survive crashes / SIGINT / power loss. The `<ts>` is fixed per launcher invocation, so within one run the same file is overwritten in place; restarting the eval (e.g. after a crash) produces a new timestamped file alongside the previous one.

## Repo layout

```
vlnverse/             # core package — imported as `import vlnverse`
├── agent/            # inference agents (CMA, CMA-CLIP, Seq2Seq, RDP, …)
├── dataset/          # LeRobot-backed training datasets
├── model/            # neural networks (instruction / visual encoders, policy heads)
├── trainer/          # IL training loops
├── evaluator/        # VLN-PE evaluation harness
├── env/              # Habitat / InternUtopia env adapters
├── configs/          # Pydantic config classes (model / trainer / evaluator)
└── projects/         # simulator extensions (InternUtopia VLN extension, dataloaders)

scripts/
├── process_final_splits.py     # VLNverse data preprocessing (run first)
├── train/                      # training launchers + per-model configs
└── eval/                       # evaluation launchers + per-task configs

data/                 # datasets & assets (not tracked)
checkpoints/          # training outputs (not tracked)
logs/                 # evaluation outputs (not tracked)
tests/                # unit / integration tests
```


## Citation

If you use VLNverse in your research, please cite:

```bibtex
@article{vlnverse2025,
  title   = {VLNverse: A Versatile, Embodied, Realistic Benchmark for Vision-Language Navigation},
  author  = {Lin, Sihao and Li, Zerui and Zhao, Xunyi and Zhou, Gengze and Wang, Liuyi and Wei, Rong and Tang, Rui and Li, Juncheng and Wang, Hanqing and Pang, Jiangmiao and van den Hengel, Anton and Liu, Jiajun and Wu, Qi},
  journal = {arXiv preprint arXiv:2512.19021},
  year    = {2025},
  url     = {https://arxiv.org/abs/2512.19021}
}
```

## License

The code is released under the [MIT License](LICENSE). The VLNverse scenes and dataset are released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — free to use and build on for research, with attribution to VLNverse, and not for commercial use.

## Acknowledgements

This repository builds on several excellent open-source projects:

- [Habitat-lab / Habitat-sim](https://github.com/facebookresearch/habitat-lab) — discrete-environment VLN simulator
- [InternUtopia](https://github.com/InternRobotics/InternUtopia) — physics-based continuous simulator for embodied eval
- [VLN-CE](https://github.com/jacobkrantz/VLN-CE) — reference implementation of the CMA and Seq2Seq baselines
- [LongCLIP](https://github.com/beichenzbc/Long-CLIP) — long-context CLIP text encoder used by CLIP-based baselines
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) — policy head used by the RDP family
- [LeRobot](https://github.com/huggingface/lerobot) — trajectory data format
