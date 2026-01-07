# SG_BabyAI — Scale Generalization in BabyAI OneRoom (MiniGrid)

This repo contains a small, reproducible experiment on **scale generalization** in **BabyAI OneRoom**: train a PPO agent on a small room (S8) and evaluate how performance degrades as the room size increases (S12/S16/S20). It also includes a simple improvement: **mix-scale training** (train on S8/S12/S16).

## Repo Structure

- `train_sb3_ppo.py` — baseline training (PPO on a single env)
- `train_mixscale_ppo.py` — mix-scale training (PPO on multiple room sizes with vectorized envs)
- `eval_minigrid.py` — evaluation script (supports 500-episode runs)
- `smoke_minigrid.py` — quick sanity check (renders a short rollout GIF)
- `render_gif.py` — render policy rollouts as GIFs (with SUCCESS/FAIL label)
- `results.txt` — example results / notes
- `A_baseline_S8_success.gif`, `B_baseline_S20_fail.gif`, `C_mix_S20_success.gif`, `D_mix_S8.gif` — example visualizations

---

## 1) Environment Setup (Conda)

If your `base` environment already has `torch` installed, the safest approach is cloning it to avoid Conda trying to download/resolve torch again.

```bash
# If you previously started creating an environment and it got stuck, stop it:
# Ctrl + C

conda deactivate
conda env remove -n minigrid_gen -y

# Clone base -> keep your existing torch
conda create -n minigrid_gen --clone base -y
conda activate minigrid_gen

python -c "import torch; print('torch=', torch.__version__)"
```

Install dependencies:

```bash
pip install -U pip
pip install "gymnasium>=0.29" "minigrid>=2.3.0" "stable-baselines3[extra]>=2.2.1" \
  opencv-python imageio matplotlib tqdm
```

Quick check:

```bash
which python
python -c "import gymnasium, minigrid; import stable_baselines3; print('ok')"
```

---

## 2) Smoke Test (one command)

Render a quick rollout GIF to confirm everything works:

```bash
python smoke_minigrid.py --env_id BabyAI-OneRoomS8-v0 --out smoke_oneroom.gif
```

---

## 3) Baseline: Train PPO on S8 (200K steps)

```bash
python train_sb3_ppo.py --train_env BabyAI-OneRoomS8-v0 --timesteps 200000 --save ppo_oneroomS8_200k.zip
```

---

## 4) Mix-Scale Training: Train PPO on S8/S12/S16 (200K steps)

```bash
python train_mixscale_ppo.py --train_envs BabyAI-OneRoomS8-v0 BabyAI-OneRoomS12-v0 BabyAI-OneRoomS16-v0 \
  --n_envs 8 --timesteps 200000 --save ppo_mix_S8S12S16_200k.zip --seed 0
```

---

## 5) Evaluation (500 episodes)

Baseline (S8-trained) evaluated on S8/S12/S16/S20:

```bash
python eval_minigrid.py --model ppo_oneroomS8_200k.zip --episodes 500 --seed0 0 --device cpu --envs \
  BabyAI-OneRoomS8-v0 BabyAI-OneRoomS12-v0 BabyAI-OneRoomS16-v0 BabyAI-OneRoomS20-v0
```

Mix-scale model evaluated on S8/S12/S16/S20:

```bash
python eval_minigrid.py --model ppo_mix_S8S12S16_200k.zip --episodes 500 --seed0 0 --device cpu --envs \
  BabyAI-OneRoomS8-v0 BabyAI-OneRoomS12-v0 BabyAI-OneRoomS16-v0 BabyAI-OneRoomS20-v0
```

---

## 6) Visualization (GIFs for the report)

Activate your env first:

```bash
conda activate minigrid_gen
```

A) Baseline (S8-only) success:

```bash
python render_gif.py --model ppo_oneroomS8_200k.zip --env_id BabyAI-OneRoomS8-v0 \
  --out A_baseline_S8_success.gif --seed 0 --deterministic --tile_size 48 --label
```

B) Baseline on S20 fail (shows scale generalization failure):

```bash
python render_gif.py --model ppo_oneroomS8_200k.zip --env_id BabyAI-OneRoomS20-v0 \
  --out B_baseline_S20_fail.gif --seed 0 --deterministic --tile_size 32 --label
```

C) Mix-scale on S20 success:

```bash
python render_gif.py --model ppo_mix_S8S12S16_200k.zip --env_id BabyAI-OneRoomS20-v0 \
  --out C_mix_S20_success.gif --seed 0 --deterministic --tile_size 32 --label
```

D) Mix-scale on S8 (still strong):

```bash
python render_gif.py --model ppo_mix_S8S12S16_200k.zip --env_id BabyAI-OneRoomS8-v0 \
  --out D_mix_S8.gif --seed 0 --deterministic --tile_size 48 --label
```

Tip: if `seed=0` doesn’t give you the “success/fail” example you want, change the seed (e.g., 1/2/3/…).
