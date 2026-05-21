# Rocket-League-Goalie-Reward-Engineering

Comparative reward engineering project for Rocket League goalie agents using PPO and RocketSim.

## Demos

| Sparse (40M) | Dense (40M) | Context v3 (40M) |
| :---: | :---: | :---: |
| <img width="260" alt="phase1_sparse40m" src="https://github.com/user-attachments/assets/a4a811f9-fcd6-478c-9b40-975f99ef4e13" /> | <img width="260" alt="phase1_dense40m" src="https://github.com/user-attachments/assets/163f13fa-b89d-4852-a8e3-72a4c474b580" /> | <img width="260" alt="phase1_contextv31" src="https://github.com/user-attachments/assets/390dbdd2-0c8f-4a3d-ac4f-c8a01fe489d4" /> |
| Little learning. | Active ball interaction. | Corrected reward, still low contact. |

**Full playbacks (all 5 runs):** [sparse](https://drive.google.com/file/d/1eYVl0I2iisNpYHPvyfPsBx8LpehN3P2t/view?usp=sharing) · [dense](https://drive.google.com/file/d/1ZQlz9Vw4eBNNCMKKI-vKwDNA-GQ46lTA/view?usp=sharing) · [context_40m](https://drive.google.com/file/d/15dLHT5lIlw1QM5xDYNnH29SG7xlMWGB0/view?usp=sharing) · [context_v2](https://drive.google.com/file/d/1dCuEvIvFEb1LMDZQiGj6DjJRMKydyFlO/view?usp=sharing) · [context_v3](https://drive.google.com/file/d/1fWNQtuZb-rYYGxlsYWN_4OnvjH7lIQZd/view?usp=sharing)

## Simple Project Map

- trainer.py
Builds the environment, picks one reward strategy, starts PPO, and saves checkpoints.

- rewards.py
Contains the three reward functions you are comparing (sparse, dense, context-aware).

- trainer.py (GoalieWandbMetricsLogger)
Contains the custom goalie metrics logger class that sends domain-specific metrics directly to W&B.

You usually only change two things:
1. strategy name (sparse, dense, context)
2. run name

## Environment Setup (Windows + Conda)

You will need Miniconda and Git installed.

Create and activate the environment:

```powershell
conda create -n rl_goalie python=3.9 -y
conda activate rl_goalie
```

Install CUDA-enabled PyTorch (RTX 3060 target):

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

Install the RLGym v2 stack:

```powershell
python -m pip install rlgym rlgym-rocket-league
python -m pip install rocketsim
python -m pip install git+https://github.com/AechPro/rlgym-ppo
```

If you need to deactivate or uninstall:

```powershell
conda deactivate
conda env remove -n rl_goalie -y
```

Optional visualizer support:

```powershell
python -m pip install "rlgym[rl-rlviser]"
```

## Path Fixes

If VS Code shows shell integration as Basic, `conda activate rl_goalie` may appear to work while `python` still points to base Python.

Quick check:

```powershell
where python
python --version
python -m pip --version
```

Expected interpreter path should include `miniconda3\\envs\\rl_goalie`.

If it does not, try this:

```powershell
# run with the exact env interpreter
C:/Users/<your-user>/miniconda3/envs/rl_goalie/python.exe -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
C:/Users/<your-user>/miniconda3/envs/rl_goalie/python.exe -m pip install rlgym rlgym-rocket-league
C:/Users/<your-user>/miniconda3/envs/rl_goalie/python.exe -m pip install rocketsim
C:/Users/<your-user>/miniconda3/envs/rl_goalie/python.exe -m pip install git+https://github.com/AechPro/rlgym-ppo
C:/Users/<your-user>/miniconda3/envs/rl_goalie/python.exe -m pip install "rlgym[rl-rlviser]"
C:/Users/<your-user>/miniconda3/envs/rl_goalie/python.exe -m pip install wandb
```

To fix PowerShell activation permanently:

```powershell
conda init powershell
```

Then fully close and reopen VS Code (or at least open a brand-new terminal) and activate again.

To undo (if you want):
```powershell
conda init --reverse powershell
```

To prevent auto-activation of base env (if you want):
```powershell
conda config --set auto_activate false
```

## If profile.ps1 Is Blocked

If you see "running scripts is disabled on this system" for the profile script, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
Unblock-File C:/Users/<your-user>/OneDrive/Documents/WindowsPowerShell/profile.ps1
```

## Asset Dumper Requirement

If you are following older `rlgym_sim` guides (like the original Zealan intro), you do need the arena asset dumper and a `collision_meshes` folder.

For this project's current stack (`rlgym` + `rlgym-rocket-league` + `rocketsim`), standard soccar collision meshes are already bundled with `rlgym-rocket-league`, so you can run training without manually dumping assets.

Only use the asset dumper if you are:

- using legacy `rlgym_sim`, or
- using custom/non-default maps where bundled meshes are not enough.

Then open a new terminal and verify:

```powershell
conda activate rl_goalie
where python
python --version
python -m pip --version
```

## Logging

The trainer uses W&B logging by default and also sends custom goalie metrics through `GoalieWandbMetricsLogger` in [trainer.py](trainer.py).

Use `--no-wandb` for runs where you do not want remote logging (for example, local visualization checks).

Before training, run:

```powershell
wandb login
```

Custom W&B metrics logged in this project:

- Goalie/ball_speed_norm_mean
- Goalie/ball_y_norm_mean
- Goalie/mean_boost_mean
- Goalie/touch_rate_mean
- Goalie/dist_to_own_goal_mean
- Goalie/dist_to_ball_mean
- Goalie/metric_samples
- Goalie/cumulative_timesteps

## Training Commands

Run sparse baseline:

```powershell
python trainer.py --strategy sparse --run-name sparse_v1 --n-proc 8 --device cuda --timestep-limit 40000000
```

Run physics-dense:

```powershell
python trainer.py --strategy dense --run-name dense_v1 --n-proc 8 --device cuda --timestep-limit 40000000
```

Run context-aware:

```powershell
python trainer.py --strategy context --run-name context_v1 --n-proc 8 --device cuda --timestep-limit 40000000
```

Checkpoints are saved under the `agents/` folder in strategy-specific subfolders.

## Visualize a Saved Checkpoint (RLViser)

You can render directly from `trainer.py` by loading a saved checkpoint and enabling render mode.

RLViser binary requirement (Windows):

- Download `rlviser.exe` from https://github.com/VirxEC/rlviser/releases/latest
- Place `rlviser.exe` in this project root (same folder as `trainer.py`)
- Run the visualization command from this project root

If RLViser opens and then closes with `memory allocation of ... bytes failed`, use RLViser `v0.8.2` specifically for this stack:

- https://github.com/VirxEC/rlviser/releases/download/v0.8.2/rlviser.exe

Without `rlviser.exe` in the current working directory, render startup will fail.

Example using the latest checkpoint inside `agents/sparse_40m`:

```powershell
python trainer.py --strategy sparse --run-name sparse_render --checkpoint-load-folder agents/sparse_40m --n-proc 1 --device cuda --render --render-delay 0.03 --timestep-limit 40500000 --no-wandb
```

Notes:

- `--checkpoint-load-folder` accepts either a specific timestep folder or the parent run folder containing numeric checkpoint subfolders.
- Use `--n-proc 1` for stable local rendering.
- Set `--timestep-limit` higher than the loaded checkpoint timesteps so the run actually continues and renders.
- `--no-wandb` disables W&B resume from checkpoints (`load_wandb=False`), so a new local W&B run folder is not created for render-only checks.

## Notes

- RocketSim collision meshes are required for simulation fidelity.
- Keep all comparison runs on matching environment settings and PPO hyperparameters.

## Useful Links

- https://rlgym.org/
- https://github.com/AechPro/rlgym-ppo
- https://github.com/ZealanL/RLGym-PPO-Guide/blob/main/intro.md
