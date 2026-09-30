# JAX on MUSICA H100 (not an LLM, but same cluster)

**Verified at Innsbruck (inn) only**, 2026-09-30, by the bohnanza project. Nobody has run it at vie or lnz yet;
treat those sites as untested until someone does and writes the job id here.

Source of the commands: `~/Projects/bohnanza/jobs/setup_jax.slurm` and `jobs/gputest.slurm` on artemis (copied
unchanged apart from the project paths), results from its `STATE.md` and the job logs at inn.

## What works

| | |
|---|---|
| Python | **3.12** (uv-managed, `cpython-3.12.12`). jax 0.11 needs >= 3.12; a 3.11 venv fails (inn job 155087). |
| Packages | `jax[cuda12]==0.11.2`, `flax==0.12.10`, `optax==0.2.8` (plus wandb, numpy, pytest) |
| Venv | `$MUSICA_ROOT/venvs/<name>` on /data, not in $HOME |
| Build | inn job 155710, CPU idle queue, prints `ok 0.11.2` / `SETUP_DONE` |
| GPU test | inn job 155715, 1 H100 (95,830 MiB), `jax.devices()` = `[CudaDevice(id=0)]`, compile+first step 4.5 s, EXIT 0 |
| Speed | bohnanza game engine, batch 16,384 x 200 steps: 6,366,118 and 6,010,205 decisions/s (2 reps, n=1 job) |

The speed is for that project's own benchmark (`scripts/bench.py`), useful only as "the GPU is doing real work".

## 1. Build the venv (CPU compute node, never the login node)

`uv` lives at `~/.local/bin/uv` (0.12.19 at inn). The 3.12 interpreter is uv's own; if it is missing at your site,
`~/.local/bin/uv python install 3.12` should fetch it (not tested as part of this recipe; bohnanza found it already there).

```bash
#!/bin/bash
#SBATCH -J jax-setup
#SBATCH -A p201276
#SBATCH -p zen4_0768
#SBATCH --qos idle_zen4_0768
#SBATCH -N 1
#SBATCH -t 00:45:00
#SBATCH -o %x_%j.log
source ~/musica-env.sh
set -e
V=$MUSICA_ROOT/venvs/myproject-jax
PY=~/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12
# a venv left by a failed non-3.12 attempt is rebuilt
if [ -d $V ] && ! $V/bin/python -V | grep -q 3.12; then rm -rf $V; fi
[ -d $V ] || ~/.local/bin/uv venv --python $PY $V
source $V/bin/activate
~/.local/bin/uv pip install "jax[cuda12]==0.11.2" "flax==0.12.10" "optax==0.2.8" numpy
python -c "import jax, flax, optax; print('ok', jax.__version__)"
echo SETUP_DONE
```

Only a CPU node is needed: pip wheels, no compiling. The CUDA libraries come from the `jax[cuda12]` wheels, so no
`module load` and no system CUDA.

## 2. Check the GPU is used (1 GPU, 15 min)

```bash
#!/bin/bash
#SBATCH -J jax-gputest
#SBATCH -A p201276
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos idle_zen4_0768_h100x4
#SBATCH --gres=gpu:1
#SBATCH -t 00:15:00
#SBATCH -o %x_%j.log
source ~/musica-env.sh && source $MUSICA_ROOT/venvs/myproject-jax/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import jax; print(jax.devices())"       # must print CudaDevice, not CpuDevice
# then a real workload of your own, timed
echo "EXIT $?"
```

A pass is `CudaDevice(id=0)` from `jax.devices()` **and** a timed run of your own code at GPU-like speed. jax falls
back to CPU silently when it cannot find CUDA, so a script that "runs" proves nothing on its own.

## Notes from the long runs (inn 156568, 1 GPU, 11 h, `--requeue`)

- Idle-queue jobs get preempted: write checkpoints and resume from the latest one; `#SBATCH --requeue` restarts it.
- Put checkpoints on `$MUSICA_ROOT/...` (/data), not in the 50 GB $HOME.
- No W&B key on MUSICA: `WANDB_MODE=offline`, then `wandb sync` the run folder from artemis after copying it back.
- Before submitting, `~/claude-code-config/bin/musica-pick-site.sh` as for every job. A site other than inn needs its
  own venv build first (disks are per site).
