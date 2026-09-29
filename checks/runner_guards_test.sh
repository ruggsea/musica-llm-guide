#!/bin/bash
# Proves run_multinode.sh's two start-up guards fire, with no GPUs and no job (seconds, safe on a login node).
# It cuts the guard lines out of the live runner, so it tests the real code:
#   1. Ray GPU count: stub `ray` reports 12 GPUs against TP*PP=16 -> must print RESULT: FAIL, exit 1; 16 -> carries on.
#   2. Worker IP: real node names resolve via getent; an unknown name with srun stubbed empty -> RESULT: FAIL, exit 1.
cd "$(dirname "$0")/.." && T=$(mktemp -d) && trap 'rm -rf $T' EXIT
R=scripts/run_multinode.sh
mkdir $T/fake && cat > $T/fake/ray.py <<'PY'
import os
def init(address=None): pass
def cluster_resources(): return {"GPU": float(os.environ["FAKE_GPUS"])}
def nodes(): return [{"Alive": True}] * (int(os.environ["FAKE_GPUS"]) // 4)
def shutdown(): pass
PY
awk '/# Verify Ray cluster/{f=1} f{print} f&&/Could not verify Ray cluster/{exit}' $R > $T/gpu.sh
awk '/for WORKER in \$WORKER_NODES/{f=1} f&&/WORKER_IP=|\[ -z/{print}' $R > $T/ip.sh
for g in 12 16; do
    echo "== Ray GPU check: stub reports $g GPUs, need 16"
    { echo 'TP=4; PP=4; HEAD_IP=127.0.0.1; RAY_PORT=6379'; cat $T/gpu.sh; echo 'echo "carried on"'; } > $T/t.sh
    FAKE_GPUS=$g PYTHONPATH=$T/fake bash $T/t.sh; echo "exit=$?"
done
echo "== Worker IP lookup: $(hostname -s) node names, then an unknown one with srun stubbed to return nothing"
{ echo 'srun() { :; }'; echo "for WORKER in ${NODES:-n3010-019 n3010-020} n9999-999; do"; cat $T/ip.sh
  echo '  echo "  $WORKER -> $WORKER_IP"'; echo done; echo 'echo "carried on"'; } > $T/t.sh
bash $T/t.sh; echo "exit=$?"
