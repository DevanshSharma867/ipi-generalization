# Storage layout on BITS Sharanga (source from jobs and cluster-side scripts).
# /scratch/csisnlp_18 has no quota but files are deleted 30 days after last modification (login banner), so it
# holds only regenerable things: venvs, the HF cache, uv/pip caches, rebuildable transfer data. Home (40G soft
# quota) holds what cannot be regenerated: results, checkpoints we still need, manifests; results are also copied
# back to the laptop and committed.
SCRATCH=${SCRATCH:-/scratch/csisnlp_18}
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
# the HF token stays in home (never copied, printed or logged); huggingface_hub reads it from this path
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$SCRATCH/uv-cache}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$SCRATCH/pip-cache}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$SCRATCH/uv-python}
VENVS=$SCRATCH/venvs

touch_scratch() {  # reset the purge clock on everything in scratch (metadata only; -h: touch links, not targets)
  [ -d "$SCRATCH" ] || return 0
  find "$SCRATCH" -xdev -mindepth 1 -exec touch -c -h {} + 2>/dev/null || true
  echo "touched $SCRATCH at $(date)"
}

check_home_free() {  # check_home_free <GiB>: fail if home has less than that left under the soft quota
  local need_kb=$(( $1 * 1024 * 1024 )) line used quota
  command -v lfs >/dev/null || return 0
  line=$(lfs quota -u "$USER" /home | awk '$1 == "/home" {print $2, $3}')
  used=${line% *}; quota=${line#* }
  used=${used%\*}
  if [ "$quota" -gt 0 ] && [ $(( quota - used )) -lt $need_kb ]; then
    echo "home has $(( (quota - used) / 1024 )) MiB left under its quota; need $1 GiB. Free space first."; exit 1
  fi
  echo "home: $(( (quota - used) / 1024 / 1024 )) GiB free under quota"
}
