#!/usr/bin/env bash
# Shared helper to locate and activate the target conda environment (default: torch-tpu)
# or fall back to a local .venv if conda / the target env is unavailable.

find_conda_bin() {
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi
  local candidate
  for candidate in \
    "${CONDA_EXE:-}" \
    "/mnt/data/miniconda/bin/conda" \
    "${HOME}/miniconda3/bin/conda" \
    "${HOME}/anaconda3/bin/conda" \
    "/opt/conda/bin/conda"
  do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

conda_env_exists() {
  local conda_bin="$1"
  local env_name="$2"
  "${conda_bin}" env list | awk '{print $1}' | grep -qx "${env_name}"
}

activate_ocrtorch_env() {
  local repo_root="$1"
  local conda_env="${CONDA_ENV:-torch-tpu}"
  local venv_dir="${VENV_DIR:-${repo_root}/.venv}"

  if [[ "${CONDA_DEFAULT_ENV:-}" == "${conda_env}" ]]; then
    return 0
  fi
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    return 0
  fi

  local conda_bin
  if conda_bin="$(find_conda_bin)" && conda_env_exists "${conda_bin}" "${conda_env}"; then
    # shellcheck disable=SC1090
    eval "$("${conda_bin}" shell.bash hook)"
    conda activate "${conda_env}"
    return 0
  fi

  if [[ -d "${venv_dir}" ]]; then
    # shellcheck disable=SC1091
    source "${venv_dir}/bin/activate"
    return 0
  fi

  return 1
}
