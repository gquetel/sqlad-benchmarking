# Build or refresh the project environment from uv.lock. Source this file, do not run it.
#
# Nix provides Python locally; uv installs it on the cluster. Separate environments
# let both use the same checkout on shared storage.
#
#   . tools/setup-env.sh                   # CUDA 12.6
#   SQLAD_EXTRA=cu130 . tools/setup-env.sh  # Blackwell GPUs
#   . tools/setup-env.sh submit            # job submission, without PyTorch
#
# Build both GPU environments on a compute node; setup exceeds the submit node's memory limit.
# Jobs select the environment for their GPU. See docs/source/slurm.md.
#
#   srun --partition=CPU --mem=16G --pty bash -lc '. tools/setup-env.sh'
#   srun --partition=CPU --mem=16G --pty bash -lc 'SQLAD_EXTRA=cu130 . tools/setup-env.sh'

_sqlad_repo="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

# Stop if the shell cannot locate the repository, as can happen outside Bash.
if [ ! -f "${_sqlad_repo}/pyproject.toml" ]; then
    echo "setup-env.sh: cannot locate the repository (got '${_sqlad_repo}'); source it from bash." >&2
    return 1 2>/dev/null || exit 1
fi

# Use SQLAD_EXTRA=cpu without a GPU, or cu130 for Blackwell.
: "${SQLAD_EXTRA:=cu126}"

if [ -n "${IN_NIX_SHELL:-}" ]; then
    export UV_PROJECT_ENVIRONMENT="${_sqlad_repo}/.venv-nix"
    # A downloaded interpreter does not run on NixOS.
    export UV_PYTHON_DOWNLOADS=never
    export UV_PYTHON_PREFERENCE=only-system
else
    # Keep these environment paths in sync with cuda_builds in configs/slurm.yaml.
    case "${SQLAD_EXTRA}" in
    cu126 | cpu) _sqlad_venv=".venv-cluster" ;;
    *) _sqlad_venv=".venv-cluster-${SQLAD_EXTRA}" ;;
    esac
    export UV_PROJECT_ENVIRONMENT="${_sqlad_repo}/${_sqlad_venv}"
    # Cluster nodes cannot run the Nix-provided Python.
    export UV_PYTHON_PREFERENCE=only-managed

    # Module setup requires unset-variable checks to be disabled.
    if [ -n "${SQLAD_MODULES:-}" ]; then
        case $- in *u*) _sqlad_u=1; set +u ;; *) _sqlad_u=0 ;; esac
        if [ -f /etc/profile.d/z00_lmod.sh ]; then . /etc/profile.d/z00_lmod.sh; fi
        module purge
        for _sqlad_m in ${SQLAD_MODULES}; do module load "${_sqlad_m}"; done
        if [ "${_sqlad_u}" = 1 ]; then set -u; fi
    fi

    # Add the user's uv installation to PATH if needed.
    if ! command -v uv >/dev/null 2>&1 && [ -f "${HOME}/.local/bin/env" ]; then
        . "${HOME}/.local/bin/env"
    fi
    if ! command -v uv >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        . "${HOME}/.local/bin/env"
    fi
    uv python install
fi

if [ "${1:-}" = "submit" ]; then
    export UV_PROJECT_ENVIRONMENT="${_sqlad_repo}/.venv-submit"
    # Install the project without its training dependencies.
    uv sync --frozen --only-group submit &&
        uv pip install --quiet --python "${UV_PROJECT_ENVIRONMENT}/bin/python" --no-deps -e "${_sqlad_repo}"
else
    uv sync --frozen --extra "${SQLAD_EXTRA}"
fi
