# Build or refresh the project environment from uv.lock. Source this file, do not run it.
#
# The packages always come from uv.lock. Only the interpreter changes: nix gives one on
# the dev machines, uv installs its own on the cluster, which has no /nix. Each source
# gets its own directory, because one checkout is visible to both over NFS.
#
#   . tools/setup-env.sh            # full environment
#   . tools/setup-env.sh submit     # torch-free submit environment
#
# The full cluster environment exceeds the submit node's memory limit.
#
#   srun --partition=CPU --mem=16G --pty bash -lc '. tools/setup-env.sh'

_sqlad_repo="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

# A shell without BASH_SOURCE (dash, e.g. a hook whose shebang is /bin/sh) resolves the line
# above to "/", which builds the venv in the filesystem root. Stop instead.
if [ ! -f "${_sqlad_repo}/pyproject.toml" ]; then
    echo "setup-env.sh: cannot locate the repository (got '${_sqlad_repo}'); source it from bash." >&2
    return 1 2>/dev/null || exit 1
fi

# Set SQLAD_EXTRA=cpu on a machine with no GPU (the lames, CI).
: "${SQLAD_EXTRA:=cu126}"

if [ -n "${IN_NIX_SHELL:-}" ]; then
    export UV_PROJECT_ENVIRONMENT="${_sqlad_repo}/.venv-nix"
    # A downloaded interpreter does not run on NixOS.
    export UV_PYTHON_DOWNLOADS=never
    export UV_PYTHON_PREFERENCE=only-system
else
    export UV_PROJECT_ENVIRONMENT="${_sqlad_repo}/.venv-cluster"
    # A nix interpreter needs /nix, which the cluster nodes do not have.
    export UV_PYTHON_PREFERENCE=only-managed

    # On demand only: the Lmod init script fails under `set -u`.
    if [ -n "${SQLAD_MODULES:-}" ]; then
        case $- in *u*) _sqlad_u=1; set +u ;; *) _sqlad_u=0 ;; esac
        if [ -f /etc/profile.d/z00_lmod.sh ]; then . /etc/profile.d/z00_lmod.sh; fi
        module purge
        for _sqlad_m in ${SQLAD_MODULES}; do module load "${_sqlad_m}"; done
        if [ "${_sqlad_u}" = 1 ]; then set -u; fi
    fi

    # uv installs itself in the home directory, which a batch shell can miss.
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
