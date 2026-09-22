# Documentation

From the repository root, build or serve the site in the project environment. Nix pins Python and uv. Without it,
install uv yourself, skip `nix-shell`, and source `.venv/bin/activate` instead:

```sh
nix-shell
uv sync --frozen --extra cpu
source .venv-nix-cpu/bin/activate
mkdocs build --config-file docs/mkdocs.yaml
mkdocs serve --config-file docs/mkdocs.yaml
```
