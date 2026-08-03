FROM nixos/nix AS base

WORKDIR /app

COPY npins npins/
COPY nix/python-env.nix nix/python-env.nix

RUN nix-env -f nix/python-env.nix -i

# uv installs into the Nix system interpreter and never downloads its own Python.
ENV UV_PYTHON_DOWNLOADS=never UV_SYSTEM_PYTHON=1

COPY requirements.txt requirements.txt
RUN uv pip install --system --no-cache -r requirements.txt

COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY src src/

RUN uv pip install --system --no-cache --no-deps .

ENTRYPOINT ["python", "-u", "src/sqlad_benchmarking/train.py"]
