FROM nixos/nix AS base

WORKDIR /app

COPY npins npins/
COPY nix/python-env.nix nix/python-env.nix

RUN nix-env -f nix/python-env.nix -i

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY src src/

RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "src/mlops_sqldetect/train.py"]
