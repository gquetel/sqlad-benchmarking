FROM nixos/nix AS base

WORKDIR /app

COPY npins npins/
COPY dockerfiles/python-env.nix dockerfiles/python-env.nix

RUN nix-env -f dockerfiles/python-env.nix -i

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY src src/

RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["uvicorn", "src.mlops_sqldetect.api:app", "--host", "0.0.0.0", "--port", "8000"]
