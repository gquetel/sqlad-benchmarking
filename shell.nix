let
  sources = import ./npins;
  pkgs = import sources.nixpkgs {
    config = {
      allowUnfree = true;
    };
  };
  pyEnv = import ./nix/python-env.nix;
in
(pkgs.buildFHSEnv {
  # buildFHSEnv is the only sustainable way I managed to exchange with CUDA
  # across different environement (NixOS + CUDA, Ubuntu + Nix + CUDA, ...).

  # The approach therefore is to rely on npins to have stable system packages version
  # And for python dependencies we pin them by specifying a version.
  name = "cuda-ml";

  targetPkgs = pkgs: with pkgs; [
    pyEnv.python314
    pyEnv.pip
    python314Packages.virtualenv
    treefmt
    ruff
  ];

  runScript = ''
    bash -c '
      if [ ! -d .venv ]; then
        python -m venv .venv
        source .venv/bin/activate
        pip install -e ".[dev]"
      else
        source .venv/bin/activate
        echo "Virtual environment already found. Run pip install -r requirements.txt to update."
      fi
      exec bash
    '
  '';
}).env
