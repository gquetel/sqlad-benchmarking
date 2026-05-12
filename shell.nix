let
  sources = import ./npins;
  pkgs = import sources.nixpkgs {
    config = {
      allowUnfree = true;
    };
  };
in
(pkgs.buildFHSEnv {
  name = "cuda-ml";

  targetPkgs = pkgs: with pkgs; [
    python314
    python314Packages.pip
    python314Packages.virtualenv
  ];

  runScript = ''
    bash -c '
      if [ ! -d .venv ]; then
        python -m venv .venv
        source .venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
      else
        source .venv/bin/activate
        echo "Virtual environment already found. Run pip install -r requirements.txt to update."
      fi
      exec bash
    '
  '';
}).env