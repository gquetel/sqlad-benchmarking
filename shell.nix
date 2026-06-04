let
  sources = import ./npins;
  pkgs = import sources.nixpkgs {
    config = {
      allowUnfree = true;
    };
  };
  pyEnv = import ./nix/python-env.nix;
  fontsConf = pkgs.makeFontsConf {
    fontDirectories = [
      pkgs.corefonts
      pkgs.dejavu_fonts
    ];
  };
in
(pkgs.buildFHSEnv {
  # buildFHSEnv is the only acceptable way I managed to exchange with CUDA
  # across different environement (NixOS + CUDA, Ubuntu + Nix + CUDA, ...).
  # However, it isolate the running environemnt from all already installed system
  # packages. Hence, the best way way to use this is to only enter the environemnt
  # to run the experiments, but have a normal shell parrallel to this.
  # I am not quite happy with this setup tho.

  # The approach therefore is to rely on npins to have stable system packages version
  # And for python dependencies we pin them by specifying a version.
  name = "cuda-ml";

  targetPkgs =
    pkgs: with pkgs; [
      pyEnv.python314
      pyEnv.pip
      python314Packages.virtualenv
      treefmt
      ruff

      # Some system-level packages I still want to have access to.
      fish
      procps # if not available breaks venv
      git # if not present mlflow can't log git SHA

      # Shared libs that pip-installed wheels link against
      zlib
      stdenv.cc.cc.lib # libstdc++, libgcc_s

      chromium # Required by kaleido to export plotly figures
      corefonts # Arial etc. (plotly's default font stack)
      dejavu_fonts # fallback
    ];

  runScript = ''
    bash -c '
      # Make the bundled fonts visible to chromium when kaleido renders figures.
      export FONTCONFIG_FILE=${fontsConf}
      # Use the Nix chromium; else kaleido grabs its own broken downloaded chrome.
      export BROWSER_PATH=${pkgs.chromium}/bin/chromium
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
