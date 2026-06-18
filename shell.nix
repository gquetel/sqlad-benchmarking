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
  name = "cuda-ml";

  targetPkgs =
    pkgs: with pkgs; [
      pyEnv.python314
      pyEnv.uv
      treefmt
      ruff

      # Some system-level packages required
      procps # if not available breaks venv
      git # if not present mlflow can't log git SHA

      # Not required but i like it 
      fish

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

      # uv must use the Nix-provided interpreter, never download its own.
      export UV_PYTHON_DOWNLOADS=never
      export UV_PYTHON_PREFERENCE=only-system

      # Build/refresh .venv from uv.lock
      uv sync --frozen
      
      exec fish -C "source .venv/bin/activate.fish"
    '
  '';
}).env
