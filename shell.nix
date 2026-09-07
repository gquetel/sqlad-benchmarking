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

  # Libraries the pip-installed wheels (torch, kaleido, ...) dlopen at runtime.
  # Modern torch/JAX wheels bundle the CUDA toolkit as nvidia-* deps, so we only
  # need libstdc++/libgcc + zlib here
  nixLibs = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib # libstdc++, libgcc_s
    pkgs.zlib
  ];

  # Where the NVIDIA driver's libcuda.so lives. Non-existent paths are ignored,
  # so the same string works on NixOS (/run/opengl-driver/lib) and on Ubuntu +
  # Nix (/usr/lib/x86_64-linux-gnu)
  driverLibs = "/run/opengl-driver/lib:/usr/lib/x86_64-linux-gnu";
in
pkgs.mkShell {
  name = "cuda-ml";

  packages = with pkgs; [
    pyEnv.python314
    pyEnv.uv
    treefmt
    ruff

    chromium # Required by kaleido to export plotly figures
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${nixLibs}:${driverLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

    # Make the bundled fonts visible to chromium when kaleido renders figures.
    export FONTCONFIG_FILE=${fontsConf}
    
    # Use the Nix chromium; else kaleido grabs its own broken downloaded chrome.
    export BROWSER_PATH=${pkgs.chromium}/bin/chromium

    # Build/refresh .venv-nix from uv.lock. The script also points uv at the Nix
    # interpreter. The cluster sources the same script and gets .venv-cluster.
    . ./tools/setup-env.sh

    # Activate the venv via PATH so it survives into the interactive shell,
    # whatever it is (bash, or a fish that the user's .bashrc exec-s).
    export VIRTUAL_ENV="$PWD/.venv-nix"
    export PATH="$VIRTUAL_ENV/bin:$PATH"
  '';
}
