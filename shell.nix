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
  name = "sqlad-benchmarking";

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

    export SQLAD_EXTRA="''${SQLAD_EXTRA:-cpu}"
    export UV_PROJECT_ENVIRONMENT="$PWD/.venv-nix-$SQLAD_EXTRA"
    export UV_PYTHON_DOWNLOADS=never
    export UV_PYTHON_PREFERENCE=only-system
    echo "Nix shell ready ($SQLAD_EXTRA). Run: uv run --frozen --extra $SQLAD_EXTRA <command>"
  '';
}
