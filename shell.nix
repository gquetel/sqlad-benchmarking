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

  driverLibs = "/run/opengl-driver/lib";
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
    export SQLAD_EXTRA="''${SQLAD_EXTRA:-cpu}"
    export LD_LIBRARY_PATH=${nixLibs}:${driverLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

    # Expose Ubuntu NVIDIA drivers without adding the host libc to the search path.
    if [ "$SQLAD_EXTRA" != cpu ]; then
      _sqlad_driver_lib_dir=""
      for _sqlad_driver_lib in /usr/lib/x86_64-linux-gnu/libcuda.so* \
        /usr/lib/x86_64-linux-gnu/libnvidia-*.so* \
        /usr/lib/x86_64-linux-gnu/nvidia/current/libcuda.so* \
        /usr/lib/x86_64-linux-gnu/nvidia/current/libnvidia-*.so*; do
        if [ -f "$_sqlad_driver_lib" ]; then
          if [ -z "$_sqlad_driver_lib_dir" ]; then
            _sqlad_driver_lib_dir="$(mktemp -d)"
            export LD_LIBRARY_PATH="$_sqlad_driver_lib_dir:$LD_LIBRARY_PATH"
            trap 'rm -rf "$_sqlad_driver_lib_dir"' EXIT
          fi
          ln -sf "$_sqlad_driver_lib" "$_sqlad_driver_lib_dir/''${_sqlad_driver_lib##*/}"
        fi
      done
    fi

    # Make the bundled fonts visible to chromium when kaleido renders figures.
    export FONTCONFIG_FILE=${fontsConf}

    # Use the Nix chromium; else kaleido grabs its own broken downloaded chrome.
    export BROWSER_PATH=${pkgs.chromium}/bin/chromium

    export UV_PROJECT_ENVIRONMENT="$PWD/.venv-nix-$SQLAD_EXTRA"
    export UV_PYTHON_DOWNLOADS=never
    export UV_PYTHON_PREFERENCE=only-system
    echo "Nix shell ready ($SQLAD_EXTRA). Run: uv run --frozen --extra $SQLAD_EXTRA <command>"
  '';
}
