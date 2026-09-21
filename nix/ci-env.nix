let
  sources = import ../npins;
  pkgs = import sources.nixpkgs { config.allowUnfree = true; };
  pyEnv = import ./python-env.nix;
in pkgs.mkShell {
  packages = [ pyEnv.python314 pyEnv.uv ];
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
    export UV_PROJECT_ENVIRONMENT="$PWD/.venv-nix-cpu"
    export UV_PYTHON_DOWNLOADS=never
    export UV_PYTHON_PREFERENCE=only-system
  '';
}
