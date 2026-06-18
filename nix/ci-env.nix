let
  sources = import ../npins;
  pkgs = import sources.nixpkgs { config.allowUnfree = true; };
  pyEnv = import ./python-env.nix;
in pkgs.mkShell {
  packages = [ pyEnv.python314 pyEnv.uv ];
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
    # uv must use the Nix-provided interpreter, never download its own.
    export UV_PYTHON_DOWNLOADS=never
    export UV_PYTHON_PREFERENCE=only-system
  '';
}
