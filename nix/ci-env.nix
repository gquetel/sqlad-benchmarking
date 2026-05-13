let
  sources = import ../npins;
  pkgs = import sources.nixpkgs { config.allowUnfree = true; };
  pyEnv = import ./python-env.nix;
in pkgs.mkShell {
  packages = [ pyEnv.python314 pyEnv.pip ];
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
  '';
}
