# Common, pinned, python environment.
# This file only manage python interpreter, not runtime system libs.
let
  sources = import ../npins;
  pkgs = import sources.nixpkgs { config.allowUnfree = true; };
in {
  inherit (pkgs) python314 uv;
}
