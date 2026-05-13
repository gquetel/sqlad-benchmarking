# Common, pinned, python environment to use in Dockerfiles.
let
  sources = import ../npins;
  pkgs = import sources.nixpkgs { config.allowUnfree = true; };
in {
  inherit (pkgs) python314;
  pip = pkgs.python314Packages.pip;
}
