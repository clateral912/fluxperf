{
  description = "dual_round_benchmark development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
        python = pkgs.python312;
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            python
            python.pkgs.aiohttp
            python.pkgs.pyyaml
            python.pkgs.tqdm
            python.pkgs.prometheus_client
          ];

          shellHook = ''
            export PYTHONPATH=${python.pkgs.makePythonPath [
              python.pkgs.aiohttp
              python.pkgs.pyyaml
              python.pkgs.tqdm
              python.pkgs.prometheus_client
            ]}
          '';
        };
      }
    );
}
