{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: {
  packages = with pkgs; [
    git
    curl
    wget
  ];

  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
  '';

  languages = {
    rust = {
      enable = true;
    };
    javascript = {
      enable = true;
      npm.enable = true;
    };
    python = {
      enable = true;
      uv.enable = true;
    };
  };
}
