#!/usr/bin/env bash
# The distribution layer on the Coprocessor environment image (ubuntu 22.04): python3.11 as
# python3 (the MLIR bindings' ABI), cmake 3.26 + conan 1.56 for TOGSim. Root.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=versions.env
source "$HERE/versions.env"

export DEBIAN_FRONTEND=noninteractive

# deadsnakes: jammy's python3 is 3.10 and the MLIR bindings in LLVM_SHA's install are cpython-3.11.
apt-get update
apt-get install -y --no-install-recommends \
  software-properties-common ca-certificates gnupg curl
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update

# build-essential/m4/pkg-config/ninja: triton, the plugin, the dialect; zlib: triton links it.
# TOGSim's libraries are conan's, and gem5's are the environment image's own.
apt-get install -y --no-install-recommends \
  build-essential git curl wget m4 pkg-config ninja-build procps file xz-utils bzip2 \
  python3.11 python3.11-dev python3.11-venv python3-distutils \
  libssl-dev zlib1g-dev
rm -rf /var/lib/apt/lists/*

# python3 is python3.11 from here on; add-apt-repository ran above, on 3.10, for that reason.
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
python3.11 -m ensurepip --upgrade
python3.11 -m pip install --no-cache-dir --upgrade pip setuptools wheel
python3 --version

python3.11 -m pip install --no-cache-dir \
  "cmake==$HOST_CMAKE_VERSION" "conan==$CONAN_VERSION" numpy PyYAML "nanobind==$NANOBIND_VERSION" pybind11

# TOGSim builds with _GLIBCXX_USE_CXX11_ABI=0, so conan's packages must too (--detect says libstdc++11).
conan profile new default --detect --force
conan profile update settings.compiler.libcxx=libstdc++ default
conan profile show default
