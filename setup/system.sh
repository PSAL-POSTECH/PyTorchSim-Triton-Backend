#!/usr/bin/env bash
# What the toolchain needs from the distribution, on a clean ubuntu:22.04: python3.11 AS
# python3 (the MLIR bindings are built against it), cmake 3.26 + conan 1.56 for TOGSim, and
# torch for the kernels' reference values. The tnpu base image used to carry this. Root.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=versions.env
source "$HERE/versions.env"

export DEBIAN_FRONTEND=noninteractive

# deadsnakes, because jammy's python3 is 3.10 and the MLIR bindings in LLVM_SHA's install
# are cpython-3.11 ABI. This is the diagnosis that moved the old base from torchsim_base to
# tnpu; it is the same diagnosis, so it is the same PPA.
apt-get update
apt-get install -y --no-install-recommends \
  software-properties-common ca-certificates gnupg curl
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update

# EVERY LINE HAS A CONSUMER. build-essential/m4/pkg-config/ninja: triton, spike, pk and the
# dialect. device-tree-compiler + boost-regex: spike's configure. protobuf/hdf5/png/
# perftools/zlib/bz2: TOGSim and the conan boost it builds. libelf/libssl: riscv-pk and curl.
apt-get install -y --no-install-recommends \
  build-essential git curl wget m4 pkg-config ninja-build procps file xz-utils bzip2 \
  python3.11 python3.11-dev python3.11-venv python3-distutils \
  libssl-dev libelf-dev zlib1g-dev libbz2-dev \
  device-tree-compiler \
  libboost-dev libboost-regex-dev libboost-system-dev \
  libprotobuf-dev libprotoc-dev protobuf-compiler \
  libhdf5-dev libpng-dev libgoogle-perftools-dev
rm -rf /var/lib/apt/lists/*

# python3 IS python3.11 from here on, and nothing apt runs afterwards: restore.sh's venvs,
# the driver triton install and `import mlir` are all that interpreter, and tests/run.py
# shells out to sys.executable. add-apt-repository ran above, on 3.10, for that reason.
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
python3.11 -m ensurepip --upgrade
python3.11 -m pip install --no-cache-dir --upgrade pip setuptools wheel
python3 --version

# cmake 3.26 and conan 1.56 ARE the TOGSim pair -- restore.sh's build_tools() puts the newer
# cmake LLVM 23 needs in its own venv precisely so these two are left alone.
python3.11 -m pip install --no-cache-dir \
  "cmake==$HOST_CMAKE_VERSION" "conan==$CONAN_VERSION" numpy PyYAML nanobind pybind11
python3.11 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
  "torch==$TORCH_VERSION"

# TOGSim compiles with -D_GLIBCXX_USE_CXX11_ABI=0 (its CMakeLists.txt), so the conan packages
# it links must be the same ABI. conan 1.56's --detect answers libstdc++11 for gcc 11, which
# is the other one, and the link fails on std::string symbols nobody reads as an ABI mismatch.
conan profile new default --detect --force
conan profile update settings.compiler.libcxx=libstdc++ default
conan profile show default
