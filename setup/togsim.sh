#!/usr/bin/env bash
# The toolchain image's one step: TOGSim and the machine configs of the PyTorchSim fork, a sparse
# checkout at the pin, built with the distribution's cmake + conan. setup/toolchain_tag.sh hashes this.
#     setup/togsim.sh [-j N]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=versions.env
source "$HERE/versions.env"
JOBS=$(nproc)
[ "${1:-}" = -j ] && JOBS="$2"

if [ ! -d "$PYTORCHSIM_ROOT/.git" ]; then
  git clone -q --filter=blob:none --no-checkout "$PYTORCHSIM_REPO" "$PYTORCHSIM_ROOT"
  git -C "$PYTORCHSIM_ROOT" sparse-checkout set TOGSim configs
fi
if [ "$(git -C "$PYTORCHSIM_ROOT" rev-parse HEAD)" != "$(git -C "$PYTORCHSIM_ROOT" rev-parse --verify -q "$PYTORCHSIM_REF^{commit}" || true)" ]; then
  git -C "$PYTORCHSIM_ROOT" fetch -q origin "$PYTORCHSIM_REF" 2>/dev/null || git -C "$PYTORCHSIM_ROOT" fetch -q origin
  git -C "$PYTORCHSIM_ROOT" checkout -q "$PYTORCHSIM_REF"
fi
# Submodules after the checkout: before it there is no .gitmodules and `git submodule` does nothing.
git -C "$PYTORCHSIM_ROOT" submodule update --init --recursive --depth 1 --jobs 8
mkdir -p "$PYTORCHSIM_ROOT/TOGSim/build"; cd "$PYTORCHSIM_ROOT/TOGSim/build"
conan install .. --build=missing && cmake .. && make -j "$JOBS"
find . -name '*.o' -delete
test -x "$PYTORCHSIM_ROOT/TOGSim/build/bin/Simulator"
