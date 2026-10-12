#!/usr/bin/env bash
# The toolchain image's one step: TOGSim out of the third_party/PyTorchSim submodule, built in its
# own tree (its CMake writes to TOGSim/build) with the distribution's cmake + conan.
# setup/toolchain_tag.sh hashes this and the submodule's TOGSim tree.
#     setup/togsim.sh [-j N]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=versions.env
source "$HERE/versions.env"
JOBS=$(nproc)
[ "${1:-}" = -j ] && JOBS="$2"

test -f "$PYTORCHSIM_ROOT/TOGSim/CMakeLists.txt" \
  || { echo "third_party/PyTorchSim is empty -- git submodule update --init" >&2; exit 1; }
# TOGSim's own submodules (booksim, ramulator2). A checkout fetches them; an image build has no
# .git to fetch with and is handed them already (CI checks out recursively).
if [ -e "$PYTORCHSIM_ROOT/.git" ]; then
  git -C "$PYTORCHSIM_ROOT" submodule update --init --recursive --depth 1 --jobs 8
fi
for ext in booksim ramulator2; do
  test -f "$PYTORCHSIM_ROOT/TOGSim/extern/$ext/CMakeLists.txt" \
    || { echo "TOGSim/extern/$ext is empty -- git submodule update --init --recursive" >&2; exit 1; }
done
mkdir -p "$PYTORCHSIM_ROOT/TOGSim/build"; cd "$PYTORCHSIM_ROOT/TOGSim/build"
conan install .. --build=missing && cmake .. && make -j "$JOBS"
find . -name '*.o' -delete
test -x "$PYTORCHSIM_ROOT/TOGSim/build/bin/Simulator"
