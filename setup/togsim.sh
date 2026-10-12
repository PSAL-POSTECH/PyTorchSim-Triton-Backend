#!/usr/bin/env bash
# The toolchain image's one step: TOGSim, the third_party/TOGSim submodule, built in its own tree
# (its CMake writes to build/) with the distribution's cmake + conan.
# setup/toolchain_tag.sh hashes this and the submodule's tree.
#     setup/togsim.sh [-j N]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=versions.env
source "$HERE/versions.env"
JOBS=$(nproc)
[ "${1:-}" = -j ] && JOBS="$2"

test -f "$TOGSIM_DIR/CMakeLists.txt" \
  || { echo "third_party/TOGSim is empty -- git submodule update --init" >&2; exit 1; }
# TOGSim's own submodules (booksim, ramulator2). A checkout fetches them; an image build is a copy
# with no repository behind it and is handed them already (CI checks out recursively).
if [ -z "$(git -C "$TOGSIM_DIR" rev-parse --show-prefix 2>/dev/null || echo copy)" ]; then
  git -C "$TOGSIM_DIR" submodule update --init --recursive --depth 1 --jobs 8
fi
for ext in booksim ramulator2; do
  test -f "$TOGSIM_DIR/extern/$ext/CMakeLists.txt" \
    || { echo "third_party/TOGSim/extern/$ext is empty -- git submodule update --init --recursive" >&2; exit 1; }
done
mkdir -p "$TOGSIM_DIR/build"; cd "$TOGSIM_DIR/build"
conan install .. --build=missing && cmake .. && make -j "$JOBS"
find . -name '*.o' -delete
test -x "$TOGSIM_DIR/build/bin/Simulator"
