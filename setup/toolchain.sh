#!/usr/bin/env bash
# The pinned toolchain under /workspace: what the route needs beyond the environment image
# (setup/system.sh lays the distribution layer first; $VCIX_ENV_IMAGE already carries spike,
# gem5, the RISC-V toolchain and pk at the submodule's pins).
#
#     ./setup/toolchain.sh                every step
#     ./setup/toolchain.sh llvm triton    only those
#     ./setup/toolchain.sh -j 8 triton    cap build parallelism
#
# Then: pytorchsim-triton-compiler doctor
#
# Steps: llvm (downloaded), triton (v3.7.0, built once), vcix (the model), ext (extension/,
# against the triton build). Every step is idempotent -- rerunning a finished one is a no-op.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=versions.env
source "$HERE/versions.env"

JOBS=$(( $(nproc) > 4 ? $(nproc) - 4 : 1 ))
STEPS=()

while [ $# -gt 0 ]; do
  case "$1" in
    -j) JOBS="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *) STEPS+=("$1"); shift ;;
  esac
done
[ ${#STEPS[@]} -eq 0 ] && STEPS=(llvm triton vcix ext)

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
want() { for s in "${STEPS[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

# Fetch exactly one commit. A full llvm-project clone is several GB of history
# we never look at; --depth 1 on the pinned sha is ~200MB.
fetch_commit() {
  local dir="$1" repo="$2" sha="$3"
  mkdir -p "$dir"; cd "$dir"
  if [ ! -d .git ]; then
    git init -q
    git remote add origin "$repo"
  fi
  if [ "$(git rev-parse HEAD 2>/dev/null || true)" != "$sha" ]; then
    git fetch --depth 1 origin "$sha"
    git checkout -q FETCH_HEAD
  fi
  echo "  $dir @ $(git rev-parse HEAD)"
}

# cmake 3.26.4 on the image is pinned for PyTorchSim's TOGSim/conan build and is
# too old for LLVM 23. Upgrading it in place would break TOGSim, so a newer cmake
# lives in its own venv used only here.
build_tools() {
  if [ ! -x /workspace/build-tools/bin/cmake ]; then
    log "build tools (isolated cmake/ninja -- the image's cmake 3.26 is left alone)"
    /usr/bin/python3.11 -m venv /workspace/build-tools
    /workspace/build-tools/bin/pip install -q --upgrade pip
    /workspace/build-tools/bin/pip install -q "cmake==3.31.6" ninja
  fi
  # triton and the extension build against pybind11; the image's python3.11 has it, a box may not.
  /workspace/build-tools/bin/python -c 'import pybind11' 2>/dev/null \
    || /workspace/build-tools/bin/pip install -q "pybind11==3.1.0"
  CMAKE=/workspace/build-tools/bin/cmake
}


# THIS REPOSITORY DOES NOT BUILD LLVM, AND MUST NOT START. Building it belongs
# to PSAL-POSTECH/llvm-project, which carries the VCIX ops on its own branch and
# publishes one install per revision; here the pin names the asset and nothing
# more. The release tag is derived from $LLVM_SHA, so a moved pin fails loudly.
if want llvm; then
  log "LLVM 23 @ $LLVM_SHA"
  if [ -f "$LLVM_INSTALL/.llvm-sha" ] \
     && [ "$(cat "$LLVM_INSTALL/.llvm-sha")" = "$LLVM_SHA" ]; then
    echo "  already at $LLVM_SHA"
  else
    rm -rf "$LLVM_INSTALL"; mkdir -p "$LLVM_INSTALL"
    SLUG="${LLVM_REPO#*github.com[:/]}"; SLUG="${SLUG%.git}"
    URL="https://github.com/$SLUG/releases/download/llvm23-${LLVM_SHA:0:12}/llvm-install.tar.gz"
    curl -fsSL --retry 3 "$URL" \
      | tar -xz -C "$LLVM_INSTALL" --strip-components=1 \
      || { echo "no asset at $URL -- publish it from $SLUG first" >&2; exit 1; }
    echo "$LLVM_SHA" > "$LLVM_INSTALL/.llvm-sha"
  fi
  # nanobind/pybind11/PyYAML land in the interpreter that will `import mlir`.
  /usr/bin/python3.11 -m pip install -q nanobind pybind11 PyYAML
  "$LLVM_INSTALL/bin/llc" --version | sed -n 2p
  test -d "$LLVM_INSTALL/python_packages/mlir_core/mlir"
fi

# -------------------------------------------------------------------- triton --
# Its own script, because the triton image is built from it alone: see setup/triton.sh.
if want triton; then
  log "triton @ $TRITON_SHA"
  "$HERE/triton.sh" -j "$JOBS"
fi

# ---------------------------------------------------------------------- vcix --
# One translation unit against a header-only interface.
if want vcix; then
  log "vcix-accelerator model"
  build_tools
  VCIX_SRC="$HERE/../third_party/vcix-accelerator"
  test -f "$VCIX_SRC/CMakeLists.txt" \
    || { echo "third_party/vcix-accelerator is empty -- git submodule update --init" >&2; exit 1; }
  "$CMAKE" -G Ninja -S "$VCIX_SRC" -B "$VCIX_BUILD" -DCMAKE_BUILD_TYPE=Release >/dev/null
  "$CMAKE" --build "$VCIX_BUILD" --target tpu -j"$JOBS"
  test -f "$VCIX_MODEL"
fi

# ----------------------------------------------------------------------- ext --
# Everything the route adds to Triton, one plugin .so built from extension/ against the Triton
# build above, and its backend linked beside nvidia and amd.
if want ext; then
  log "torchsim-triton-ext"
  build_tools
  "$CMAKE" -G Ninja -S "$HERE/../extension" -B "$EXT_BUILD" \
    -DMLIR_DIR="$LLVM_INSTALL/lib/cmake/mlir" -DLLVM_DIR="$LLVM_INSTALL/lib/cmake/llvm" \
    -DTRITON_SRC="$TRITON_ROOT" -DTRITON_BUILD="$TRITON_BUILD" \
    -DPYBIND11_INCLUDE="$(/workspace/build-tools/bin/python -c 'import pybind11;print(pybind11.get_include())')" \
    -DPYTHON_INCLUDE="$(/usr/bin/python3.11 -c 'import sysconfig;print(sysconfig.get_paths()["include"])')" \
    -DCMAKE_SHARED_LINKER_FLAGS="-B$LLVM_INSTALL/bin -fuse-ld=lld" >/dev/null
  /workspace/build-tools/bin/ninja -C "$EXT_BUILD" -j"$JOBS"
  find "$EXT_BUILD" -name '*.o' -delete
  ln -sfnT "$(cd "$HERE/.." && pwd)/extension/python/torchsim_triton_backend" \
    "$TRITON_ROOT/python/triton/backends/torchsim"
  echo "  $EXT_BUILD/libtorchsim_triton_ext.so"
fi

log "done -- now run: pytorchsim-triton-compiler doctor"
