#!/usr/bin/env bash
# Everything this harness needs that the distribution does not give it.
# setup/system.sh lays the distribution layer first; this lays the rest.
#
#     ./setup/restore.sh                 every step
#     ./setup/restore.sh spike llvm      only those
#     ./setup/restore.sh -j 8 triton     cap build parallelism
#
# Then: pytorchsim-triton-compiler doctor
#
# Steps: riscv (bare-metal gcc + pk), spike, llvm (downloaded), triton (v3.7.0, built once),
# vcix, gem5 (downloaded), ext (extension/, against the triton build).
# Every step is idempotent -- rerunning a finished one is a no-op.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=versions.env
source "$HERE/versions.env"

JOBS=$(( $(nproc) > 4 ? $(nproc) - 4 : 1 ))
STEPS=()

while [ $# -gt 0 ]; do
  case "$1" in
    -j) JOBS="$2"; shift 2 ;;
    -h|--help) sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *) STEPS+=("$1"); shift ;;
  esac
done
[ ${#STEPS[@]} -eq 0 ] && STEPS=(riscv spike llvm triton vcix gem5 ext)

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


# --------------------------------------------------------------------- riscv --
# The bare-metal gcc every ELF here is linked with, and the pk Spike boots. THE BASE IMAGE
# USED TO CARRY BOTH and ubuntu:22.04 does not, so they are restored here now. The pins are
# the submodule's, not a second copy: third_party/vcix-accelerator/setup/ already names
# riscv-collab's release tarball and riscv-pk's commit, and that setup and this one have to
# be the same toolchain -- the model Spike loads is built by one and the ELF by the other.
if want riscv; then
  log "RISC-V toolchain + pk"
  if [ -x "$RISCV_ROOT/bin/riscv64-unknown-elf-gcc" ] \
     && [ "$(cat "$RISCV_ROOT/.toolchain-sha256" 2>/dev/null || true)" = "$TOOLCHAIN_SHA256" ]; then
    echo "  already at ${TOOLCHAIN_SHA256:0:12}"
  else
    rm -rf "$RISCV_ROOT"; mkdir -p "$RISCV_ROOT"
    curl -fsSL --retry 3 -o /tmp/riscv-toolchain.tar.gz "$TOOLCHAIN_URL"
    echo "$TOOLCHAIN_SHA256  /tmp/riscv-toolchain.tar.gz" | sha256sum -c -
    tar -xzf /tmp/riscv-toolchain.tar.gz -C "$RISCV_ROOT" \
        --strip-components=1 --no-same-owner
    rm -f /tmp/riscv-toolchain.tar.gz
    # share/ is man pages and locale, and the host binaries carry debug info nobody here
    # steps through: together they are more than half of what the tarball unpacks to.
    rm -rf "$RISCV_ROOT/share"
    find "$RISCV_ROOT/bin" "$RISCV_ROOT/libexec" -type f \
      -exec sh -c 'for f; do strip --strip-debug "$f" 2>/dev/null || true; done' _ {} +
    echo "$TOOLCHAIN_SHA256" > "$RISCV_ROOT/.toolchain-sha256"
  fi
  "$RISCV_ROOT/bin/riscv64-unknown-elf-gcc" --version | head -1

  # Built, not installed: TORCHSIM_COMPILE_PK names the build tree, which is where the base
  # image left it too.
  if [ ! -f "$PK_BIN" ]; then
    fetch_commit "$PK_ROOT" "$PK_REPO" "$PK_SHA"
    mkdir -p "$PK_ROOT/build"; cd "$PK_ROOT/build"
    export PATH="$RISCV_ROOT/bin:$PATH"
    [ -f Makefile ] || ../configure --prefix="$RISCV_ROOT" --host=riscv64-unknown-elf
    make -j"$JOBS"
  fi
  test -f "$PK_BIN" || { echo "pk did not land at $PK_BIN" >&2; exit 1; }
  echo "  $PK_BIN"
fi

# --------------------------------------------------------------------- spike --
if want spike; then
  log "spike (the fork with the vcix adapter and no accelerator of its own)"
  fetch_commit "$SPIKE_ROOT" "$SPIKE_REPO" "$SPIKE_SHA"
  test -f "$SPIKE_ROOT/riscv/vcix_accel_extension.cc" \
    || { echo "vcix_accel_extension.cc missing -- wrong branch?" >&2; exit 1; }
  mkdir -p "$SPIKE_ROOT/build"; cd "$SPIKE_ROOT/build"
  [ -f Makefile ] || ../configure --prefix="$SPIKE_ROOT/install"
  make -j"$JOBS"
  make install
  # head closes the pipe early, which is SIGPIPE under pipefail.
  printf '%s\n' "$("$SPIKE_ROOT/install/bin/spike" --help 2>&1)" | head -1
fi

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

# ---------------------------------------------------------------------- gem5 --
# Downloaded, not built: the release vcix-accelerator pins, whose VCIX path is
# generic and whose accelerator is the same $VCIX_MODEL Spike loads.
if want gem5; then
  log "gem5 (the release vcix-accelerator pins)"
  if [ -x "$GEM5_ROOT/$GEM5_TARGET" ] && [ -f "$GEM5_ROOT/.gem5-sha256" ] \
     && [ "$(cat "$GEM5_ROOT/.gem5-sha256")" = "$GEM5_SHA256" ]; then
    echo "  already at ${GEM5_SHA256:0:12}"
  else
    rm -rf "$GEM5_ROOT"; mkdir -p "$GEM5_ROOT"
    curl -fsSL --retry 3 -o /tmp/gem5.tar.gz "$GEM5_URL"
    echo "$GEM5_SHA256  /tmp/gem5.tar.gz" | sha256sum -c -
    tar -xzf /tmp/gem5.tar.gz -C "$GEM5_ROOT"; rm -f /tmp/gem5.tar.gz
    echo "$GEM5_SHA256" > "$GEM5_ROOT/.gem5-sha256"
  fi
  "$GEM5_ROOT/$GEM5_TARGET" --help >/dev/null
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
