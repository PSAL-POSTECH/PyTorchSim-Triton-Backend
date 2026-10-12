#!/usr/bin/env bash
# The one installer: every step the route needs, at the pins in versions.env, under $PREFIX.
# The Dockerfile's stages run these steps (one file per image: system.sh, triton.sh, togsim.sh);
# a bare box runs them all with TORCHSIM_PREFIX set.
#     setup/toolchain.sh [-j N] [system llvm triton vcixenv togsim vcix dialect ext env]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
# shellcheck source=versions.env
source "$HERE/versions.env"

JOBS=$(( $(nproc) > 4 ? $(nproc) - 4 : 1 ))
STEPS=()
while [ $# -gt 0 ]; do
  case "$1" in
    -j) JOBS="$2"; shift 2 ;;
    -h|--help) sed -n '2,5p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    system|llvm|triton|vcixenv|togsim|vcix|dialect|ext|env) STEPS+=("$1"); shift ;;
    *) echo "unknown step: $1 (system llvm triton vcixenv togsim vcix dialect ext env)" >&2; exit 2 ;;
  esac
done
[ ${#STEPS[@]} -eq 0 ] && STEPS=(llvm triton vcixenv togsim vcix dialect ext env)

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
want() { for s in "${STEPS[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
tools() { CMAKE=$BUILD_TOOLS/bin/cmake; [ -x "$CMAKE" ] || { echo "no $BUILD_TOOLS -- the llvm step makes it" >&2; exit 1; }; }
submodule() { test -f "$REPO/third_party/vcix-accelerator/$1" \
  || { echo "third_party/vcix-accelerator is empty -- git submodule update --init" >&2; exit 1; }; }

# The distribution layer, as root. Not a default step.
if want system; then
  log "distribution layer (root)"
  [ "$(id -u)" = 0 ] || { echo "the system step installs packages: run it as root" >&2; exit 1; }
  "$HERE/system.sh"
fi

if want llvm; then
  log "LLVM 23 @ $LLVM_SHA"
  "$HERE/triton.sh" -j "$JOBS" llvm
fi

if want triton; then
  log "triton @ $TRITON_SHA"
  "$HERE/triton.sh" -j "$JOBS" triton
fi

# spike, gem5, the RISC-V toolchain and pk: the submodule's own setup.sh, unless the tree is there already.
if want vcixenv; then
  log "vcix-accelerator environment @ $VCIX_ENV_ROOT"
  if [ -x "$SPIKE_BIN" ] && [ -x "$GEM5_BIN" ] && [ -f "$PK_BIN" ] && [ -x "$TOOLCHAIN_ROOT/bin/riscv64-unknown-elf-gcc" ]; then
    echo "  already laid"
  else
    submodule setup/setup.sh
    VCIX_ENV_ROOT=$VCIX_ENV_ROOT "$REPO/third_party/vcix-accelerator/setup/setup.sh" -j "$JOBS" toolchain pk spike gem5
  fi
fi

if want togsim; then
  log "TOGSim of third_party/TOGSim @ $(git -C "$TOGSIM_DIR" rev-parse --short HEAD 2>/dev/null || echo "the copy here")"
  "$HERE/togsim.sh" -j "$JOBS"
fi

if want vcix; then
  log "vcix-accelerator model"
  tools; submodule CMakeLists.txt
  "$CMAKE" -G Ninja -S "$REPO/third_party/vcix-accelerator" -B "$VCIX_BUILD" -DCMAKE_BUILD_TYPE=Release >/dev/null
  "$CMAKE" --build "$VCIX_BUILD" --target tpu -j"$JOBS"
  test -f "$VCIX_MODEL"
fi

if want dialect; then
  log "torchsim dialect"
  tools
  "$PYTHON" -m pip install -q "nanobind==$NANOBIND_VERSION"
  "$CMAKE" -G Ninja -S "$REPO/pytorchsim_triton_compiler/dialects/src" -B "$DIALECT_BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DMLIR_INSTALL="$LLVM_INSTALL" -DPython3_EXECUTABLE="$PYTHON" >/dev/null
  "$BUILD_TOOLS/bin/ninja" -C "$DIALECT_BUILD"
fi

if want ext; then
  log "torchsim-triton-ext"
  tools
  "$CMAKE" -G Ninja -S "$REPO/pytorchsim_triton_compiler/triton_extension" -B "$EXT_BUILD" \
    -DMLIR_DIR="$LLVM_INSTALL/lib/cmake/mlir" -DLLVM_DIR="$LLVM_INSTALL/lib/cmake/llvm" \
    -DTRITON_SRC="$TRITON_ROOT" -DTRITON_BUILD="$TRITON_BUILD" \
    -DCMAKE_SHARED_LINKER_FLAGS="-B$LLVM_INSTALL/bin -fuse-ld=lld" >/dev/null
  "$BUILD_TOOLS/bin/ninja" -C "$EXT_BUILD" -j"$JOBS"
  find "$EXT_BUILD" -name '*.o' -delete
  ln -sfnT "$REPO/pytorchsim_triton_compiler/triton_extension/python/torchsim_triton_backend" "$TRITON_ROOT/python/triton/backends/torchsim"
  ln -sfnT "$REPO/pytorchsim_triton_compiler/triton_extension/python/language/torchsim" "$TRITON_ROOT/python/triton/language/extra/torchsim"
  echo "  $EXT_BUILD/libtorchsim_triton_ext.so"
fi

# The compiler's defaults are the /workspace names; any other prefix needs these exports.
if want env; then
  log "env: $PREFIX/env.sh"
  sed -e "s#@PREFIX@#$PREFIX#g" -e "s#@REPO@#$REPO#g" -e "s#@VCIX_ENV_ROOT@#$VCIX_ENV_ROOT#g" \
      "$HERE/env.sh.in" > "$PREFIX/env.sh"
  echo "  source $PREFIX/env.sh && pytorchsim-triton-compiler doctor"
fi

log "done -- now run: pytorchsim-triton-compiler doctor"
