#!/usr/bin/env bash
# The one installer: every step the route needs, at the pins in versions.env, under $PREFIX.
# The Dockerfile's stages run these steps (one file per image: system.sh, triton.sh, togsim.sh);
# a bare box runs them all with PYTORCHSIM_PREFIX set.
#     setup/toolchain.sh [-j N] [system llvm triton vcixenv togsim vcix dialect ext torch env jaxtpu]
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
    system|llvm|triton|vcixenv|togsim|vcix|dialect|ext|torch|env|jaxtpu) STEPS+=("$1"); shift ;;
    *) echo "unknown step: $1 (system llvm triton vcixenv togsim vcix dialect ext torch env jaxtpu)" >&2; exit 2 ;;
  esac
done
[ ${#STEPS[@]} -eq 0 ] && STEPS=(llvm triton vcixenv togsim vcix dialect ext torch env)

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
want() { for s in "${STEPS[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
tools() { CMAKE=$BUILD_TOOLS/bin/cmake; [ -x "$CMAKE" ] || { echo "no $BUILD_TOOLS -- the llvm step makes it" >&2; exit 1; }; }
submodule() { test -f "$REPO/third_party/Coprocessor/$1" \
  || { echo "third_party/Coprocessor is empty -- git submodule update --init" >&2; exit 1; }; }

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
  log "Coprocessor environment @ $COPROCESSOR_ENV_ROOT"
  if [ -x "$SPIKE_BIN" ] && [ -x "$GEM5_BIN" ] && [ -f "$PK_BIN" ] && [ -x "$TOOLCHAIN_ROOT/bin/riscv64-unknown-elf-gcc" ]; then
    echo "  already laid"
  else
    submodule setup/setup.sh
    COPROCESSOR_ENV_ROOT=$COPROCESSOR_ENV_ROOT "$REPO/third_party/Coprocessor/setup/setup.sh" -j "$JOBS" toolchain pk spike gem5
  fi
fi

if want togsim; then
  log "TOGSim of third_party/TOGSim @ $(git -C "$TOGSIM_DIR" rev-parse --short HEAD 2>/dev/null || echo "the copy here")"
  "$HERE/togsim.sh" -j "$JOBS"
fi

if want vcix; then
  log "Coprocessor model"
  tools; submodule CMakeLists.txt
  "$CMAKE" -G Ninja -S "$REPO/third_party/Coprocessor" -B "$COPROCESSOR_BUILD" -DCMAKE_BUILD_TYPE=Release >/dev/null
  # The models the machine files' isa names; Coprocessor names each CMake target as its model.
  models=$(cd "$REPO" && "$PYTHON" -c 'from pytorchsim_triton_compiler.contract import target; print(*target.models())')
  [ -n "$models" ] || { echo "no machine file under $TOGSIM_DIR/configs names a pytorchsim_<model>" >&2; exit 1; }
  "$CMAKE" --build "$COPROCESSOR_BUILD" --target $models -j"$JOBS"
  for m in $models; do test -f "$COPROCESSOR_BUILD/lib$m.so"; done
fi

if want dialect; then
  log "pytorchsim dialect"
  tools
  "$PYTHON" -m pip install -q "nanobind==$NANOBIND_VERSION"
  "$CMAKE" -G Ninja -S "$REPO/pytorchsim_triton_compiler/dialects/src" -B "$DIALECT_BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DMLIR_INSTALL="$LLVM_INSTALL" -DTRITON_SRC="$TRITON_ROOT" -DPython3_EXECUTABLE="$PYTHON" >/dev/null
  "$BUILD_TOOLS/bin/ninja" -C "$DIALECT_BUILD"
fi

if want ext; then
  log "pytorchsim-triton-ext"
  tools
  "$CMAKE" -G Ninja -S "$REPO/pytorchsim_triton_compiler/triton_extension" -B "$EXT_BUILD" \
    -DMLIR_DIR="$LLVM_INSTALL/lib/cmake/mlir" -DLLVM_DIR="$LLVM_INSTALL/lib/cmake/llvm" \
    -DTRITON_SRC="$TRITON_ROOT" -DTRITON_BUILD="$TRITON_BUILD" \
    -DCMAKE_SHARED_LINKER_FLAGS="-B$LLVM_INSTALL/bin -fuse-ld=lld" >/dev/null
  "$BUILD_TOOLS/bin/ninja" -C "$EXT_BUILD" -j"$JOBS"
  find "$EXT_BUILD" -name '*.o' -delete
  ln -sfnT "$REPO/pytorchsim_triton_compiler/triton_extension/python/pytorchsim_triton_backend" "$TRITON_ROOT/python/triton/backends/pytorchsim"
  ln -sfnT "$REPO/pytorchsim_triton_compiler/triton_extension/python/language/pytorchsim" "$TRITON_ROOT/python/triton/language/extra/pytorchsim"
  echo "  $EXT_BUILD/libpytorchsim_triton_ext.so"
fi

# torch (CPU) gives the kernels their reference values and loads PyTorchSim's kernels: PyTorchSim states it, and
# validation/pytorchsim/bundle.json, the bundle it built them under, names it.
if want torch; then
  TORCH=$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["torch"])' \
    "$REPO/validation/pytorchsim/bundle.json")
  log "torch $TORCH (validation/pytorchsim/bundle.json)"
  "$PYTHON" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==$TORCH"
fi

# The compiler's defaults are the /opt/pytorchsim names; any other prefix needs these exports.
if want env; then
  log "env: $PREFIX/env.sh"
  sed -e "s#@PREFIX@#$PREFIX#g" -e "s#@REPO@#$REPO#g" -e "s#@COPROCESSOR_ENV_ROOT@#$COPROCESSOR_ENV_ROOT#g" \
      "$HERE/env.sh.in" > "$PREFIX/env.sh"
  echo "  source $PREFIX/env.sh && pytorchsim-triton-compiler doctor"
fi

# Google's TPU compiler (libtpu, ~150 MB) behind JAX, for tools/dump_tpu_vliw.py alone, in its own
# venv: jax wants a newer Python than $PYTHON. Not a default step, and no image runs it.
if want jaxtpu; then
  log "jax + libtpu @ $JAX_TPU_VENV"
  command -v "$JAX_TPU_PYTHON" >/dev/null \
    || { echo "no $JAX_TPU_PYTHON -- jax needs Python >= 3.12: set JAX_TPU_PYTHON" >&2; exit 1; }
  "$JAX_TPU_PYTHON" -m venv "$JAX_TPU_VENV"
  "$JAX_TPU_VENV/bin/python" -m pip install -q -r "$HERE/requirements-jax-tpu.txt"
  echo "  $JAX_TPU_VENV/bin/python -I $REPO/tools/dump_tpu_vliw.py -h"
fi

log "done -- now run: pytorchsim-triton-compiler doctor"
