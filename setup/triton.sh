#!/usr/bin/env bash
# The triton image's two steps: LLVM 23 (the fork's published install, never built here) and
# triton v3.7.0 unmodified, built against it with its symbols exposed for the extension.
#     setup/triton.sh [-j N] [llvm|triton ...]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=versions.env
source "$HERE/versions.env"

JOBS=$(nproc)
STEPS=()
while [ $# -gt 0 ]; do
  case "$1" in
    -j) JOBS="$2"; shift 2 ;;
    llvm|triton) STEPS+=("$1"); shift ;;
    *) echo "unknown argument: $1 (llvm, triton)" >&2; exit 2 ;;
  esac
done
[ ${#STEPS[@]} -eq 0 ] && STEPS=(llvm triton)
want() { for s in "${STEPS[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

# cmake 3.31, ninja and pybind11 in a venv, beside the distribution's cmake 3.26 that TOGSim keeps.
[ -x "$BUILD_TOOLS/bin/cmake" ] || { "$PYTHON" -m venv "$BUILD_TOOLS"; "$BUILD_TOOLS/bin/pip" install -q --upgrade pip; }
"$BUILD_TOOLS/bin/pip" install -q "cmake==3.31.6" ninja "pybind11==3.1.0"
BT=$BUILD_TOOLS

if want llvm; then
  if [ "$(cat "$LLVM_INSTALL/.llvm-sha" 2>/dev/null || true)" = "$LLVM_SHA" ]; then
    echo "  LLVM already at $LLVM_SHA"
  else
    rm -rf "$LLVM_INSTALL"; mkdir -p "$LLVM_INSTALL"
    SLUG="${LLVM_REPO#*github.com[:/]}"; SLUG="${SLUG%.git}"
    URL="https://github.com/$SLUG/releases/download/llvm23-${LLVM_SHA:0:12}/llvm-install.tar.gz"
    curl -fsSL --retry 3 "$URL" | tar -xz -C "$LLVM_INSTALL" --strip-components=1 \
      || { echo "no asset at $URL -- publish it from $SLUG first" >&2; exit 1; }
    echo "$LLVM_SHA" > "$LLVM_INSTALL/.llvm-sha"
  fi
  "$PYTHON" -m pip install -q nanobind pybind11 PyYAML
  "$LLVM_INSTALL/bin/llc" --version | sed -n 2p
  test -d "$LLVM_INSTALL/python_packages/mlir_core/mlir"
fi

if want triton; then
  mkdir -p "$TRITON_ROOT"; cd "$TRITON_ROOT"
  [ -d .git ] || { git init -q; git remote add origin "$TRITON_REPO"; }
  if [ "$(git rev-parse HEAD 2>/dev/null || true)" != "$TRITON_SHA" ]; then
    git fetch -q --depth 1 origin "$TRITON_SHA"; git checkout -q FETCH_HEAD
  fi

  # -Wno-error=attributes: gluon_ir.cc's pybind11 fields warn under EXT=ON and the build is -Werror.
  LD="-B$LLVM_INSTALL/bin -fuse-ld=lld"
  "$BT/bin/cmake" -G Ninja -S "$TRITON_ROOT" -B "$TRITON_BUILD" -DCMAKE_MAKE_PROGRAM="$BT/bin/ninja" \
    -DCMAKE_BUILD_TYPE=TritonRelBuildWithAsserts \
    -DLLVM_SYSPATH="$LLVM_INSTALL" -DLLVM_DIR="$LLVM_INSTALL/lib/cmake/llvm" \
    -DMLIR_DIR="$LLVM_INSTALL/lib/cmake/mlir" -DLLD_DIR="$LLVM_INSTALL/lib/cmake/lld" \
    -DTRITON_EXT_ENABLED=ON -DCMAKE_CXX_FLAGS="-Wno-error=attributes" \
    -DTRITON_BUILD_UT=OFF -DTRITON_BUILD_PROTON=OFF -DTRITON_BUILD_PYTHON_MODULE=ON \
    -DTRITON_CODEGEN_BACKENDS="nvidia;amd" -DTRITON_WHEEL_DIR="$TRITON_BUILD/wheel" \
    -DPython3_EXECUTABLE="$PYTHON" -Dpybind11_DIR="$("$BT/bin/python" -m pybind11 --cmakedir)" \
    -DCMAKE_EXE_LINKER_FLAGS="$LD" -DCMAKE_SHARED_LINKER_FLAGS="$LD" -DCMAKE_MODULE_LINKER_FLAGS="$LD" >/dev/null
  # TableGen outputs first: libtriton.so misses an edge to some of them and a wide build races past.
  "$BT/bin/ninja" -C "$TRITON_BUILD" -j"$JOBS" \
    $("$BT/bin/ninja" -C "$TRITON_BUILD" -t targets all | sed -n 's/^\([^:]*\.inc\): .*/\1/p')
  "$BT/bin/ninja" -C "$TRITON_BUILD" -j"$JOBS" libtriton.so
  find "$TRITON_BUILD" -name '*.o' -delete

  # A link, not a copy: the extension links to this very file, and two copies would be two MLIRs.
  ln -sfn "$TRITON_BUILD/libtriton.so" "$TRITON_ROOT/python/triton/_C/libtriton.so"
  for b in amd nvidia; do
    ln -sfnT "$TRITON_ROOT/third_party/$b/backend" "$TRITON_ROOT/python/triton/backends/$b"
    for x in "$TRITON_ROOT/third_party/$b/language"/*; do
      ln -sfnT "$x" "$TRITON_ROOT/python/triton/language/extra/$(basename "$x")"
    done
  done
  echo "  $TRITON_BUILD/libtriton.so"
fi
