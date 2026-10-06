#!/usr/bin/env bash
# Triton v3.7.0, unmodified, built once against the LLVM toolchain.sh's `llvm` step downloads:
# libtriton.so with its symbols exposed (TRITON_EXT_ENABLED=ON) for the extension to link to and
# load into. The triton image is this script alone, and setup/triton_tag.sh hashes it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=versions.env
source "$HERE/versions.env"
JOBS=$(nproc)
[ "${1:-}" = -j ] && JOBS="$2"

# cmake 3.31 and ninja beside the image's cmake 3.26, which TOGSim's conan build is pinned to;
# pybind11 there too, since a box's python3.11 may not have it.
BT=/workspace/build-tools
[ -x "$BT/bin/cmake" ] || { /usr/bin/python3.11 -m venv "$BT"; "$BT/bin/pip" install -q --upgrade pip; }
"$BT/bin/pip" install -q "cmake==3.31.6" ninja "pybind11==3.1.0"

mkdir -p "$TRITON_ROOT"; cd "$TRITON_ROOT"
[ -d .git ] || { git init -q; git remote add origin "$TRITON_REPO"; }
if [ "$(git rev-parse HEAD 2>/dev/null || true)" != "$TRITON_SHA" ]; then
  git fetch -q --depth 1 origin "$TRITON_SHA"; git checkout -q FETCH_HEAD
fi

# -Wno-error=attributes: GCC warns on gluon_ir.cc's pybind11 fields under EXT=ON, and the build is
# -Werror. lld: the LLVM install's, as for every link against it.
LD="-B$LLVM_INSTALL/bin -fuse-ld=lld"
"$BT/bin/cmake" -G Ninja -S "$TRITON_ROOT" -B "$TRITON_BUILD" -DCMAKE_MAKE_PROGRAM="$BT/bin/ninja" \
  -DCMAKE_BUILD_TYPE=TritonRelBuildWithAsserts \
  -DLLVM_SYSPATH="$LLVM_INSTALL" -DLLVM_DIR="$LLVM_INSTALL/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM_INSTALL/lib/cmake/mlir" -DLLD_DIR="$LLVM_INSTALL/lib/cmake/lld" \
  -DTRITON_EXT_ENABLED=ON -DCMAKE_CXX_FLAGS="-Wno-error=attributes" \
  -DTRITON_BUILD_UT=OFF -DTRITON_BUILD_PROTON=OFF -DTRITON_BUILD_PYTHON_MODULE=ON \
  -DTRITON_CODEGEN_BACKENDS="nvidia;amd" -DTRITON_WHEEL_DIR="$TRITON_BUILD/wheel" \
  -DPython3_EXECUTABLE=/usr/bin/python3.11 -Dpybind11_DIR="$("$BT/bin/python" -m pybind11 --cmakedir)" \
  -DCMAKE_EXE_LINKER_FLAGS="$LD" -DCMAKE_SHARED_LINKER_FLAGS="$LD" -DCMAKE_MODULE_LINKER_FLAGS="$LD" >/dev/null
# Every TableGen output first: libtriton.so's own target misses an edge to some of them
# (TritonGPU's OpInterfaces.h.inc), which a wide build then races past.
"$BT/bin/ninja" -C "$TRITON_BUILD" -j"$JOBS" \
  $("$BT/bin/ninja" -C "$TRITON_BUILD" -t targets all | sed -n 's/^\([^:]*\.inc\): .*/\1/p')
"$BT/bin/ninja" -C "$TRITON_BUILD" -j"$JOBS" libtriton.so
# The objects go; the headers and .inc files the extension builds against stay with libtriton.so.
find "$TRITON_BUILD" -name '*.o' -delete

# A link, not a copy: the extension links to this very file, and two copies would be two MLIRs.
ln -sfn "$TRITON_BUILD/libtriton.so" "$TRITON_ROOT/python/triton/_C/libtriton.so"
# What Triton's install step (setup.py:add_link_to_backends) links into python/ and a cmake build
# does not: the nvidia and amd backends and their language extras, which backend discovery needs.
for b in amd nvidia; do
  ln -sfnT "$TRITON_ROOT/third_party/$b/backend" "$TRITON_ROOT/python/triton/backends/$b"
  for x in "$TRITON_ROOT/third_party/$b/language"/*; do
    ln -sfnT "$x" "$TRITON_ROOT/python/triton/language/extra/$(basename "$x")"
  done
done
echo "  $TRITON_BUILD/libtriton.so"
