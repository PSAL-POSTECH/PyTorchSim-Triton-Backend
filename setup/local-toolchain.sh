#!/usr/bin/env bash
# Build the whole Triton route on LOCAL disk from the LATEST commit of one branch name.
#
#     ./setup/local-toolchain.sh                     every step, branch adapt-ttgir
#     ./setup/local-toolchain.sh --branch my-round   another branch name, same in every repo
#     ./setup/local-toolchain.sh triton gem5         only those steps (after a pull)
#     ./setup/local-toolchain.sh --root /scratch/ts  somewhere other than /opt/torchsim-local
#
# Then: source <root>/env.sh && pytorchsim-triton-compiler doctor
#
# Needs GITHUB_TOKEN (or ~/.gh-token) with read access to the private repos.
# Steps: sources llvm riscv pk spike vcix togsim gem5 triton dialect ext python env
set -euo pipefail

ROOT=/opt/torchsim-local
BRANCH=adapt-ttgir
JOBS=$(nproc)
STEPS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    -j) JOBS="$2"; shift 2 ;;
    -h|--help) sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *) STEPS+=("$1"); shift ;;
  esac
done
[ ${#STEPS[@]} -eq 0 ] && STEPS=(sources llvm riscv pk spike vcix togsim gem5 triton dialect ext python env)
want() { local s; for s in "${STEPS[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# One branch name across every repo of the round; a repo can be pointed elsewhere
# by its own variable (e.g. GEM5_BRANCH=TorchSim).
REPOS=(
  "pytorchsim-triton-compiler PSAL-POSTECH/PyTorchSim-Triton-Backend ${PSTO_BRANCH:-main}"
  "triton                     PSAL-POSTECH/Triton                ${TRITON_BRANCH:-PyTorchSim}"
  "gem5                       student-Jungmin/gem5               ${GEM5_BRANCH:-$BRANCH}"
  "PyTorchSim                 student-Jungmin/PyTorchSim         ${PYTORCHSIM_BRANCH:-$BRANCH}"
)
# Triton is upstream's release, unmodified (its PyTorchSim branch is v3.7.0): everything the route
# adds to it is the torchsim extension, extension/ in the compiler's own repository.
# Spike is taken at the pin third_party/vcix-accelerator names (the accelerator is the
# model it loads, $VCIX_MODEL), as setup/restore.sh does.
# riscv-pk is upstream's, not ours, so it is NOT taken at its latest: master's
# head hangs every kernel inside spike. The pin is the one Dockerfile.base builds.
# LLVM is a release asset, not a build: the newest release whose tag names this
# branch's head (llvm23-<sha12>), published by that repo's CI.
LLVM_REPO=PSAL-POSTECH/llvm-project
LLVM_BRANCH=${LLVM_BRANCH:-torchsim-llvm23}
# Build-tool versions: setup/restore.sh's cmake, SCons that gem5's SConstruct
# still reads (4.6+ breaks its configure links), nanobind from Dockerfile.base.
CMAKE_VERSION=3.31.6
PYBIND11_VERSION=3.1.0
SCONS_VERSION=4.5.2
NANOBIND_SPEC="nanobind>=2.9,<3.0"
RISCV_URL=https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2023.12.14/riscv64-elf-ubuntu-20.04-llvm-nightly-2023.12.14-nightly.tar.gz

TOKEN=${GITHUB_TOKEN:-$(cat ~/.gh-token 2>/dev/null || true)}
[ -n "$TOKEN" ] || { echo "GITHUB_TOKEN (or ~/.gh-token) is needed: three of the repos are private" >&2; exit 1; }
# Basic auth in a header: the token never lands in .git/config or in a URL.
GIT_AUTH="http.extraHeader=Authorization: Basic $(printf 'x-access-token:%s' "$TOKEN" | base64 -w0)"
api() { curl -fsSL -H "Authorization: Bearer $TOKEN" "$@"; }

SRC=$ROOT/src; BUILD=$ROOT/build; TC=$ROOT/toolchain; LOGS=$ROOT/logs; BT=$ROOT/build-tools
LLVM=$ROOT/llvm23/install
mkdir -p "$SRC" "$BUILD" "$TC" "$LOGS"

# ------------------------------------------------------------------- sources --
# Latest commit of the branch. A clone with local changes is left alone rather
# than reset: that is somebody's work.
if want sources; then
  log "sources @ latest $BRANCH"
  for r in "${REPOS[@]}"; do
    read -r name slug br <<<"$r"
    d=$SRC/$name
    if [ ! -d "$d/.git" ]; then
      git -c "$GIT_AUTH" clone -q --filter=blob:none --branch "$br" "https://github.com/$slug.git" "$d"
    elif [ -n "$(git -C "$d" status --porcelain -uno)" ]; then
      echo "  $name has local changes -- not updated" >&2
    else
      git -C "$d" -c "$GIT_AUTH" fetch -q origin "$br"
      git -C "$d" checkout -q -B "$br" FETCH_HEAD
    fi
    printf '  %-28s %-24s %s\n' "$name" "$br" "$(git -C "$d" log -1 --format='%h %cs %<(60,trunc)%s')"
  done
  git -C "$SRC/pytorchsim-triton-compiler" -c "$GIT_AUTH" submodule update -q --init third_party/vcix-accelerator
fi
VCIX_SRC=$SRC/pytorchsim-triton-compiler/third_party/vcix-accelerator

if want sources || want pk; then
  pk_ref=$(sed -n 's/.*riscv-pk.*git checkout \([0-9a-f]\{40\}\).*/\1/p' "$SRC/PyTorchSim/Dockerfile.base" | head -1)
  [ -n "$pk_ref" ] || { echo "no riscv-pk pin found in PyTorchSim/Dockerfile.base" >&2; exit 1; }
  [ -d "$SRC/riscv-pk/.git" ] || git clone -q --filter=blob:none https://github.com/riscv-software-src/riscv-pk.git "$SRC/riscv-pk"
  if [ "$(git -C "$SRC/riscv-pk" rev-parse HEAD)" != "$pk_ref" ]; then
    git -C "$SRC/riscv-pk" fetch -q origin "$pk_ref"; git -C "$SRC/riscv-pk" checkout -q "$pk_ref"; rm -rf "$SRC/riscv-pk/build"
  fi
  printf '  %-28s %-24s %s\n' riscv-pk "(Dockerfile.base pin)" "${pk_ref:0:7}"
fi

# ---------------------------------------------------------------- build tools --
# Its own venv, so the image's cmake 3.26 (pinned for TOGSim's conan) is left alone.
if [ ! -x "$BT/bin/scons" ]; then
  log "build tools venv"
  python3.11 -m venv "$BT"
  "$BT/bin/pip" install -q "cmake==$CMAKE_VERSION" ninja "pybind11==$PYBIND11_VERSION" "scons==$SCONS_VERSION"
fi

# ---------------------------------------------------------------------- llvm --
if want llvm; then
  head=$(api "https://api.github.com/repos/$LLVM_REPO/git/ref/heads/$LLVM_BRANCH" | python3 -c 'import sys,json;print(json.load(sys.stdin)["object"]["sha"])')
  tag=llvm23-${head:0:12}
  if [ "$(cat "$ROOT/llvm23/TAG" 2>/dev/null)" = "$tag" ]; then
    log "llvm: $tag already here"
  else
    log "llvm: $LLVM_REPO $LLVM_BRANCH head $tag"
    asset=$(api "https://api.github.com/repos/$LLVM_REPO/releases/tags/$tag" 2>/dev/null \
      | python3 -c 'import sys,json;print(next(a["id"] for a in json.load(sys.stdin)["assets"] if a["name"]=="llvm-install.tar.gz"))' 2>/dev/null || true)
    [ -n "$asset" ] || { echo "no release $tag with llvm-install.tar.gz -- has $LLVM_REPO's CI published the head yet?" >&2; exit 1; }
    rm -rf "$ROOT/llvm23"; mkdir -p "$ROOT/llvm23"
    api -H "Accept: application/octet-stream" "https://api.github.com/repos/$LLVM_REPO/releases/assets/$asset" \
      | tar -xz -C "$ROOT/llvm23"
    echo "$tag" > "$ROOT/llvm23/TAG"
  fi
  "$LLVM/bin/mlir-opt" --version | sed -n 2p
fi

# Each build below is independent of the others; run them side by side and wait.
pids=()
run() {  # run <name> <function>: background, logged, remembered
  local name=$1; shift
  ( set -euo pipefail; s=$(date +%s); "$@"; echo "$name ok $(( $(date +%s)-s ))s" ) > "$LOGS/$name.log" 2>&1 &
  pids+=("$!:$name")
}
share=$(( JOBS / 2 > 1 ? JOBS / 2 : 1 ))

build_riscv() {
  [ -x "$TC/riscv/bin/riscv64-unknown-elf-gcc" ] && return 0
  curl -fsSL --retry 3 "$RISCV_URL" | tar -xz -C "$TC"
  rm -rf "$TC/riscv/share"
}
build_spike() {
  local repo sha d=$SRC/riscv-isa-sim
  eval "$(cd "$VCIX_SRC" && . setup/versions.env >/dev/null 2>&1 && declare -p SPIKE_REPO SPIKE_SHA)"
  repo=$SPIKE_REPO sha=$SPIKE_SHA
  [ -d "$d/.git" ] || git init -q "$d"
  git -C "$d" remote remove origin 2>/dev/null || true
  git -C "$d" remote add origin "$repo"
  if [ "$(git -C "$d" rev-parse -q --verify HEAD || true)" != "$sha" ]; then
    git -C "$d" -c "$GIT_AUTH" fetch -q --depth 1 origin "$sha"
    git -C "$d" checkout -q FETCH_HEAD; rm -rf "$d/build"
  fi
  test -f "$d/riscv/vcix_accel_extension.cc" || { echo "spike at $sha has no vcix adapter" >&2; return 1; }
  mkdir -p "$d/build"; cd "$d/build"
  [ -f Makefile ] || ../configure --prefix="$TC/spike"
  make -j"$share" && make install
}
build_vcix() {
  "$BT/bin/cmake" -G Ninja -S "$VCIX_SRC" -B "$BUILD/vcix" -DCMAKE_BUILD_TYPE=Release >/dev/null
  "$BT/bin/cmake" --build "$BUILD/vcix" --target tpu -j"$share"
  test -f "$BUILD/vcix/libtpu.so"
}
build_togsim() {
  git -C "$SRC/PyTorchSim" submodule update --init --recursive --depth 1 --jobs 8
  mkdir -p "$SRC/PyTorchSim/TOGSim/build"; cd "$SRC/PyTorchSim/TOGSim/build"
  conan install .. --build=missing && cmake .. && make -j"$share"
}
build_gem5() {
  cd "$SRC/gem5"
  # python3-config may name 3.10 on this image; gem5 must embed the 3.11 the rest runs.
  PATH=$BT/bin:$PATH PYTHON_CONFIG=python3.11-config scons build/RISCV/gem5.opt -j"$share" --ignore-style
}
build_triton() {
  # Unmodified, built once: libtriton.so with its symbols exposed (EXT=ON) for the extension to
  # link to and load into. -Wno-error=attributes: GCC warns on gluon_ir.cc's pybind11 fields
  # under EXT=ON, and the build is -Werror.
  local B=$BUILD/triton-host LD="-B$LLVM/bin -fuse-ld=lld"
  "$BT/bin/cmake" -G Ninja -S "$SRC/triton" -B "$B" \
    -DCMAKE_BUILD_TYPE=TritonRelBuildWithAsserts \
    -DLLVM_SYSPATH="$LLVM" -DLLVM_DIR="$LLVM/lib/cmake/llvm" -DMLIR_DIR="$LLVM/lib/cmake/mlir" -DLLD_DIR="$LLVM/lib/cmake/lld" \
    -DTRITON_EXT_ENABLED=ON -DCMAKE_CXX_FLAGS="-Wno-error=attributes" \
    -DTRITON_BUILD_UT=OFF -DTRITON_BUILD_PROTON=OFF -DTRITON_BUILD_PYTHON_MODULE=ON \
    -DTRITON_CODEGEN_BACKENDS="nvidia;amd" -DTRITON_WHEEL_DIR="$B/wheel" \
    -DPython3_EXECUTABLE="$BT/bin/python3.11" -Dpybind11_DIR="$("$BT/bin/python3.11" -m pybind11 --cmakedir)" \
    -DCMAKE_EXE_LINKER_FLAGS="$LD" -DCMAKE_SHARED_LINKER_FLAGS="$LD" -DCMAKE_MODULE_LINKER_FLAGS="$LD"
  "$BT/bin/ninja" -C "$B" -j"$share" libtriton.so
  # A link, not a copy: the extension links to this very file, and two copies would be two MLIRs.
  ln -sfn "$B/libtriton.so" "$SRC/triton/python/triton/_C/libtriton.so"
}
build_dialect() {
  python3.11 -c 'import nanobind' 2>/dev/null || python3.11 -m pip install -q --root-user-action=ignore "$NANOBIND_SPEC"
  local B=$BUILD/dialect
  "$BT/bin/cmake" -G Ninja -S "$SRC/pytorchsim-triton-compiler/dialect" -B "$B" -DCMAKE_BUILD_TYPE=Release \
    -DMLIR_INSTALL="$LLVM" -DLLVM_TBLGEN="$LLVM/bin/llvm-tblgen" -DMLIR_TBLGEN="$LLVM/bin/mlir-tblgen" \
    -DMLIR_PY_CAPI="$LLVM/python_packages/mlir_core/mlir/_mlir_libs/libMLIRPythonCAPI.so" \
    -Dnanobind_DIR="$(python3.11 -c 'import nanobind;print(nanobind.cmake_dir())')" -DPython_EXECUTABLE=/usr/bin/python3.11
  "$BT/bin/ninja" -C "$B"
  rm -rf "$TC/torchsim-dialect"; mkdir -p "$TC/torchsim-dialect"
  cp -a "$B"/_torchsim_*.py "$B"/_torchsim_ext*.so "$B/TorchSimOps.json" "$TC/torchsim-dialect/"
}

want riscv   && run riscv   build_riscv
want spike   && run spike   build_spike
want vcix    && run vcix    build_vcix
want togsim  && run togsim  build_togsim
want gem5    && run gem5    build_gem5
want triton  && run triton  build_triton
want dialect && run dialect build_dialect
# pk is built by the cross compiler, so it waits for riscv
if want pk; then
  for p in "${pids[@]}"; do [ "${p#*:}" = riscv ] && wait "${p%%:*}"; done
  ( set -e; cd "$SRC/riscv-pk"; mkdir -p build; cd build; export PATH=$TC/riscv/bin:$PATH
    [ -f Makefile ] || CC=riscv64-unknown-elf-gcc ../configure --host=riscv64-unknown-elf
    make -j"$share"; cp pk "$TC/pk" ) > "$LOGS/pk.log" 2>&1 && echo "  pk ok" || { echo "pk failed: $LOGS/pk.log" >&2; exit 1; }
fi
fail=0
for p in "${pids[@]}"; do
  if wait "${p%%:*}"; then echo "  $(tail -1 "$LOGS/${p#*:}.log")"
  else echo "  ${p#*:} FAILED -- $LOGS/${p#*:}.log" >&2; fail=1; fi
done
[ "$fail" = 0 ] || exit 1

# ----------------------------------------------------------------------- ext --
# Everything the route adds to Triton, one plugin .so: built against the Triton build tree above,
# triton_shared's parts and the TorchSim dialect compiled in from this repository.
if want ext; then
  log "torchsim-triton-ext"
  "$BT/bin/cmake" -G Ninja -S "$SRC/pytorchsim-triton-compiler/extension" -B "$BUILD/torchsim-triton-ext" \
    -DMLIR_DIR="$LLVM/lib/cmake/mlir" -DLLVM_DIR="$LLVM/lib/cmake/llvm" \
    -DTRITON_SRC="$SRC/triton" -DTRITON_BUILD="$BUILD/triton-host" \
    -DPYBIND11_INCLUDE="$("$BT/bin/python3.11" -c 'import pybind11;print(pybind11.get_include())')" \
    -DPYTHON_INCLUDE="$(python3.11 -c 'import sysconfig;print(sysconfig.get_paths()["include"])')" \
    -DCMAKE_SHARED_LINKER_FLAGS="-B$LLVM/bin -fuse-ld=lld" > "$LOGS/ext.log" 2>&1
  "$BT/bin/ninja" -C "$BUILD/torchsim-triton-ext" -j"$JOBS" >> "$LOGS/ext.log" 2>&1 \
    || { echo "ext failed: $LOGS/ext.log" >&2; exit 1; }
  echo "  $BUILD/torchsim-triton-ext/libtorchsim_triton_ext.so"
fi

# -------------------------------------------------------------------- python --
# What Triton's own install step links into its python/ tree, and the extension's backend beside
# its nvidia and amd ones. The box's installed triton is left alone: the compiler's workers put
# this tree first on their path and find backends in it alone.
if want python; then
  log "python: triton's backends and language extras, and the torchsim backend"
  T=$SRC/triton
  for b in amd nvidia; do
    ln -sfnT "$T/third_party/$b/backend" "$T/python/triton/backends/$b"
    for x in "$T/third_party/$b/language"/*; do [ -e "$x" ] && ln -sfnT "$x" "$T/python/triton/language/extra/$(basename "$x")"; done
  done
  ln -sfnT "$SRC/pytorchsim-triton-compiler/extension/python/torchsim_triton_backend" "$T/python/triton/backends/torchsim"
fi

# ----------------------------------------------------------------------- env --
if want env; then
  log "env: $ROOT/env.sh"
  sed "s#@ROOT@#$ROOT#" "$(dirname "${BASH_SOURCE[0]}")/local-env.sh.in" > "$ROOT/env.sh"
  echo "  source $ROOT/env.sh && pytorchsim-triton-compiler doctor"
fi
