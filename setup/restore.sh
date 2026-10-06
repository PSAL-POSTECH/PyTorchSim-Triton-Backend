#!/usr/bin/env bash
# Everything this harness needs that the distribution does not give it.
# setup/system.sh lays the distribution layer first; this lays the rest.
#
#     ./setup/restore.sh                 every step
#     ./setup/restore.sh spike llvm      only those
#     ./setup/restore.sh -j 8 tritonenv  cap build parallelism
#
# Then: pytorchsim-triton-compiler doctor
#
# Steps: riscv (bare-metal gcc + pk), spike, layout (triton sources),
# tritonshared, llvm (downloaded), tritonenv, vcix, gem5 (downloaded).
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
[ ${#STEPS[@]} -eq 0 ] && STEPS=(riscv spike layout tritonshared llvm tritonenv vcix gem5)

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
want() { for s in "${STEPS[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

# `python/triton/backends/<name>` are symlinks the triton build creates
# (setup.py:add_link_to_backends), so they are in neither git nor the tarballs.
# Without them `import triton` fails in entry-point discovery.
link_triton_backends() {
  local T="$TRITON_ROOT" b x
  [ -d "$T/python/triton/backends" ] || return 0
  for b in amd nvidia; do
    [ -d "$T/third_party/$b/backend" ] || continue
    ln -sfnT "$T/third_party/$b/backend" "$T/python/triton/backends/$b"
    [ -d "$T/third_party/$b/language" ] || continue
    for x in "$T/third_party/$b/language"/*; do
      ln -sfnT "$x" "$T/python/triton/language/extra/$(basename "$x")"
    done
  done
  # Called before the tritonshared step too, when the checkout is not there yet;
  # that step calls this again once it is.
  local TS="$TRITON_SHARED_ROOT/backend"
  if [ -d "$TS" ]; then
    ln -sfnT "$TS" "$T/python/triton/backends/triton_shared"
  fi
  echo "  triton backend links: $(ls "$T/python/triton/backends" | tr '\n' ' ')"
}

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

# -------------------------------------------------------------------- layout --
# pytorchsim_triton_compiler/config.py's defaults point at /workspace/triton-src and
# /workspace/triton_shared, and the coverage kernels' docstrings cite line
# numbers relative to the latter. Reproducing that layout means zero env vars
# and keeps those references valid.
if want layout; then
  log "triton sources"
  fetch_commit "$TRITON_ROOT" "$TRITON_REPO" "$TRITON_SHA"
  cd "$TRITON_ROOT"
  for p in nvvm_reduction_kind_compatibility libdevice_sigmoid global_access_any_layout; do
    pf="$HERE/patches/triton/$p.patch"
    if GIT_CONFIG_NOSYSTEM=1 git apply --reverse --check "$pf" 2>/dev/null; then
      echo "  patch already applied: $p"
    else
      GIT_CONFIG_NOSYSTEM=1 git apply "$pf"
      echo "  patch applied: $p"
    fi
  done

  link_triton_backends
fi

# ---------------------------------------------------------- triton_shared --
if want tritonshared; then
  log "triton_shared @ $TRITON_SHARED_BRANCH"
  # SSH is what a developer authenticates with; a container has only the token.
  TS_URL="$TRITON_SHARED_REPO"
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    TS_SLUG="${TRITON_SHARED_REPO#*github.com[:/]}"; TS_SLUG="${TS_SLUG%.git}"
    TS_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${TS_SLUG}.git"
  fi
  if [ ! -d "$TRITON_SHARED_ROOT/.git" ]; then
    git clone "$TS_URL" "$TRITON_SHARED_ROOT"
  fi
  cd "$TRITON_SHARED_ROOT"
  # SET THE URL EVERY TIME, not just on the clone. An existing checkout keeps
  # whatever remote it was cloned with, so a tree from before the pin moved to
  # PSAL-POSTECH would still be fetching student-Jungmin's fork and would
  # silently never see a commit pushed to ours.
  git remote set-url origin "$TS_URL"
  git fetch -q origin "$TRITON_SHARED_BRANCH"
  git checkout -q "$TRITON_SHARED_SHA"
  # Never leave the token in .git/config -- this tree ships inside the image.
  git remote set-url origin "$TRITON_SHARED_REPO"
  # No patch loop: every fix is a commit in the pin. See versions.env for the
  # ancestry and for why the loop went away.
  if [ "$(git rev-parse HEAD)" != "$TRITON_SHARED_SHA" ]; then
    echo "triton_shared is not at $TRITON_SHARED_SHA" >&2; exit 1
  fi
  echo "  $PWD @ $(git rev-parse HEAD)"
  link_triton_backends
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

# ----------------------------------------------------------------- tritonenv --
# triton-shared-opt cannot be built standalone: tools/triton-shared-opt/
# CMakeLists.txt links ${TRITON_LIBS} and TritonTransforms, which only exist
# inside a triton build tree. TRITON_PLUGIN_DIRS is how triton picks it up.
# Only triton_shared is listed -- qcom_hexagon_backend would drag in the
# Hexagon SDK, and this harness does not use it.
if want tritonenv; then
  log "triton venv + triton-shared-opt"
  if [ ! -x "$TRITON_VENV/bin/python" ]; then
    /usr/bin/python3.11 -m venv "$TRITON_VENV"
  fi
  "$TRITON_VENV/bin/pip" install -q --upgrade pip setuptools wheel
  # lit IS A BUILD INPUT, not a test-time one. triton's setup.py does
  # `lit_dir = shutil.which('lit')` and only passes -DLLVM_EXTERNAL_LIT when it
  # finds one; miss it and cmake bakes an empty path into every check-* target,
  # so `ninja check-triton-shared-lit-tests` runs `python /llvm-lit` and dies on
  # a file that does not exist. Nobody sees a red test then -- they see a broken
  # target -- and lit failures pile up unread. Named here rather than left to
  # chance: it was already in this venv, with nothing depending on it.
  "$TRITON_VENV/bin/pip" install -q "cmake==3.31.6" ninja pybind11 lit
  cd "$TRITON_ROOT"
  # PATH, because shutil.which searches PATH and pip does not add its own
  # venv's bin to it for the build subprocess.
  PATH="$TRITON_VENV/bin:$PATH" \
  TRITON_PLUGIN_DIRS="$TRITON_SHARED_ROOT" \
  LLVM_INCLUDE_DIRS="$LLVM_INSTALL/include" \
  LLVM_LIBRARY_DIR="$LLVM_INSTALL/lib" \
  LLVM_SYSPATH="$LLVM_INSTALL" \
  MAX_JOBS="$JOBS" \
    "$TRITON_VENV/bin/pip" install -e . --no-build-isolation

  # THE DRIVER NEEDS THIS BUILD TOO, because stage 1 runs there now and not in
  # this venv. What the image ships is the PyPI wheel, and the difference does
  # not show in the version: both say `__version__ == '3.6.0'`. It shows in the
  # ENTRY POINTS -- the wheel registers amd and nvidia, and `triton_shared` is
  # registered by the TRITON_PLUGIN_DIRS build above. A driver on the wheel
  # answers make_backend(GPUTarget("cpu")) with "0 compatible backends for
  # target (cpu)", which is why doctor checks the backend LIST and not a
  # version string.
  #
  # An editable install is three pointers, not a copy: the .pth that runs at
  # interpreter start, the finder that maps `triton` onto $TRITON_ROOT/python,
  # and the dist-info whose entry_points.txt carries triton_shared. Each
  # interpreter needs its own three; both then read the one tree and cannot
  # drift apart. The venv stays because it is the BUILD environment -- cmake
  # 3.31.6 for LLVM 23, which must not displace the image's 3.26.4 that
  # TOGSim's conan build is pinned to.
  DRIVER_PY=/usr/bin/python3.11
  DRIVER_SP="$("$DRIVER_PY" -c 'import site; print(site.getsitepackages()[0])')"
  VENV_SP="$("$TRITON_VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
  test -d "$DRIVER_SP" || { echo "driver site-packages not found: $DRIVER_SP" >&2; exit 1; }
  "$DRIVER_PY" -m pip uninstall -y triton >/dev/null 2>&1 || true
  rm -f  "$DRIVER_SP"/__editable__*triton*
  rm -rf "$DRIVER_SP"/triton-*.dist-info
  cp -a "$VENV_SP"/__editable__*triton* "$VENV_SP"/triton-*.dist-info "$DRIVER_SP/"
  # Loudly, here: the alternative is a RuntimeError out of triton's own
  # make_backend once a compile is already under way.
  "$DRIVER_PY" - <<'CHECK' || exit 1
import sys, triton
have = sorted(triton.backends.backends)
if "triton_shared" not in have:
    sys.exit(f"  driver triton has {have} -- triton_shared did not land")
print(f"  driver triton: {triton.__file__}")
print(f"  backends     : {', '.join(have)}")
CHECK

  # torch is still not installed in the venv, and kernel files still take it as
  # an argument (make_inputs(torch, seed)) rather than importing it at module
  # scope. That was once what let ONE kernel file serve two interpreters; now
  # it is what keeps load_spec cheap. See pytorchsim_triton_compiler/contract/kernel_spec.py.
  PYV=$("$TRITON_VENV/bin/python" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  OPT="$TRITON_ROOT/build/cmake.linux-x86_64-cpython-$PYV/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt"
  test -x "$OPT" || { echo "triton-shared-opt not at $OPT" >&2; exit 1; }
  printf '%s\n' "$("$OPT" --version 2>&1)" | head -2
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

log "done -- now run: pytorchsim-triton-compiler doctor"
