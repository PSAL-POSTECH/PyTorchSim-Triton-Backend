# syntax=docker/dockerfile:1.7
# The compiler and everything it runs, with no PyTorchSim frontend, in FOUR images:
#
#   system     FROM ubuntu:22.04: the distribution layer setup/system.sh installs --
#              python3.11 as python3, cmake 3.26 + conan 1.56 for TOGSim, torch.
#   triton     FROM system: one LLVM 23 (vcix patch), downloaded, and triton v3.7.0
#              unmodified, built against it (setup/triton.sh). Changes only when its
#              pins or that script do (setup/triton_tag.sh), so triton is built once.
#   toolchain  FROM triton: the RISC-V toolchain and pk, spike and gem5 from their
#              forks, TOGSim. Changes when setup/ does.
#   compiler   FROM toolchain: the compiler itself, the dialect it generates, and the
#              extension (extension/) loaded into triton. Minutes, changes on every push.
#
# Every path matches restore.sh's layout under /workspace, so the compiler's
# defaults need no env. Every repository it reads is public, so no token. Locally, one
# command builds all three:
#   docker build -f setup/Dockerfile --target compiler -t pytorchsim-triton-backend .
# CI builds `triton` and `toolchain` on the big self-hosted runner and passes each in as
# TRITON_IMAGE and TOOLCHAIN_IMAGE (.github/workflows/triton-image.yml, toolchain-image.yml,
# compiler-image.yml).
#
# THE BASE IS A CLEAN UBUNTU, and that is the point. It used to be
# ghcr.io/psal-postech/torchsim_tnpu, which carried its own LLVM, gem5, spike, TOGSim and an
# old checkout of this repository at /workspace/triton-npu -- every one of them a second
# copy of something the stages below build, and 8.53 GB compressed before this file started.
# What the base really gave is python3.11 (the MLIR bindings in LLVM_SHA's install are
# cpython-3.11 ABI; torchsim_base's conda 3.10 is why the base moved to tnpu in the first
# place) and the RISC-V toolchain: system.sh and restore.sh's `riscv` step answer both.
# ubuntu:22.04 is the SAME distribution the LLVM release is built in -- jammy, glibc 2.35,
# GLIBCXX_3.4.30 -- so the downloaded install keeps its ABI partner.
ARG BASE_IMAGE=ubuntu:22.04
ARG SYSTEM_IMAGE=system
ARG TRITON_IMAGE=triton
ARG TOOLCHAIN_IMAGE=toolchain

# =============================================================== system ======
FROM ${BASE_IMAGE} AS system

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Only what system.sh reads, so a pin that moves does not reinstall the distribution unless
# it is one of system.sh's own. versions.env reads the submodule's, so that one file comes too.
COPY setup/system.sh setup/versions.env /workspace/pytorchsim-triton-compiler/setup/
COPY third_party/vcix-accelerator/setup/versions.env /workspace/pytorchsim-triton-compiler/third_party/vcix-accelerator/setup/versions.env
RUN /workspace/pytorchsim-triton-compiler/setup/system.sh

# =============================================================== triton ======
FROM ${SYSTEM_IMAGE} AS triton

ARG JOBS=24
WORKDIR /workspace

# Only what the triton build reads, so nothing else moves this layer.
COPY setup/versions.env setup/restore.sh setup/triton.sh /workspace/pytorchsim-triton-compiler/setup/

# --- LLVM 23 + vcix patch, downloaded from the fork that publishes it; triton v3.7.0 built ---
# --- against it, its headers and libtriton.so kept for the extension                    ---
RUN /workspace/pytorchsim-triton-compiler/setup/restore.sh -j ${JOBS} llvm triton

# ============================================================ toolchain ======
FROM ${TRITON_IMAGE} AS toolchain

ARG PYTORCHSIM_REPO=student-Jungmin/PyTorchSim
ARG PYTORCHSIM_REF=84fec6e
ARG JOBS=24
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Only setup/ is needed here, so a compiler change does not rebuild this stage.
COPY setup/ /workspace/pytorchsim-triton-compiler/setup/

# --- the RISC-V toolchain and pk, at the submodule's pins ---
RUN /workspace/pytorchsim-triton-compiler/setup/restore.sh -j ${JOBS} riscv
ENV RISCV=/workspace/riscv \
    PATH=/workspace/riscv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# --- spike: the fork, one build for spike AND libcustomext.so (the pair rule) ---
RUN /workspace/pytorchsim-triton-compiler/setup/restore.sh -j ${JOBS} spike && \
    rm -rf /workspace/riscv-isa-sim/build
ENV TORCHSIM_COMPILE_SPIKE=/workspace/riscv-isa-sim/install/bin/spike \
    TORCHSIM_COMPILE_PK=/workspace/riscv-pk/build/pk

# --- gem5: the release vcix-accelerator pins, not a build of our own ---
RUN /workspace/pytorchsim-triton-compiler/setup/restore.sh gem5
ENV GEM5_PATH=/workspace/gem5/build/RISCV/gem5.opt

# --- TOGSim at the pin. Submodules recursive AFTER the checkout: the clone is ---
# --- --no-checkout, so before it `git submodule` has no .gitmodules to read    ---
# --- and exits 0 having done nothing. protobuf needs its googletest to configure. ---
# --- and nothing else of PyTorchSim: no frontend, no device  ---
RUN git clone -q --filter=blob:none --no-checkout \
        "https://github.com/${PYTORCHSIM_REPO}.git" /workspace/PyTorchSim && \
    cd /workspace/PyTorchSim && git sparse-checkout set TOGSim configs && \
    git checkout -q "${PYTORCHSIM_REF}" && \
    git submodule update --init --recursive --depth 1 --jobs 8 && \
    test -f TOGSim/extern/onnx/CMakeLists.txt && \
    cd TOGSim && mkdir -p build && cd build && \
    conan install .. --build=missing && cmake .. && make -j ${JOBS} && \
    find . -name '*.o' -delete
ENV TORCHSIM_DIR=/workspace/PyTorchSim \
    TORCHSIM_COMPILE_TOGSIM_INCLUDE=/workspace/PyTorchSim/TOGSim/include \
    TOGSIM_CONFIG=/workspace/PyTorchSim/configs/systolic_ws_256x256_c1_simple_noc_tpuv6e_functional_only.yml \
    TORCHSIM_COMPILE_LLVM_PATH=/workspace/LLVM_DIR/llvm-project/build/install/bin \
    TORCHSIM_TRITON_CODEGEN=1 \
    LIT_USE_INTERNAL_SHELL=1

# --- the accelerator model, from the submodule alone so that a compiler ---
# --- change still does not rebuild this stage                           ---
COPY third_party/vcix-accelerator /workspace/pytorchsim-triton-compiler/third_party/vcix-accelerator
RUN /workspace/pytorchsim-triton-compiler/setup/restore.sh -j ${JOBS} vcix
ENV TORCHSIM_COMPILE_VCIX_MODEL=/workspace/vcix-build/libtpu.so

# ============================================================= compiler ======
FROM ${TOOLCHAIN_IMAGE} AS compiler
WORKDIR /workspace

COPY . /workspace/pytorchsim-triton-compiler
RUN cd /workspace/pytorchsim-triton-compiler && python3 -m pip install --no-deps -e .

# The torchsim dialect: tablegen output + a nanobind extension built against THE
# one LLVM install above. The compiler refuses a build whose stamp does not match
# its own .td files, so this is rebuilt whenever the compiler layer is.
RUN /workspace/build-tools/bin/cmake -G Ninja \
        -S /workspace/pytorchsim-triton-compiler/dialect -B /workspace/torchsim-dialect-build \
        -DMLIR_INSTALL=/workspace/LLVM_DIR/llvm-project/build/install \
        -DCMAKE_BUILD_TYPE=Release && \
    /workspace/build-tools/bin/ninja -C /workspace/torchsim-dialect-build

# The extension: everything the route adds to triton, one plugin .so, and its backend.
RUN /workspace/pytorchsim-triton-compiler/setup/restore.sh ext

ENV TORCHSIM_COMPILE_DIR=/workspace/pytorchsim-triton-compiler \
    TORCHSIM_COMPILE_TORCHSIM_DIALECT=/workspace/torchsim-dialect-build \
    TORCHSIM_COMPILE_TARGET=/workspace/pytorchsim-triton-compiler/tests/targets/tpuv6e.json

# doctor stops the build if any of the above is not where the compiler looks,
# and add.py proves the whole route (triton -> spike -> verify) once.
RUN python3 /workspace/pytorchsim-triton-compiler/pytorchsim-triton-compiler doctor && \
    cd /workspace/pytorchsim-triton-compiler && \
    python3 tests/run.py tests/cases/smoke/add.py --no-baseline --workdir /tmp/smoke \
        | tee /tmp/smoke.out && grep -q '^1 of 1 passing' /tmp/smoke.out && rm -rf /tmp/smoke /tmp/smoke.out
