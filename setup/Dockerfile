# syntax=docker/dockerfile:1.7
# Four stages, each FROM the one before, each a step list of setup/toolchain.sh:
#   system     FROM the vcix-accelerator environment image (spike, gem5, RISC-V toolchain, pk):
#              the distribution layer (setup/system.sh)
#   triton     LLVM 23 downloaded, triton v3.7.0 built -- tagged by setup/triton_tag.sh, built once per pin
#   toolchain  TOGSim and the accelerator model -- tagged by setup/toolchain_tag.sh
#   compiler   the compiler, its dialect and extension; doctor + smoke -- every push
# Locally: docker build -f setup/Dockerfile --target compiler \
#            --build-arg VCIX_ENV_IMAGE=$(. setup/versions.env; echo $VCIX_ENV_IMAGE) .
# CI (.github/workflows/images.yml) passes TRITON_IMAGE / TOOLCHAIN_IMAGE to start above a stage in the registry;
# VCIX_ENV_IMAGE=scratch then, since FROM may not be blank and `system` is never built.
ARG VCIX_ENV_IMAGE=scratch
ARG SYSTEM_IMAGE=system
ARG TRITON_IMAGE=triton
ARG TOOLCHAIN_IMAGE=toolchain

# =============================================================== system ======
FROM ${VCIX_ENV_IMAGE} AS system

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Only what system.sh reads (versions.env reads the submodule's, so that file comes too).
COPY setup/system.sh setup/versions.env /workspace/pytorchsim-triton-compiler/setup/
COPY third_party/vcix-accelerator/setup/versions.env /workspace/pytorchsim-triton-compiler/third_party/vcix-accelerator/setup/versions.env
RUN /workspace/pytorchsim-triton-compiler/setup/system.sh

# =============================================================== triton ======
FROM ${SYSTEM_IMAGE} AS triton

ARG JOBS=24
WORKDIR /workspace

COPY setup/versions.env setup/toolchain.sh setup/triton.sh /workspace/pytorchsim-triton-compiler/setup/

RUN /workspace/pytorchsim-triton-compiler/setup/toolchain.sh -j ${JOBS} llvm triton

# ============================================================ toolchain ======
FROM ${TRITON_IMAGE} AS toolchain

ARG JOBS=24
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

COPY setup/ /workspace/pytorchsim-triton-compiler/setup/
COPY third_party/vcix-accelerator /workspace/pytorchsim-triton-compiler/third_party/vcix-accelerator

ENV RISCV=/opt/vcix-env/riscv \
    PATH=/opt/vcix-env/riscv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    GEM5_PATH=/opt/vcix-env/gem5/build/RISCV/gem5.opt

RUN /workspace/pytorchsim-triton-compiler/setup/toolchain.sh -j ${JOBS} pytorchsim vcix
ENV TORCHSIM_DIR=/workspace/PyTorchSim \
    TORCHSIM_COMPILE_TOGSIM_INCLUDE=/workspace/PyTorchSim/TOGSim/include \
    TOGSIM_CONFIG=/workspace/PyTorchSim/configs/systolic_ws_256x256_c1_simple_noc_tpuv6e_functional_only.yml \
    TORCHSIM_COMPILE_LLVM_PATH=/workspace/LLVM_DIR/llvm-project/build/install/bin \
    TORCHSIM_COMPILE_VCIX_MODEL=/workspace/vcix-build/libtpu.so \
    TORCHSIM_TRITON_CODEGEN=1 \
    LIT_USE_INTERNAL_SHELL=1

# ============================================================= compiler ======
FROM ${TOOLCHAIN_IMAGE} AS compiler
WORKDIR /workspace

COPY . /workspace/pytorchsim-triton-compiler
RUN cd /workspace/pytorchsim-triton-compiler && python3 -m pip install --no-deps -e .

RUN /workspace/pytorchsim-triton-compiler/setup/toolchain.sh dialect ext

ENV TORCHSIM_COMPILE_DIR=/workspace/pytorchsim-triton-compiler \
    TORCHSIM_COMPILE_TORCHSIM_DIALECT=/workspace/torchsim-dialect-build \
    TORCHSIM_COMPILE_TARGET=/workspace/pytorchsim-triton-compiler/tests/targets/tpuv6e.json

# doctor: everything is where the compiler looks. smoke/: the route runs down each lowering --
# elementwise (add, mul, relu), the systolic array (gemm, bmm), the vector reduction (reduce).
RUN python3 /workspace/pytorchsim-triton-compiler/pytorchsim-triton-compiler doctor && \
    cd /workspace/pytorchsim-triton-compiler && \
    python3 tests/run.py tests/cases/smoke/*.py --no-baseline -j 6 --workdir /tmp/smoke \
        | tee /tmp/smoke.out && grep -q '^6 of 6 passing' /tmp/smoke.out && rm -rf /tmp/smoke /tmp/smoke.out
