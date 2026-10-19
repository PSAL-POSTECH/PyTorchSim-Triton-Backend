# syntax=docker/dockerfile:1.7
# Five stages, each FROM the one before, each a step list of setup/toolchain.sh, and a runtime image of the last:
#   system     FROM the vcix-accelerator environment image (spike, gem5, RISC-V toolchain, pk):
#              the distribution layer (setup/system.sh)
#   triton     LLVM 23 downloaded, triton v3.8.0 built -- tagged by setup/triton_tag.sh, built once per pin
#   toolchain  TOGSim, the third_party/TOGSim submodule (setup/togsim.sh) -- tagged by setup/toolchain_tag.sh
#   prebuilt   the accelerator model, the dialect and the extension built, then the toolchain less what only
#              builds them, in a few layers -- tagged by setup/prebuilt_tag.sh
#   compiler   the compiler's checkout over the prebuilt image; doctor + smoke -- every push
#   runtime    the compiler image less torch: what PyTorchSim's CI pulls -- every push
# Locally: docker build -f setup/Dockerfile --target compiler \
#            --build-arg VCIX_ENV_IMAGE=$(. setup/versions.env; echo $VCIX_ENV_IMAGE) .
# CI (.github/workflows/images.yml) passes TRITON_IMAGE / TOOLCHAIN_IMAGE / PREBUILT_IMAGE to start above a stage
# in the registry; VCIX_ENV_IMAGE=scratch then, since FROM may not be blank and `system` is never built.
ARG VCIX_ENV_IMAGE=scratch
ARG SYSTEM_IMAGE=system
ARG TRITON_IMAGE=triton
ARG TOOLCHAIN_IMAGE=toolchain
ARG PREBUILT_IMAGE=prebuilt

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

# TOGSim is built where the compiler looks for it: in the submodule's tree, under the checkout the
# compiler stage copies over it (the build directory is not in the context, so it stays).
COPY setup/versions.env setup/togsim.sh /workspace/pytorchsim-triton-compiler/setup/
COPY third_party/TOGSim /workspace/pytorchsim-triton-compiler/third_party/TOGSim
RUN /workspace/pytorchsim-triton-compiler/setup/togsim.sh -j ${JOBS}

# ============================================================= prebuilt ======
# What the compiler builds and does not change per push: the accelerator model, the torchsim dialect and the
# extension, from only what toolchain.sh's vcix, dialect and ext steps read (setup/prebuilt_tag.sh hashes the same
# list). Their builds are outside the checkout; ext links two names in triton-src into it, which the compiler
# stage's copy fills. The vcix step imports the package for target.models(), hence its five files.
FROM ${TOOLCHAIN_IMAGE} AS prebuilt-build

ARG JOBS=24
WORKDIR /workspace

COPY setup/versions.env setup/toolchain.sh /workspace/pytorchsim-triton-compiler/setup/
COPY third_party/vcix-accelerator /workspace/pytorchsim-triton-compiler/third_party/vcix-accelerator
COPY pytorchsim_triton_compiler/__init__.py pytorchsim_triton_compiler/machine.py \
     pytorchsim_triton_compiler/toolchain.py /workspace/pytorchsim-triton-compiler/pytorchsim_triton_compiler/
COPY pytorchsim_triton_compiler/contract/__init__.py pytorchsim_triton_compiler/contract/target.py \
     /workspace/pytorchsim-triton-compiler/pytorchsim_triton_compiler/contract/
COPY pytorchsim_triton_compiler/dialects/src /workspace/pytorchsim-triton-compiler/pytorchsim_triton_compiler/dialects/src
COPY pytorchsim_triton_compiler/triton_extension/CMakeLists.txt \
     /workspace/pytorchsim-triton-compiler/pytorchsim_triton_compiler/triton_extension/
COPY pytorchsim_triton_compiler/triton_extension/include \
     /workspace/pytorchsim-triton-compiler/pytorchsim_triton_compiler/triton_extension/include
COPY pytorchsim_triton_compiler/triton_extension/lib \
     /workspace/pytorchsim-triton-compiler/pytorchsim_triton_compiler/triton_extension/lib
RUN /workspace/pytorchsim-triton-compiler/setup/toolchain.sh -j ${JOBS} vcix dialect ext && \
    rm -rf /workspace/pytorchsim-triton-compiler/pytorchsim_triton_compiler

# Deleted here, then copied up FROM scratch, so the deleted bytes are in no layer at all:
#   LLVM 23's install but llc, opt, mlir-translate (toolchain.py's LLVM_BIN) and its python_packages
#     (the MLIR bindings); lib, include, src, share are what the dialect and the extension build against
#   the LLVM 22 (clang) the vcix-accelerator environment carries beside its RISC-V GCC, which is the one
#     the route calls (toolchain.py's RISCV_*)
#   libtriton.so's debug info (TritonRelBuildWithAsserts, setup/triton.sh)
#   what built the images: build-tools, nlohmann-json, triton's .git, conan's packages, pip's cache
# torch stays: the smoke cases take their reference values from it.
FROM prebuilt-build AS prebuilt-tree
RUN I=/workspace/LLVM_DIR/llvm-project/build/install && R=/opt/vcix-env/riscv && \
    find "$I/bin" -mindepth 1 -maxdepth 1 ! -name llc ! -name opt ! -name mlir-translate -exec rm -rf {} + && \
    rm -rf "$I/lib" "$I/include" "$I/src" "$I/share" && \
    find "$R/bin" -mindepth 1 -maxdepth 1 \( ! -name 'riscv64-unknown-elf-*' -o -name 'riscv64-unknown-elf-clang*' \) \
        -exec rm -rf {} + && \
    rm -rf "$R"/lib/libclang.so* "$R"/lib/libLTO.so* "$R"/lib/libRemarks.so* "$R/lib/LLVMgold.so" \
           "$R/lib/clang" "$R/lib/cmake" "$R/lib/libear" "$R/lib/libscanbuild" && \
    strip --strip-debug /workspace/triton-src/build/libtriton.so && \
    rm -rf /workspace/build-tools /workspace/nlohmann-json /workspace/triton-src/.git /root/.conan /root/.cache

# The tree less its four big directories, which go up as layers of their own: no blob is most of the image,
# and the five download side by side.
FROM prebuilt-tree AS prebuilt-rest
RUN rm -rf /opt/vcix-env /workspace/LLVM_DIR /workspace/triton-src /usr/local/lib/python3.11/dist-packages/torch

FROM scratch AS prebuilt
COPY --from=prebuilt-rest / /
COPY --from=prebuilt-tree /usr/local/lib/python3.11/dist-packages/torch /usr/local/lib/python3.11/dist-packages/torch
COPY --from=prebuilt-tree /opt/vcix-env /opt/vcix-env
COPY --from=prebuilt-tree /workspace/LLVM_DIR /workspace/LLVM_DIR
COPY --from=prebuilt-tree /workspace/triton-src /workspace/triton-src
# The toolchain image's config, which a copy of its tree does not carry.
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    VCIX_ENV_ROOT=/opt/vcix-env \
    DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace
CMD ["/bin/bash"]

# ============================================================= compiler ======
FROM ${PREBUILT_IMAGE} AS compiler
WORKDIR /workspace

COPY . /workspace/pytorchsim-triton-compiler
RUN cd /workspace/pytorchsim-triton-compiler && python3 -m pip install --no-deps -e .

# The tree is the compiler's defaults (/workspace, /opt/vcix-env); the machine is a choice.
ENV TORCHSIM_COMPILE_TARGET=tpuv6e

# doctor: everything is where the compiler looks. smoke/: the route runs down each lowering --
# elementwise (add, mul, relu), the matrix unit (gemm, bmm), the vector reduction (reduce).
RUN python3 /workspace/pytorchsim-triton-compiler/pytorchsim-triton-compiler doctor && \
    cd /workspace/pytorchsim-triton-compiler && \
    python3 tests/run.py tests/cases/smoke/*.py --no-baseline -j 6 --workdir /tmp/smoke \
        | tee /tmp/smoke.out && grep -q '^6 of 6 passing' /tmp/smoke.out && rm -rf /tmp/smoke /tmp/smoke.out

# ============================================================== runtime ======
# What runs the route and nothing that builds it: PyTorchSim's base image is FROM this one, pulled where no
# layer is cached. The prebuilt image already dropped what builds; this drops torch (PyTorchSim installs its
# own, and the compiler does not import it -- tests/runtime and the smoke cases do) and pip's cache, then
# copies the tree up as one layer, so the deleted bytes are in no layer at all.
# The dialect and the extension rebuild on the toolchain image, which keeps what they build against.
FROM compiler AS runtime-tree
RUN python3 -m pip uninstall -y torch && rm -rf /root/.cache && \
    python3 /workspace/pytorchsim-triton-compiler/pytorchsim-triton-compiler doctor

FROM scratch AS runtime
COPY --from=runtime-tree / /
# The compiler image's config, which a copy of its tree does not carry.
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    VCIX_ENV_ROOT=/opt/vcix-env \
    DEBIAN_FRONTEND=noninteractive \
    TORCHSIM_COMPILE_TARGET=tpuv6e
WORKDIR /workspace
CMD ["/bin/bash"]
