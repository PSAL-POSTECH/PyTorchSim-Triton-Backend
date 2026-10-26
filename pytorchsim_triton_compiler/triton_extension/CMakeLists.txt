# pytorchsim-triton-ext: everything the pytorchsim route adds to Triton, as one plugin .so loaded into
# Triton with one edit, its tensor numel limit (setup/versions.env), through TRITON_PLUGIN_PATHS.
cmake_minimum_required(VERSION 3.20)
project(pytorchsim_triton_ext CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TRITON_SRC "" CACHE PATH "Triton source tree (v3.8.0, unmodified)")
set(TRITON_BUILD "" CACHE PATH "Triton build tree: libtriton.so built with TRITON_EXT_ENABLED=ON")

set(PYTORCHSIM_DIALECT ${CMAKE_CURRENT_SOURCE_DIR}/../dialects/src)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/include
  ${PYTORCHSIM_DIALECT}/include
  ${TRITON_SRC}/include ${TRITON_BUILD}/include ${TRITON_SRC}/third_party ${TRITON_BUILD}/third_party
  ${TRITON_SRC}
  ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

set(LLVM_TARGET_DEFINITIONS include/PyTorchSimExt/Passes.td)
mlir_tablegen(include/PyTorchSimExt/Passes.h.inc -gen-pass-decls -name PyTorchSimExt)
add_public_tablegen_target(PyTorchSimExtPassIncGen)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/PyTorchSim)
set(LLVM_TARGET_DEFINITIONS ${PYTORCHSIM_DIALECT}/include/PyTorchSim/PyTorchSimOps.td)
mlir_tablegen(include/PyTorchSim/PyTorchSimOps.h.inc -gen-op-decls)
mlir_tablegen(include/PyTorchSim/PyTorchSimOps.cpp.inc -gen-op-defs)
mlir_tablegen(include/PyTorchSim/PyTorchSimOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(include/PyTorchSim/PyTorchSimOpsDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(PyTorchSimDialectIncGen)
set(LLVM_TARGET_DEFINITIONS ${PYTORCHSIM_DIALECT}/include/PyTorchSim/PyTorchSimExternOps.td)
mlir_tablegen(include/PyTorchSim/PyTorchSimExternOps.h.inc -gen-op-decls)
mlir_tablegen(include/PyTorchSim/PyTorchSimExternOps.cpp.inc -gen-op-defs)
mlir_tablegen(include/PyTorchSim/PyTorchSimExternOpsDialect.h.inc -gen-dialect-decls -dialect=pytorchsim_extern)
mlir_tablegen(include/PyTorchSim/PyTorchSimExternOpsDialect.cpp.inc -gen-dialect-defs -dialect=pytorchsim_extern)
add_public_tablegen_target(PyTorchSimExternDialectIncGen)

add_subdirectory(include)

set(RDV_DIR ${CMAKE_CURRENT_BINARY_DIR}/rdv)
add_custom_command(
  OUTPUT ${RDV_DIR}/RemoveDeadValues.cpp.o ${RDV_DIR}/LivenessAnalysis.cpp.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${RDV_DIR}
  COMMAND ${CMAKE_AR} x ${LLVM_LIBRARY_DIR}/libMLIRTransforms.a RemoveDeadValues.cpp.o
  COMMAND ${CMAKE_AR} x ${LLVM_LIBRARY_DIR}/libMLIRAnalysis.a LivenessAnalysis.cpp.o
  WORKING_DIRECTORY ${RDV_DIR}
  COMMENT "remove-dead-values objects from ${LLVM_LIBRARY_DIR}")

set(LOWERING_SOURCES
  lib/Analysis/MaskAnalysis.cpp lib/Analysis/OpFoldResultUtils.cpp
  lib/AnalysisStructured/PtrAnalysis.cpp lib/Utils/Utils.cpp
  lib/Conversion/TritonToStructured/TritonToStructuredPass.cpp
  lib/Conversion/TritonToUnstructured/TritonToUnstructuredPass.cpp
  lib/Conversion/TritonArithToLinalg/TritonArithToLinalg.cpp
  lib/Conversion/TritonArithToLinalg/TritonArithToLinalgPass.cpp
  lib/Conversion/TritonPtrToMemref/TritonPtrToMemrefPass.cpp
  lib/Conversion/TritonToLinalgExperimental/TritonToPtrPass.cpp
  lib/Conversion/TritonToLinalgExperimental/ReconcilePtrCastsPass.cpp
  lib/Dialect/TPtr/IR/TPtrDialect.cpp lib/Dialect/TPtr/IR/TPtrOps.cpp
  lib/Dialect/TritonStructured/IR/TritonStructuredDialect.cpp
  lib/Dialect/TritonStructured/IR/TritonStructuredOps.cpp
  lib/Dialect/TritonTilingExt/IR/BufferizableOpInterfaceImpl.cpp
  lib/Dialect/TritonTilingExt/IR/CumSum.cpp
  lib/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.cpp)
set(OWN_SOURCES lib/PyTorchSimAnchorLayouts.cpp lib/PyTorchSimRemoveLayoutConversions.cpp lib/PyTorchSimRegularizeLayouts.cpp
                lib/PyTorchSimConvertTritonToTritonGPU.cpp lib/PyTorchSimGPUConversion.cpp lib/Plugin.cpp)
set(DIALECT_SOURCES ${PYTORCHSIM_DIALECT}/lib/PyTorchSimDialect.cpp ${PYTORCHSIM_DIALECT}/lib/PyTorchSimOps.cpp
                    ${PYTORCHSIM_DIALECT}/lib/PyTorchSimExternDialect.cpp)

add_library(pytorchsim_triton_ext SHARED ${OWN_SOURCES} ${LOWERING_SOURCES} ${DIALECT_SOURCES}
            ${RDV_DIR}/RemoveDeadValues.cpp.o ${RDV_DIR}/LivenessAnalysis.cpp.o)
add_dependencies(pytorchsim_triton_ext PyTorchSimExtPassIncGen PyTorchSimDialectIncGen PyTorchSimExternDialectIncGen
  TPtrTableGen TritonStructuredTableGen TritonTilingExtOpsIncGen TritonTilingExtInterfacesIncGen
  TritonToStructuredConversionPassIncGen TritonToUnstructuredConversionPassIncGen
  TritonArithToLinalgConversionPassIncGen TritonPtrToMemrefConversionPassIncGen
  TritonToLinalgExperimentalConversionPassIncGen)
# No RTTI, no exceptions, as libtriton and its example plugins; Plugin.cpp too.
target_compile_options(pytorchsim_triton_ext PRIVATE -O2 -fno-rtti -fno-exceptions)
target_link_libraries(pytorchsim_triton_ext PRIVATE ${TRITON_BUILD}/libtriton.so
  MLIRLinalgDialect MLIRLinalgUtils MLIRLinalgTransforms MLIRTensorDialect MLIRTensorTransforms
  MLIRBufferizationDialect MLIRPtrDialect)
