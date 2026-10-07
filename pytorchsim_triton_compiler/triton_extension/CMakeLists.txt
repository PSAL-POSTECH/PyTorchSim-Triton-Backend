# torchsim-triton-ext: everything the torchsim route adds to Triton, as one plugin .so loaded into
# an unmodified Triton (TRITON_PLUGIN_PATHS).
cmake_minimum_required(VERSION 3.20)
project(torchsim_triton_ext CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TRITON_SRC "" CACHE PATH "Triton source tree (v3.7.0, unmodified)")
set(TRITON_BUILD "" CACHE PATH "Triton build tree: libtriton.so built with TRITON_EXT_ENABLED=ON")
set(PYBIND11_INCLUDE "" CACHE PATH "pybind11 include dir")
set(PYTHON_INCLUDE "" CACHE PATH "Python include dir")

set(TRITON_SHARED_SRC ${CMAKE_CURRENT_SOURCE_DIR}/triton_shared)
set(TORCHSIM_DIALECT ${CMAKE_CURRENT_SOURCE_DIR}/../dialects/src)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/include
  ${TRITON_SHARED_SRC}/include ${CMAKE_CURRENT_BINARY_DIR}/ts-include
  ${TORCHSIM_DIALECT}/include
  ${TRITON_SRC}/include ${TRITON_BUILD}/include ${TRITON_SRC}/third_party ${TRITON_BUILD}/third_party
  ${TRITON_SRC}
  ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS} ${PYBIND11_INCLUDE} ${PYTHON_INCLUDE})

set(LLVM_TARGET_DEFINITIONS include/TorchsimExt/Passes.td)
mlir_tablegen(include/TorchsimExt/Passes.h.inc -gen-pass-decls -name TorchsimExt)
add_public_tablegen_target(TorchsimExtPassIncGen)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/TorchSim)
set(LLVM_TARGET_DEFINITIONS ${TORCHSIM_DIALECT}/include/TorchSim/TorchSimOps.td)
mlir_tablegen(include/TorchSim/TorchSimOps.h.inc -gen-op-decls)
mlir_tablegen(include/TorchSim/TorchSimOps.cpp.inc -gen-op-defs)
mlir_tablegen(include/TorchSim/TorchSimOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(include/TorchSim/TorchSimOpsDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(TorchSimDialectIncGen)

add_subdirectory(${TRITON_SHARED_SRC}/include ${CMAKE_CURRENT_BINARY_DIR}/ts-include)

set(RDV_DIR ${CMAKE_CURRENT_BINARY_DIR}/rdv)
add_custom_command(
  OUTPUT ${RDV_DIR}/RemoveDeadValues.cpp.o ${RDV_DIR}/LivenessAnalysis.cpp.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${RDV_DIR}
  COMMAND ${CMAKE_AR} x ${LLVM_LIBRARY_DIR}/libMLIRTransforms.a RemoveDeadValues.cpp.o
  COMMAND ${CMAKE_AR} x ${LLVM_LIBRARY_DIR}/libMLIRAnalysis.a LivenessAnalysis.cpp.o
  WORKING_DIRECTORY ${RDV_DIR}
  COMMENT "remove-dead-values objects from ${LLVM_LIBRARY_DIR}")

set(TS ${TRITON_SHARED_SRC}/lib)
set(TRITON_SHARED_SOURCES
  ${TS}/Analysis/MaskAnalysis.cpp ${TS}/Analysis/OpFoldResultUtils.cpp
  ${TS}/AnalysisStructured/PtrAnalysis.cpp ${TS}/Utils/Utils.cpp
  ${TS}/Conversion/TritonToStructured/TritonToStructuredPass.cpp
  ${TS}/Conversion/TritonToUnstructured/TritonToUnstructuredPass.cpp
  ${TS}/Conversion/TritonArithToLinalg/TritonArithToLinalg.cpp
  ${TS}/Conversion/TritonArithToLinalg/TritonArithToLinalgPass.cpp
  ${TS}/Conversion/TritonPtrToMemref/TritonPtrToMemrefPass.cpp
  ${TS}/Conversion/TritonToLinalgExperimental/TritonToPtrPass.cpp
  ${TS}/Conversion/TritonToLinalgExperimental/ReconcilePtrCastsPass.cpp
  ${TS}/Dialect/TPtr/IR/TPtrDialect.cpp ${TS}/Dialect/TPtr/IR/TPtrOps.cpp
  ${TS}/Dialect/TritonStructured/IR/TritonStructuredDialect.cpp
  ${TS}/Dialect/TritonStructured/IR/TritonStructuredOps.cpp
  ${TS}/Dialect/TritonTilingExt/IR/BufferizableOpInterfaceImpl.cpp
  ${TS}/Dialect/TritonTilingExt/IR/CumSum.cpp
  ${TS}/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.cpp)
set(OWN_SOURCES lib/TorchsimAnchorLayouts.cpp lib/TorchsimRemoveLayoutConversions.cpp lib/TorchsimRegularizeLayouts.cpp
                lib/TorchsimConvertTritonToTritonGPU.cpp lib/TorchsimGPUConversion.cpp lib/Plugin.cpp)
set(DIALECT_SOURCES ${TORCHSIM_DIALECT}/lib/TorchSimDialect.cpp ${TORCHSIM_DIALECT}/lib/TorchSimOps.cpp)

add_library(torchsim_triton_ext SHARED ${OWN_SOURCES} ${TRITON_SHARED_SOURCES} ${DIALECT_SOURCES}
            ${RDV_DIR}/RemoveDeadValues.cpp.o ${RDV_DIR}/LivenessAnalysis.cpp.o)
add_dependencies(torchsim_triton_ext TorchsimExtPassIncGen TorchSimDialectIncGen
  TPtrTableGen TritonStructuredTableGen TritonTilingExtOpsIncGen TritonTilingExtInterfacesIncGen
  TritonToStructuredConversionPassIncGen TritonToUnstructuredConversionPassIncGen
  TritonArithToLinalgConversionPassIncGen TritonPtrToMemrefConversionPassIncGen
  TritonToLinalgExperimentalConversionPassIncGen)

set_source_files_properties(lib/TorchsimAnchorLayouts.cpp lib/TorchsimRemoveLayoutConversions.cpp
  lib/TorchsimRegularizeLayouts.cpp lib/TorchsimConvertTritonToTritonGPU.cpp lib/TorchsimGPUConversion.cpp
  ${TRITON_SHARED_SOURCES} ${DIALECT_SOURCES} PROPERTIES COMPILE_OPTIONS "-fno-rtti;-fno-exceptions")
target_compile_options(torchsim_triton_ext PRIVATE -O2)

target_link_libraries(torchsim_triton_ext PRIVATE ${TRITON_BUILD}/libtriton.so
  MLIRLinalgDialect MLIRLinalgUtils MLIRLinalgTransforms MLIRTensorDialect MLIRTensorTransforms
  MLIRBufferizationDialect MLIRPtrDialect)
