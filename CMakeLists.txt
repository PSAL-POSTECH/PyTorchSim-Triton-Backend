# torchsim-triton-ext: everything the torchsim route adds to Triton, as one plugin .so loaded into
# an unmodified Triton (TRITON_PLUGIN_PATHS). Built against a finished Triton build tree and linked
# to its libtriton.so; triton_shared and psto's TorchSim dialect are compiled in from their trees.
cmake_minimum_required(VERSION 3.20)
project(torchsim_triton_ext CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TRITON_SRC "" CACHE PATH "Triton source tree (v3.7.0, unmodified)")
set(TRITON_BUILD "" CACHE PATH "Triton build tree: libtriton.so built with TRITON_EXT_ENABLED=ON")
set(TRITON_SHARED_SRC "" CACHE PATH "triton_shared source tree")
set(TORCHSIM_DIALECT "" CACHE PATH "psto's dialect/ tree (TorchSim ODS and sources)")
set(PYBIND11_INCLUDE "" CACHE PATH "pybind11 include dir")
set(PYTHON_INCLUDE "" CACHE PATH "Python include dir")

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

# ---- ODS ---------------------------------------------------------------------------------------
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

# triton_shared's include/ tree only runs TableGen: reused as it is.
add_subdirectory(${TRITON_SHARED_SRC}/include ${CMAKE_CURRENT_BINARY_DIR}/ts-include)

# ---- remove-dead-values: MLIR's own pass, which libtriton.so does not carry ------------------------
# The two objects it needs, taken out of this LLVM's static libraries; every symbol they read
# libtriton.so exports, and none they define clashes with it.
set(RDV_DIR ${CMAKE_CURRENT_BINARY_DIR}/rdv)
add_custom_command(
  OUTPUT ${RDV_DIR}/RemoveDeadValues.cpp.o ${RDV_DIR}/LivenessAnalysis.cpp.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${RDV_DIR}
  COMMAND ${CMAKE_AR} x ${LLVM_LIBRARY_DIR}/libMLIRTransforms.a RemoveDeadValues.cpp.o
  COMMAND ${CMAKE_AR} x ${LLVM_LIBRARY_DIR}/libMLIRAnalysis.a LivenessAnalysis.cpp.o
  WORKING_DIRECTORY ${RDV_DIR}
  COMMENT "remove-dead-values objects from ${LLVM_LIBRARY_DIR}")

# ---- sources -------------------------------------------------------------------------------------
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
# As libtriton's own Triton code: no RTTI, no exceptions. Plugin.cpp keeps both: the plugin
# header pulls in pybind11.
set_source_files_properties(lib/TorchsimAnchorLayouts.cpp lib/TorchsimRemoveLayoutConversions.cpp
  lib/TorchsimRegularizeLayouts.cpp lib/TorchsimConvertTritonToTritonGPU.cpp lib/TorchsimGPUConversion.cpp
  ${TRITON_SHARED_SOURCES} ${DIALECT_SOURCES} PROPERTIES COMPILE_OPTIONS "-fno-rtti;-fno-exceptions")
target_compile_options(torchsim_triton_ext PRIVATE -O2)
# libtriton.so FIRST: every MLIR symbol it carries is taken from it, so the one MLIR in the process
# stays its. The static libraries after it only fill in what Triton's GPU route leaves out --
# linalg, tensor, bufferization and ptr, which triton_shared lowers into.
target_link_libraries(torchsim_triton_ext PRIVATE ${TRITON_BUILD}/libtriton.so
  MLIRLinalgDialect MLIRLinalgUtils MLIRLinalgTransforms MLIRTensorDialect MLIRTensorTransforms
  MLIRBufferizationDialect MLIRPtrDialect)
