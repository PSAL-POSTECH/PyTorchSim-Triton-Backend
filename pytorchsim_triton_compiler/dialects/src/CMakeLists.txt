cmake_minimum_required(VERSION 3.20)
project(torchsim-dialect LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(MLIR_INSTALL "${MLIR_INSTALL}" CACHE PATH "LLVM/MLIR install prefix")
set(MLIR_PY_LIBS "${MLIR_INSTALL}/python_packages/mlir_core/mlir/_mlir_libs")
find_program(MLIR_TBLGEN mlir-tblgen HINTS "${MLIR_INSTALL}/bin" REQUIRED)
find_library(MLIR_PY_CAPI NAMES MLIRPythonCAPI HINTS "${MLIR_PY_LIBS}" REQUIRED)

set(GEN "${CMAKE_CURRENT_BINARY_DIR}/include")
file(MAKE_DIRECTORY "${GEN}/PyTorchSim")
set(DIALECT_TD "${CMAKE_CURRENT_SOURCE_DIR}/include/PyTorchSim/PyTorchSimDialect.td")
set(TD "${CMAKE_CURRENT_SOURCE_DIR}/include/PyTorchSim/PyTorchSimOps.td")
set(TOGSIM_TD "${CMAKE_CURRENT_SOURCE_DIR}/include/PyTorchSim/TogsimOps.td")
set(TPU_TD "${CMAKE_CURRENT_SOURCE_DIR}/include/PyTorchSim/PyTorchSimTpuOps.td")

# `td` defines one dialect's ops over DIALECT_TD, which holds the shared resources and the pytorchsim
# dialect itself; -dialect= picks that dialect out where both are in scope.
function(tblgen td dialect out args)
  add_custom_command(
    OUTPUT "${GEN}/PyTorchSim/${out}"
    COMMAND "${MLIR_TBLGEN}" ${args} -dialect=${dialect} "${td}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include" -I "${MLIR_INSTALL}/include"
            -o "${GEN}/PyTorchSim/${out}"
    DEPENDS "${td}" "${DIALECT_TD}"
    COMMENT "tblgen ${out}")
endfunction()

tblgen("${TD}" pytorchsim PyTorchSimOps.h.inc          -gen-op-decls)
tblgen("${TD}" pytorchsim PyTorchSimOps.cpp.inc        -gen-op-defs)
tblgen("${TD}" pytorchsim PyTorchSimOpsDialect.h.inc   -gen-dialect-decls)
tblgen("${TD}" pytorchsim PyTorchSimOpsDialect.cpp.inc -gen-dialect-defs)
tblgen("${TOGSIM_TD}" togsim TogsimOps.h.inc          -gen-op-decls)
tblgen("${TOGSIM_TD}" togsim TogsimOps.cpp.inc        -gen-op-defs)
tblgen("${TOGSIM_TD}" togsim TogsimOpsDialect.h.inc   -gen-dialect-decls)
tblgen("${TOGSIM_TD}" togsim TogsimOpsDialect.cpp.inc -gen-dialect-defs)
tblgen("${TPU_TD}" pytorchsim_tpu PyTorchSimTpuOps.h.inc          -gen-op-decls)
tblgen("${TPU_TD}" pytorchsim_tpu PyTorchSimTpuOps.cpp.inc        -gen-op-defs)
tblgen("${TPU_TD}" pytorchsim_tpu PyTorchSimTpuOpsDialect.h.inc   -gen-dialect-decls)
tblgen("${TPU_TD}" pytorchsim_tpu PyTorchSimTpuOpsDialect.cpp.inc -gen-dialect-defs)

function(python_ops td dialect out)
  add_custom_command(
    OUTPUT "${out}"
    COMMAND "${MLIR_TBLGEN}" -gen-python-op-bindings -bind-dialect=${dialect} "${td}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include" -I "${MLIR_INSTALL}/include"
            -o "${out}"
    DEPENDS "${td}" "${DIALECT_TD}"
    COMMENT "tblgen ${out}")
endfunction()

set(PY_OPS "${CMAKE_CURRENT_BINARY_DIR}/_pytorchsim_ops_gen.py")
python_ops("${TD}" pytorchsim "${PY_OPS}")
set(PY_TOGSIM_OPS "${CMAKE_CURRENT_BINARY_DIR}/_togsim_ops_gen.py")
python_ops("${TOGSIM_TD}" togsim "${PY_TOGSIM_OPS}")
set(PY_TPU_OPS "${CMAKE_CURRENT_BINARY_DIR}/_pytorchsim_tpu_ops_gen.py")
python_ops("${TPU_TD}" pytorchsim_tpu "${PY_TPU_OPS}")

find_program(LLVM_TBLGEN llvm-tblgen HINTS "${MLIR_INSTALL}/bin" REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
set(PY_TABLE "${CMAKE_CURRENT_BINARY_DIR}/_pytorchsim_operands.py")
add_custom_command(
  OUTPUT "${PY_TABLE}"
  COMMAND "${LLVM_TBLGEN}" --dump-json "${TD}"
          -I "${CMAKE_CURRENT_SOURCE_DIR}/include" -I "${MLIR_INSTALL}/include"
          -o "${CMAKE_CURRENT_BINARY_DIR}/PyTorchSimOps.json"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/operand_table.py"
          "${CMAKE_CURRENT_BINARY_DIR}/PyTorchSimOps.json" "${PY_TABLE}"
  DEPENDS "${TD}" "${CMAKE_CURRENT_SOURCE_DIR}/include/PyTorchSim/PyTorchSimDialect.td"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/operand_table.py"
  COMMENT "operand table _pytorchsim_operands.py")

set(PY_STAMP "${CMAKE_CURRENT_BINARY_DIR}/_pytorchsim_stamp.py")
add_custom_command(
  OUTPUT "${PY_STAMP}"
  COMMAND "${CMAKE_COMMAND}" -DOPS_TD=${TD} -DDIALECT_TD=${DIALECT_TD} -DTOGSIM_TD=${TOGSIM_TD} -DTPU_TD=${TPU_TD}
          -DOUT=${PY_STAMP} -P "${CMAKE_CURRENT_SOURCE_DIR}/stamp.cmake"
  DEPENDS "${TD}" "${DIALECT_TD}" "${TOGSIM_TD}" "${TPU_TD}" "${CMAKE_CURRENT_SOURCE_DIR}/stamp.cmake"
  COMMENT "stamp _pytorchsim_stamp.py")

add_custom_target(PyTorchSimPythonOps ALL DEPENDS "${PY_OPS}" "${PY_TOGSIM_OPS}" "${PY_TPU_OPS}" "${PY_TABLE}" "${PY_STAMP}")

execute_process(COMMAND "${Python3_EXECUTABLE}" -m nanobind --cmake_dir
                OUTPUT_VARIABLE NB_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE NB_FOUND)
if(NOT NB_FOUND EQUAL 0)
  message(FATAL_ERROR "nanobind is not installed for ${Python3_EXECUTABLE}. MLIR's own "
                      "bindings want >=2.9,<3.0; install it there and re-run cmake.")
endif()
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED PATHS "${NB_CMAKE_DIR}" NO_DEFAULT_PATH)

nanobind_add_module(_pytorchsim_ext NB_STATIC
  python/PyTorchSimExtension.cpp
  lib/CAPI/Dialects.cpp
  lib/PyTorchSimDialect.cpp lib/PyTorchSimOps.cpp lib/PyTorchSimBufferize.cpp lib/TogsimDialect.cpp
  lib/PyTorchSimTpuDialect.cpp
  lib/HoistLoopInvariants.cpp lib/Effects.cpp lib/Loops.cpp
  "${GEN}/PyTorchSim/PyTorchSimOps.h.inc" "${GEN}/PyTorchSim/PyTorchSimOps.cpp.inc"
  "${GEN}/PyTorchSim/PyTorchSimOpsDialect.h.inc" "${GEN}/PyTorchSim/PyTorchSimOpsDialect.cpp.inc"
  "${GEN}/PyTorchSim/TogsimOps.h.inc" "${GEN}/PyTorchSim/TogsimOps.cpp.inc"
  "${GEN}/PyTorchSim/TogsimOpsDialect.h.inc" "${GEN}/PyTorchSim/TogsimOpsDialect.cpp.inc"
  "${GEN}/PyTorchSim/PyTorchSimTpuOps.h.inc" "${GEN}/PyTorchSim/PyTorchSimTpuOps.cpp.inc"
  "${GEN}/PyTorchSim/PyTorchSimTpuOpsDialect.h.inc" "${GEN}/PyTorchSim/PyTorchSimTpuOpsDialect.cpp.inc")
target_include_directories(_pytorchsim_ext PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include" "${GEN}" "${MLIR_INSTALL}/include")
target_link_libraries(_pytorchsim_ext PRIVATE "${MLIR_PY_CAPI}")

set_source_files_properties(
  lib/CAPI/Dialects.cpp lib/PyTorchSimDialect.cpp lib/PyTorchSimOps.cpp lib/TogsimDialect.cpp lib/PyTorchSimTpuDialect.cpp
  lib/PyTorchSimBufferize.cpp lib/HoistLoopInvariants.cpp lib/Effects.cpp lib/Loops.cpp
  PROPERTIES COMPILE_OPTIONS "-fno-rtti")
add_dependencies(_pytorchsim_ext PyTorchSimPythonOps)
