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
set(TD "${CMAKE_CURRENT_SOURCE_DIR}/include/PyTorchSim/PyTorchSimOps.td")

function(tblgen out args)
  add_custom_command(
    OUTPUT "${GEN}/PyTorchSim/${out}"
    COMMAND "${MLIR_TBLGEN}" ${args} "${TD}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include" -I "${MLIR_INSTALL}/include"
            -o "${GEN}/PyTorchSim/${out}"
    DEPENDS "${TD}" "${CMAKE_CURRENT_SOURCE_DIR}/include/PyTorchSim/PyTorchSimDialect.td"
    COMMENT "tblgen ${out}")
endfunction()

tblgen(PyTorchSimOps.h.inc          -gen-op-decls)
tblgen(PyTorchSimOps.cpp.inc        -gen-op-defs)
tblgen(PyTorchSimOpsDialect.h.inc   -gen-dialect-decls)
tblgen(PyTorchSimOpsDialect.cpp.inc -gen-dialect-defs)

set(PY_OPS "${CMAKE_CURRENT_BINARY_DIR}/_pytorchsim_ops_gen.py")
add_custom_command(
  OUTPUT "${PY_OPS}"
  COMMAND "${MLIR_TBLGEN}" -gen-python-op-bindings -bind-dialect=pytorchsim "${TD}"
          -I "${CMAKE_CURRENT_SOURCE_DIR}/include" -I "${MLIR_INSTALL}/include"
          -o "${PY_OPS}"
  DEPENDS "${TD}" "${CMAKE_CURRENT_SOURCE_DIR}/include/PyTorchSim/PyTorchSimDialect.td"
  COMMENT "tblgen _pytorchsim_ops_gen.py")

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
  COMMAND "${CMAKE_COMMAND}" -DOPS_TD=${TD}
          -DDIALECT_TD=${CMAKE_CURRENT_SOURCE_DIR}/include/PyTorchSim/PyTorchSimDialect.td
          -DOUT=${PY_STAMP} -P "${CMAKE_CURRENT_SOURCE_DIR}/stamp.cmake"
  DEPENDS "${TD}" "${CMAKE_CURRENT_SOURCE_DIR}/include/PyTorchSim/PyTorchSimDialect.td"
          "${CMAKE_CURRENT_SOURCE_DIR}/stamp.cmake"
  COMMENT "stamp _pytorchsim_stamp.py")

add_custom_target(PyTorchSimPythonOps ALL DEPENDS "${PY_OPS}" "${PY_TABLE}" "${PY_STAMP}")

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
  lib/PyTorchSimDialect.cpp lib/PyTorchSimOps.cpp lib/PyTorchSimBufferize.cpp
  lib/HoistLoopInvariants.cpp lib/Effects.cpp lib/Loops.cpp
  "${GEN}/PyTorchSim/PyTorchSimOps.h.inc" "${GEN}/PyTorchSim/PyTorchSimOps.cpp.inc"
  "${GEN}/PyTorchSim/PyTorchSimOpsDialect.h.inc" "${GEN}/PyTorchSim/PyTorchSimOpsDialect.cpp.inc")
target_include_directories(_pytorchsim_ext PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include" "${GEN}" "${MLIR_INSTALL}/include")
target_link_libraries(_pytorchsim_ext PRIVATE "${MLIR_PY_CAPI}")

set_source_files_properties(
  lib/CAPI/Dialects.cpp lib/PyTorchSimDialect.cpp lib/PyTorchSimOps.cpp
  lib/PyTorchSimBufferize.cpp lib/HoistLoopInvariants.cpp lib/Effects.cpp lib/Loops.cpp
  PROPERTIES COMPILE_OPTIONS "-fno-rtti")
add_dependencies(_pytorchsim_ext PyTorchSimPythonOps)
