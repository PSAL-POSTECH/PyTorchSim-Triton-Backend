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
file(MAKE_DIRECTORY "${GEN}/TorchSim")
set(TD "${CMAKE_CURRENT_SOURCE_DIR}/include/TorchSim/TorchSimOps.td")

function(tblgen out args)
  add_custom_command(
    OUTPUT "${GEN}/TorchSim/${out}"
    COMMAND "${MLIR_TBLGEN}" ${args} "${TD}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include" -I "${MLIR_INSTALL}/include"
            -o "${GEN}/TorchSim/${out}"
    DEPENDS "${TD}" "${CMAKE_CURRENT_SOURCE_DIR}/include/TorchSim/TorchSimDialect.td"
    COMMENT "tblgen ${out}")
endfunction()

tblgen(TorchSimOps.h.inc          -gen-op-decls)
tblgen(TorchSimOps.cpp.inc        -gen-op-defs)
tblgen(TorchSimOpsDialect.h.inc   -gen-dialect-decls)
tblgen(TorchSimOpsDialect.cpp.inc -gen-dialect-defs)

set(PY_OPS "${CMAKE_CURRENT_BINARY_DIR}/_torchsim_ops_gen.py")
add_custom_command(
  OUTPUT "${PY_OPS}"
  COMMAND "${MLIR_TBLGEN}" -gen-python-op-bindings -bind-dialect=torchsim "${TD}"
          -I "${CMAKE_CURRENT_SOURCE_DIR}/include" -I "${MLIR_INSTALL}/include"
          -o "${PY_OPS}"
  DEPENDS "${TD}" "${CMAKE_CURRENT_SOURCE_DIR}/include/TorchSim/TorchSimDialect.td"
  COMMENT "tblgen _torchsim_ops_gen.py")

find_program(LLVM_TBLGEN llvm-tblgen HINTS "${MLIR_INSTALL}/bin" REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
set(PY_TABLE "${CMAKE_CURRENT_BINARY_DIR}/_torchsim_operands.py")
add_custom_command(
  OUTPUT "${PY_TABLE}"
  COMMAND "${LLVM_TBLGEN}" --dump-json "${TD}"
          -I "${CMAKE_CURRENT_SOURCE_DIR}/include" -I "${MLIR_INSTALL}/include"
          -o "${CMAKE_CURRENT_BINARY_DIR}/TorchSimOps.json"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/operand_table.py"
          "${CMAKE_CURRENT_BINARY_DIR}/TorchSimOps.json" "${PY_TABLE}"
  DEPENDS "${TD}" "${CMAKE_CURRENT_SOURCE_DIR}/include/TorchSim/TorchSimDialect.td"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/operand_table.py"
  COMMENT "operand table _torchsim_operands.py")

set(PY_STAMP "${CMAKE_CURRENT_BINARY_DIR}/_torchsim_stamp.py")
add_custom_command(
  OUTPUT "${PY_STAMP}"
  COMMAND "${CMAKE_COMMAND}" -DOPS_TD=${TD}
          -DDIALECT_TD=${CMAKE_CURRENT_SOURCE_DIR}/include/TorchSim/TorchSimDialect.td
          -DOUT=${PY_STAMP} -P "${CMAKE_CURRENT_SOURCE_DIR}/stamp.cmake"
  DEPENDS "${TD}" "${CMAKE_CURRENT_SOURCE_DIR}/include/TorchSim/TorchSimDialect.td"
          "${CMAKE_CURRENT_SOURCE_DIR}/stamp.cmake"
  COMMENT "stamp _torchsim_stamp.py")

add_custom_target(TorchSimPythonOps ALL DEPENDS "${PY_OPS}" "${PY_TABLE}" "${PY_STAMP}")

execute_process(COMMAND "${Python3_EXECUTABLE}" -m nanobind --cmake_dir
                OUTPUT_VARIABLE NB_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE NB_FOUND)
if(NOT NB_FOUND EQUAL 0)
  message(FATAL_ERROR "nanobind is not installed for ${Python3_EXECUTABLE}. MLIR's own "
                      "bindings want >=2.9,<3.0; install it there and re-run cmake.")
endif()
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED PATHS "${NB_CMAKE_DIR}" NO_DEFAULT_PATH)

nanobind_add_module(_torchsim_ext NB_STATIC
  python/TorchSimExtension.cpp
  lib/CAPI/Dialects.cpp
  lib/TorchSimDialect.cpp lib/TorchSimOps.cpp lib/TorchSimBufferize.cpp
  "${GEN}/TorchSim/TorchSimOps.h.inc" "${GEN}/TorchSim/TorchSimOps.cpp.inc"
  "${GEN}/TorchSim/TorchSimOpsDialect.h.inc" "${GEN}/TorchSim/TorchSimOpsDialect.cpp.inc")
target_include_directories(_torchsim_ext PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include" "${GEN}" "${MLIR_INSTALL}/include")
target_link_libraries(_torchsim_ext PRIVATE "${MLIR_PY_CAPI}")

set_source_files_properties(
  lib/CAPI/Dialects.cpp lib/TorchSimDialect.cpp lib/TorchSimOps.cpp
  lib/TorchSimBufferize.cpp
  PROPERTIES COMPILE_OPTIONS "-fno-rtti")
add_dependencies(_torchsim_ext TorchSimPythonOps)
