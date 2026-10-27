cmake_minimum_required(VERSION 3.20)
project(pytorchsim-dialect LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(MLIR_INSTALL "${MLIR_INSTALL}" CACHE PATH "LLVM/MLIR install prefix")
set(TRITON_SRC "${TRITON_SRC}" CACHE PATH "Triton source tree: its LinearLayout code is built in here")
if(NOT EXISTS "${TRITON_SRC}/lib/Tools/LinearLayout.cpp")
  message(FATAL_ERROR "TRITON_SRC=${TRITON_SRC} holds no lib/Tools/LinearLayout.cpp")
endif()
# Triton's lib/Tools (TritonTools), built against these bindings' MLIR: the extension answers layout
# questions with Triton's own code. It loads no dialect, so it shares nothing with libtriton.so's MLIR.
set(TRITON_TOOLS
  "${TRITON_SRC}/lib/Tools/LinearLayout.cpp" "${TRITON_SRC}/lib/Tools/LayoutUtils.cpp"
  "${TRITON_SRC}/lib/Tools/GenericSwizzling.cpp" "${TRITON_SRC}/third_party/f2reduce/f2reduce.cpp")
set(MLIR_PY_LIBS "${MLIR_INSTALL}/python_packages/mlir_core/mlir/_mlir_libs")
find_program(MLIR_TBLGEN mlir-tblgen HINTS "${MLIR_INSTALL}/bin" REQUIRED)
find_library(MLIR_PY_CAPI NAMES MLIRPythonCAPI HINTS "${MLIR_PY_LIBS}" REQUIRED)

set(GEN "${CMAKE_CURRENT_BINARY_DIR}/include")
file(MAKE_DIRECTORY "${GEN}/PyTorchSim")
set(DIALECT_TD "${CMAKE_CURRENT_SOURCE_DIR}/include/PyTorchSim/PyTorchSimDialect.td")
set(TD "${CMAKE_CURRENT_SOURCE_DIR}/include/PyTorchSim/PyTorchSimOps.td")
set(TOGSIM_TD "${CMAKE_CURRENT_SOURCE_DIR}/include/PyTorchSim/TogsimOps.td")
set(EXTERN_TD "${CMAKE_CURRENT_SOURCE_DIR}/include/PyTorchSim/PyTorchSimExternOps.td")

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
tblgen("${TD}" pytorchsim PyTorchSimEnums.h.inc        -gen-enum-decls)
tblgen("${TD}" pytorchsim PyTorchSimEnums.cpp.inc      -gen-enum-defs)
tblgen("${TD}" pytorchsim PyTorchSimAttrs.h.inc        "-gen-attrdef-decls;-attrdefs-dialect=pytorchsim")
tblgen("${TD}" pytorchsim PyTorchSimAttrs.cpp.inc      "-gen-attrdef-defs;-attrdefs-dialect=pytorchsim")
tblgen("${TOGSIM_TD}" togsim TogsimOps.h.inc          -gen-op-decls)
tblgen("${TOGSIM_TD}" togsim TogsimOps.cpp.inc        -gen-op-defs)
tblgen("${TOGSIM_TD}" togsim TogsimOpsDialect.h.inc   -gen-dialect-decls)
tblgen("${TOGSIM_TD}" togsim TogsimOpsDialect.cpp.inc -gen-dialect-defs)
tblgen("${EXTERN_TD}" pytorchsim_extern PyTorchSimExternOps.h.inc          -gen-op-decls)
tblgen("${EXTERN_TD}" pytorchsim_extern PyTorchSimExternOps.cpp.inc        -gen-op-defs)
tblgen("${EXTERN_TD}" pytorchsim_extern PyTorchSimExternOpsDialect.h.inc   -gen-dialect-decls)
tblgen("${EXTERN_TD}" pytorchsim_extern PyTorchSimExternOpsDialect.cpp.inc -gen-dialect-defs)

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
set(PY_ENUMS "${CMAKE_CURRENT_BINARY_DIR}/_pytorchsim_enum_gen.py")
add_custom_command(
  OUTPUT "${PY_ENUMS}"
  COMMAND "${MLIR_TBLGEN}" -gen-python-enum-bindings -bind-dialect=pytorchsim "${TD}"
          -I "${CMAKE_CURRENT_SOURCE_DIR}/include" -I "${MLIR_INSTALL}/include" -o "${PY_ENUMS}"
  DEPENDS "${TD}" "${DIALECT_TD}"
  COMMENT "tblgen ${PY_ENUMS}")
set(PY_TOGSIM_OPS "${CMAKE_CURRENT_BINARY_DIR}/_togsim_ops_gen.py")
python_ops("${TOGSIM_TD}" togsim "${PY_TOGSIM_OPS}")
set(PY_EXTERN_OPS "${CMAKE_CURRENT_BINARY_DIR}/_pytorchsim_extern_ops_gen.py")
python_ops("${EXTERN_TD}" pytorchsim_extern "${PY_EXTERN_OPS}")

# Every target (targets.cmake): its dialect's C++ and Python from its .td, its sources in the extension.
include("${CMAKE_CURRENT_SOURCE_DIR}/targets.cmake")
pytorchsim_include_targets("${GEN}")
set(TARGET_GENERATED "")
foreach(dialect td IN ZIP_LISTS TARGET_NAMES TARGET_TD_FILES)
  get_filename_component(stem "${td}" NAME_WE)
  tblgen("${td}" ${dialect} ${stem}.h.inc          -gen-op-decls)
  tblgen("${td}" ${dialect} ${stem}.cpp.inc        -gen-op-defs)
  tblgen("${td}" ${dialect} ${stem}Dialect.h.inc   -gen-dialect-decls)
  tblgen("${td}" ${dialect} ${stem}Dialect.cpp.inc -gen-dialect-defs)
  python_ops("${td}" ${dialect} "${CMAKE_CURRENT_BINARY_DIR}/_${dialect}_ops_gen.py")
  list(APPEND TARGET_GENERATED "${CMAKE_CURRENT_BINARY_DIR}/_${dialect}_ops_gen.py"
       "${GEN}/PyTorchSim/${stem}.h.inc" "${GEN}/PyTorchSim/${stem}.cpp.inc"
       "${GEN}/PyTorchSim/${stem}Dialect.h.inc" "${GEN}/PyTorchSim/${stem}Dialect.cpp.inc")
endforeach()
# The registration list lib/CAPI/Dialects.cpp includes: every target's header, and its dialect class.
set(TARGETS_INC "// Generated by dialect/src/CMakeLists.txt from every target; do not edit.\n")
foreach(h ${TARGET_HEADERS})
  string(APPEND TARGETS_INC "#include \"${h}\"\n")
endforeach()
string(REPLACE ";" ", " TARGET_CLASS_LIST "${TARGET_CLASSES}")
string(APPEND TARGETS_INC "#define PYTORCHSIM_TARGET_DIALECTS ${TARGET_CLASS_LIST}\n")
file(CONFIGURE OUTPUT "${GEN}/PyTorchSim/Targets.h.inc" CONTENT "${TARGETS_INC}")

find_program(LLVM_TBLGEN llvm-tblgen HINTS "${MLIR_INSTALL}/bin" REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
set(PY_TABLE "${CMAKE_CURRENT_BINARY_DIR}/_pytorchsim_operands.py")
set(TABLE_DUMPS "")
set(TABLE_ARGS "")
set(TABLE_NAMES pytorchsim ${TARGET_NAMES})
set(TABLE_TDS "${TD}" ${TARGET_TD_FILES})
foreach(dialect path IN ZIP_LISTS TABLE_NAMES TABLE_TDS)
  set(json "${CMAKE_CURRENT_BINARY_DIR}/${dialect}.ops.json")
  list(APPEND TABLE_DUMPS COMMAND "${LLVM_TBLGEN}" --dump-json "${path}"
       -I "${CMAKE_CURRENT_SOURCE_DIR}/include" -I "${MLIR_INSTALL}/include" -o "${json}")
  list(APPEND TABLE_ARGS "${dialect}=${json}")
endforeach()
add_custom_command(
  OUTPUT "${PY_TABLE}"
  ${TABLE_DUMPS}
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/operand_table.py" "${PY_TABLE}"
          ${TABLE_ARGS}
  DEPENDS "${TD}" ${TARGET_TD_FILES} "${CMAKE_CURRENT_SOURCE_DIR}/include/PyTorchSim/PyTorchSimDialect.td"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/operand_table.py"
  COMMENT "operand table _pytorchsim_operands.py")

set(PY_STAMP "${CMAKE_CURRENT_BINARY_DIR}/_pytorchsim_stamp.py")
cmake_path(SET DIALECTS_PACKAGE NORMALIZE "${CMAKE_CURRENT_SOURCE_DIR}/..")
string(REPLACE ";" "|" TARGET_TD_FILES_BARS "${TARGET_TD_FILES}")
add_custom_command(
  OUTPUT "${PY_STAMP}"
  COMMAND "${CMAKE_COMMAND}" "-DTDS=${TD}|${DIALECT_TD}|${TOGSIM_TD}|${EXTERN_TD}"
          "-DTARGET_TDS=${TARGET_TD_FILES_BARS}" "-DPACKAGE=${DIALECTS_PACKAGE}"
          -DOUT=${PY_STAMP} -P "${CMAKE_CURRENT_SOURCE_DIR}/stamp.cmake"
  DEPENDS "${TD}" "${DIALECT_TD}" "${TOGSIM_TD}" "${EXTERN_TD}" ${TARGET_TD_FILES}
          "${CMAKE_CURRENT_SOURCE_DIR}/stamp.cmake"
  COMMENT "stamp _pytorchsim_stamp.py" VERBATIM)

add_custom_target(PyTorchSimPythonOps ALL DEPENDS "${PY_OPS}" "${PY_ENUMS}" "${PY_TOGSIM_OPS}"
  "${PY_EXTERN_OPS}" ${TARGET_GENERATED}
  "${PY_TABLE}" "${PY_STAMP}")

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
  lib/PyTorchSimExternDialect.cpp lib/CrossLaneFolds.cpp ${TARGET_SOURCES} ${TARGET_FOLD_SOURCES}
  lib/HoistLoopInvariants.cpp lib/Effects.cpp lib/Loops.cpp lib/Ranges.cpp lib/LinearLayouts.cpp
  ${TRITON_TOOLS}
  "${GEN}/PyTorchSim/PyTorchSimOps.h.inc" "${GEN}/PyTorchSim/PyTorchSimOps.cpp.inc"
  "${GEN}/PyTorchSim/PyTorchSimOpsDialect.h.inc" "${GEN}/PyTorchSim/PyTorchSimOpsDialect.cpp.inc"
  "${GEN}/PyTorchSim/PyTorchSimEnums.h.inc" "${GEN}/PyTorchSim/PyTorchSimEnums.cpp.inc"
  "${GEN}/PyTorchSim/PyTorchSimAttrs.h.inc" "${GEN}/PyTorchSim/PyTorchSimAttrs.cpp.inc"
  "${GEN}/PyTorchSim/TogsimOps.h.inc" "${GEN}/PyTorchSim/TogsimOps.cpp.inc"
  "${GEN}/PyTorchSim/TogsimOpsDialect.h.inc" "${GEN}/PyTorchSim/TogsimOpsDialect.cpp.inc"
  "${GEN}/PyTorchSim/PyTorchSimExternOps.h.inc" "${GEN}/PyTorchSim/PyTorchSimExternOps.cpp.inc"
  "${GEN}/PyTorchSim/PyTorchSimExternOpsDialect.h.inc" "${GEN}/PyTorchSim/PyTorchSimExternOpsDialect.cpp.inc")
target_include_directories(_pytorchsim_ext PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include" ${TARGET_INCLUDES} "${GEN}" "${MLIR_INSTALL}/include"
  "${TRITON_SRC}/include" "${TRITON_SRC}")
target_link_libraries(_pytorchsim_ext PRIVATE "${MLIR_PY_CAPI}")

set_source_files_properties(
  lib/CAPI/Dialects.cpp lib/PyTorchSimDialect.cpp lib/PyTorchSimOps.cpp lib/TogsimDialect.cpp
  lib/PyTorchSimExternDialect.cpp lib/CrossLaneFolds.cpp ${TARGET_SOURCES} ${TARGET_FOLD_SOURCES}
  lib/PyTorchSimBufferize.cpp lib/HoistLoopInvariants.cpp lib/Effects.cpp lib/Loops.cpp lib/Ranges.cpp
  lib/LinearLayouts.cpp ${TRITON_TOOLS}
  PROPERTIES COMPILE_OPTIONS "-fno-rtti")
add_dependencies(_pytorchsim_ext PyTorchSimPythonOps)
