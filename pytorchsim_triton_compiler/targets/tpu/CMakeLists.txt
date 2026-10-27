# The tpu target's dialect, pytorchsim_tpu, registered with the dialect build and the Triton plugin
# (dialects/src/targets.cmake, pytorchsim_target): its .td, its C++, the header and class that declare it, and
# what its XLU reduces across the lanes, which the plugin's anchor reads too.
set(IR "Dialect/PyTorchSimTpu/IR")
pytorchsim_target(pytorchsim_tpu
  INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include"
  TD "${CMAKE_CURRENT_SOURCE_DIR}/include/${IR}/PyTorchSimTpuOps.td"
  SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/lib/${IR}/PyTorchSimTpuDialect.cpp"
  HEADER ${IR}/PyTorchSimTpuDialect.h
  CLASS ::pytorchsim::tpu::PyTorchSimTpuDialect
  FOLDS ::pytorchsim::tpu::crossLaneFolds
  FOLDS_HEADER ${IR}/PyTorchSimTpuFolds.h
  FOLD_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/lib/${IR}/PyTorchSimTpuFolds.cpp")
