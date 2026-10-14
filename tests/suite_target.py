"""The machine the suite, the image and a sourced env.sh compile for when TORCHSIM_COMPILE_TARGET
does not say. tests/run.py and tests/torchsim-compile read NAME; setup/Dockerfile and
setup/env.sh.in cannot import it, and target_cases.default_target_is_one_statement holds them to it."""
NAME = "tpuv6e"
