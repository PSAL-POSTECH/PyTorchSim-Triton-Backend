# torchsim-triton-ext

Everything the pytorchsim-triton-compiler route adds to Triton, as one plugin `.so` loaded into an
unmodified Triton (v3.7.0) through `TRITON_PLUGIN_PATHS`. Triton is built once, with its symbols
exposed (`TRITON_EXT_ENABLED=ON`); this extension builds against that build tree and links to its
`libtriton.so`.

- `lib/` -- the TTGIR passes (anchor-layouts with tt.dot -> torchsim.dot, strip-pins, a copy of
  remove-layout-conversions, a copy of convert-triton-to-tritongpu with the lane machine's default
  layout), regularize-layouts, and the plugin entry point.
- `triton_shared/` -- the parts of triton_shared the route uses (its conversions, analyses and
  dialects, MIT, copied at 48afb14); the TorchSim dialect is the compiler's `../dialects/src`. MLIR's linalg, tensor, bufferization and ptr dialects and remove-dead-values,
  which libtriton.so leaves out, linked after it.
- `python/torchsim_triton_backend` -- the Triton backend stage 1 targets (TTIR, nothing further),
  linked into Triton's `python/triton/backends/torchsim`.

Built by `../setup/toolchain.sh` (`triton` and `ext` steps).
