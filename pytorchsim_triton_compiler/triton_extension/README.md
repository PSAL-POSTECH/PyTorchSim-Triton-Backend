# torchsim-triton-ext

Everything the pytorchsim-triton-compiler route adds to Triton, as one plugin `.so` loaded into an
Triton v3.8.0 (one edit: its tensor numel limit, see setup/versions.env) through `TRITON_PLUGIN_PATHS`. Triton is built once, with its symbols
exposed (`TRITON_EXT_ENABLED=ON`); this extension builds against that build tree and links to its
`libtriton.so`.

- `lib/` -- the TTGIR passes (anchor-layouts with tt.dot -> torchsim.dot, strip-pins, a copy of
  remove-layout-conversions, a copy of convert-triton-to-tritongpu with the lane machine's default
  layout), regularize-layouts, and the plugin entry point.
- `include/TorchsimExt/{Analysis,AnalysisStructured,Conversion,Dialect,Utils}` and the same under
  `lib/` -- the lowering from TTGIR to linalg/memref (the tts, tptr and ttx dialects, the pointer and
  mask analyses, the conversions). They began as a copy of microsoft/triton-shared at 48afb14 (MIT,
  `LICENSE.triton-shared`) and are maintained here. The TorchSim dialect is the compiler's
  `../dialects/src`. MLIR's linalg, tensor, bufferization and ptr dialects and remove-dead-values,
  which libtriton.so leaves out, linked after it.
- `python/torchsim_triton_backend` -- the Triton backend the frontend targets (TTIR, nothing further),
  linked into Triton's `python/triton/backends/torchsim`.

Built by `../setup/toolchain.sh` (`triton` and `ext` steps).
