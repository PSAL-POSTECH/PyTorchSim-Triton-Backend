"""Empty on purpose: the bindings import a module for an op's dialect each time they wrap an op and cache only
a found one. `pytorchsim_tpu`'s classes are targets/tpu/pytorchsim_tpu.py's, off the search prefix, so without
this module every op costs two failed imports."""
