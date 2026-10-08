// The recorder that stands in for TOGSim: dlopen a trace producer and print
// every callback it makes as one JSON line, so the suite can READ the trace
// instead of only proving that its C++ compiled.
//
//     trace_replay <trace.so> [shape_arg ...]
//
// The togsim_* symbols are the ones TOGSim resolves when it loads the producer
// (TOGSim/include/togsim_runtime.h); this defines them and nothing else, so
// what comes out is exactly what the producer would have told the simulator.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <vector>

extern "C" {
typedef struct EmitCtx EmitCtx;
typedef void (*togsim_tile_fn)(EmitCtx*, int64_t*, int32_t);
}

// The work item being dispatched, so every event says which one it belongs to.
static std::vector<int64_t> g_iv;

static void arr(const char* name, const int64_t* p, int32_t n) {
  printf(", \"%s\": [", name);
  for (int32_t i = 0; i < n; i++)
    printf(i ? ",%lld" : "%lld", (long long)(p ? p[i] : 0));
  printf("]");
}

static void iv_out() {
  printf(", \"iv\": [");
  for (size_t i = 0; i < g_iv.size(); i++)
    printf(i ? ",%lld" : "%lld", (long long)g_iv[i]);
  printf("]");
}

extern "C" void togsim_dma(EmitCtx*, int32_t dir, int32_t arg_id, uint64_t offset,
                           int32_t ndim, const int64_t* dims, const int64_t* strides,
                           int32_t elem_bits, int32_t is_async, int32_t tag_id,
                           uint64_t tag_slot, const int64_t* read_bufs, int32_t n_read,
                           const int64_t* write_bufs, int32_t n_write, int32_t indirect) {
  printf("{\"op\": \"dma\", \"dir\": %d, \"arg\": %d, \"offset\": %llu, "
         "\"elem_bits\": %d, \"async\": %d, \"tag\": [%d,%llu], \"indirect\": %d",
         dir, arg_id, (unsigned long long)offset, elem_bits, is_async, tag_id,
         (unsigned long long)tag_slot, indirect);
  arr("dims", dims, ndim);
  arr("strides", strides, strides ? ndim : 0);
  arr("read", read_bufs, n_read);
  arr("write", write_bufs, n_write);
  iv_out();
  printf("}\n");
}

extern "C" void togsim_compute(EmitCtx*, uint64_t tile_id, int32_t compute_type,
                               int32_t ndim, const int64_t* dims,
                               const int64_t* read_bufs, int32_t n_read,
                               const int64_t* write_bufs, int32_t n_write) {
  printf("{\"op\": \"compute\", \"tile\": %llu, \"type\": %d",
         (unsigned long long)tile_id, compute_type);
  arr("dims", dims, ndim);
  arr("read", read_bufs, n_read);
  arr("write", write_bufs, n_write);
  iv_out();
  printf("}\n");
}

extern "C" void togsim_memory_barrier(EmitCtx*, int32_t tag_id, uint64_t tag_slot,
                                      const int64_t* write_bufs, int32_t n_write) {
  printf("{\"op\": \"barrier\", \"tag\": [%d,%llu]", tag_id,
         (unsigned long long)tag_slot);
  arr("write", write_bufs, n_write);
  iv_out();
  printf("}\n");
}

// THE WORK-ITEM SCOPE IS THE CALL, so this brackets the tile function the way
// TOGSim does rather than merely recording that a dispatch happened.
extern "C" void togsim_dispatch(EmitCtx* ctx, togsim_tile_fn fn, int64_t* iv,
                                int32_t n_iv, int64_t key) {
  g_iv.assign(iv, iv + n_iv);
  printf("{\"op\": \"dispatch\", \"key\": %lld", (long long)key);
  iv_out();
  printf("}\n");
  fn(ctx, iv, n_iv);
  g_iv.clear();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: trace_replay <trace.so> [shape_arg ...]\n");
    return 2;
  }
  void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
  auto fn = (void (*)(EmitCtx*, int64_t*, int32_t))dlsym(h, "togsim_kernel");
  if (!fn) { fprintf(stderr, "dlsym togsim_kernel: %s\n", dlerror()); return 2; }
  std::vector<int64_t> shape;
  for (int i = 2; i < argc; i++) shape.push_back(strtoll(argv[i], nullptr, 0));
  fn(nullptr, shape.empty() ? nullptr : shape.data(), (int32_t)shape.size());
  return 0;
}
