// The recorder that stands in for TOGSim: dlopen a TOG and print
// every callback it makes as one JSON line, so the suite can READ what it says
// instead of only proving that its C++ compiled.
//
//     tog_replay <tog.so> [shape_arg ...]
//
// The togsim_* symbols are the ones TOGSim resolves when it loads a TOG
// (include/togsim_runtime.h in its repository); this defines them and nothing else, so
// what comes out is exactly what the TOG would have told the simulator.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <algorithm>
#include <climits>
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
                           int32_t elem_bits, int32_t is_async, uint64_t sem,
                           const int64_t* read_bufs, int32_t n_read,
                           const int64_t* write_bufs, int32_t n_write, int32_t indirect) {
  printf("{\"op\": \"dma\", \"dir\": %d, \"arg\": %d, \"offset\": %llu, "
         "\"elem_bits\": %d, \"async\": %d, \"sem\": %llu, \"indirect\": %d",
         dir, arg_id, (unsigned long long)offset, elem_bits, is_async,
         (unsigned long long)sem, indirect);
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

extern "C" void togsim_memory_barrier(EmitCtx*, uint64_t sem, int64_t expected) {
  printf("{\"op\": \"barrier\", \"sem\": %llu, \"expected\": %lld",
         (unsigned long long)sem, (long long)expected);
  iv_out();
  printf("}\n");
}

// THE WORK-ITEM SCOPE IS THE CALL, so this brackets the tile function the way
// TOGSim does rather than merely recording that a dispatch happened. Like TOGSim,
// the entry only registers work-items; they run after it returns, against a
// shape_args the caller has overwritten, so a TOG that keeps the caller's buffer
// reads garbage here as it would there.
struct Item { togsim_tile_fn fn; std::vector<int64_t> iv; int64_t key; };
static std::vector<Item> g_items;

extern "C" void togsim_dispatch(EmitCtx*, togsim_tile_fn fn, int64_t* iv,
                                int32_t n_iv, int64_t key) {
  g_items.push_back({fn, std::vector<int64_t>(iv, iv + n_iv), key});
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: tog_replay <tog.so> [shape_arg ...]\n");
    return 2;
  }
  void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
  auto fn = (void (*)(EmitCtx*, int64_t*, int32_t))dlsym(h, "togsim_kernel");
  if (!fn) { fprintf(stderr, "dlsym togsim_kernel: %s\n", dlerror()); return 2; }
  std::vector<int64_t> shape;
  for (int i = 2; i < argc; i++) shape.push_back(strtoll(argv[i], nullptr, 0));
  fn(nullptr, shape.empty() ? nullptr : shape.data(), (int32_t)shape.size());
  std::fill(shape.begin(), shape.end(), INT64_MIN);
  for (Item& it : g_items) {
    g_iv = it.iv;
    printf("{\"op\": \"dispatch\", \"key\": %lld", (long long)it.key);
    iv_out();
    printf("}\n");
    it.fn(nullptr, it.iv.empty() ? nullptr : it.iv.data(), (int32_t)it.iv.size());
    g_iv.clear();
  }
  return 0;
}
