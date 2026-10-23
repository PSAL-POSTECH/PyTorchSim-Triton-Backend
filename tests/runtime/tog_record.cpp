// The recorder that stands in for TOGSim: dlopen a TOG and print every callback it makes as one JSON line,
// so the suite can READ what it says; the runtime attributes it reads come from the functional run's files,
// checked as TOGSim checks them (its message texts, prefixed "tog_record: "), every gap an exit 1.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#define TOGSIM_HOST
#include "togsim_runtime.h"

static const int32_t ABI = 17;
static_assert(TOGSIM_ABI_VERSION == ABI, "tog_record implements TOG ABI 17");
static const uint32_t PROGRAMS_MAGIC = 0x50415354, RECORD_MAGIC = 0x41525354;

struct Record {
  int32_t rank;
  std::vector<int64_t> extents, values;
};

static std::vector<int64_t> g_iv;
static size_t g_item = 0;
static int64_t g_key = -1;
static bool g_in_item = false;
static std::string g_so, g_dir;
static int32_t g_declared = 0;
static const int32_t* g_rank = nullptr;
static std::map<std::pair<int64_t, int32_t>, Record> g_records;
static std::set<int64_t> g_programs, g_dispatched;

[[noreturn]] static void refuse(const std::string& what) {
  fprintf(stderr, "tog_record: %s\n", what.c_str());
  exit(1);
}

static void arr(const char* name, const int64_t* p, int64_t n) {
  printf(", \"%s\": [", name);
  for (int64_t i = 0; i < n; i++) printf(i ? ",%lld" : "%lld", (long long)(p ? p[i] : 0));
  printf("]");
}

static void iv_out() {
  printf(", \"iv\": [");
  for (size_t i = 0; i < g_iv.size(); i++) printf(i ? ",%lld" : "%lld", (long long)g_iv[i]);
  printf("]");
}

static std::vector<char> slurp(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) refuse(path + " does not open");
  std::vector<char> out;
  char buf[65536];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, f)) > 0) out.insert(out.end(), buf, buf + n);
  fclose(f);
  return out;
}

template <class T>
static T at(const std::vector<char>& b, size_t off) {
  T v;
  memcpy(&v, b.data() + off, sizeof v);
  return v;
}

static std::string tog() { return "TOG " + g_so + ": "; }

// The attribute dir as TOGSim loads it at launch: programs.bin and every record, each held to its name, its
// declaration and its size; what needs the dispatches is checked as they happen and after.
static void load_attributes() {
  if (g_declared > 0 && g_dir.empty())
    refuse(tog() + "it declares " + std::to_string(g_declared) +
           " runtime attribute(s) and was launched with no attribute_path");
  if (g_dir.empty()) return;
  DIR* d = opendir(g_dir.c_str());
  if (!d) refuse(tog() + "attribute_path " + g_dir + " does not open");
  std::vector<std::string> names;
  while (dirent* e = readdir(d))
    if (strcmp(e->d_name, ".") && strcmp(e->d_name, "..")) names.push_back(e->d_name);
  closedir(d);
  bool have_programs = false;
  std::vector<std::string> records;
  for (const std::string& n : names) {
    long long key;
    int id;
    char tail[8] = {};
    if (n == "programs.bin")
      have_programs = true;
    else if (sscanf(n.c_str(), "attr_%lld_%d%7s", &key, &id, tail) == 3 && !strcmp(tail, ".bin") &&
             n == "attr_" + std::to_string(key) + "_" + std::to_string(id) + ".bin")
      records.push_back(n);
    else
      refuse(tog() + g_dir + "/" + n + " is neither programs.bin nor attr_<key>_<id>.bin");
  }
  if (g_declared == 0) {
    if (!records.empty())
      refuse(tog() + "it declares no runtime attribute and " + g_dir + " holds " + std::to_string(records.size()) +
             " record(s)");
    return;
  }
  if (!have_programs) refuse(tog() + g_dir + " has no programs.bin: the functional run recorded no program");
  const std::string pf = g_dir + "/programs.bin";
  std::vector<char> p = slurp(pf);
  if (p.size() < 16 || at<uint32_t>(p, 0) != PROGRAMS_MAGIC || at<uint32_t>(p, 4) != (uint32_t)ABI)
    refuse(tog() + pf + ": not a runtime attribute file of ABI 17");
  const uint64_t n = at<uint64_t>(p, 8);
  if (p.size() != 16 + 8 * n)
    refuse(tog() + pf + " is " + std::to_string(p.size()) + " bytes, its header says " + std::to_string(16 + 8 * n));
  for (uint64_t k = 0; k < n; k++) g_programs.insert((int64_t)at<uint64_t>(p, 16 + 8 * k));
  for (const std::string& r : records) {
    const std::string path = g_dir + "/" + r;
    std::vector<char> b = slurp(path);
    if (b.size() < 24 || at<uint32_t>(b, 0) != RECORD_MAGIC || at<uint32_t>(b, 4) != (uint32_t)ABI)
      refuse(tog() + path + ": not a runtime attribute file of ABI 17");
    const int32_t id = at<int32_t>(b, 8), rank = at<int32_t>(b, 12);
    const int64_t key = (int64_t)at<uint64_t>(b, 16);
    if (r != "attr_" + std::to_string(key) + "_" + std::to_string(id) + ".bin")
      refuse(tog() + path + " holds key " + std::to_string(key) + " id " + std::to_string(id));
    if (id < 0 || id >= g_declared)
      refuse(tog() + path + " records attribute " + std::to_string(id) + ", which the TOG does not declare");
    if (rank != g_rank[id])
      refuse(tog() + path + " records attribute " + std::to_string(id) + " at rank " + std::to_string(rank) +
             ", the TOG declares rank " + std::to_string(g_rank[id]));
    if (b.size() < 24 + 8 * (size_t)rank)
      refuse(tog() + path + " is " + std::to_string(b.size()) + " bytes, its header says " +
             std::to_string(24 + 8 * rank));
    Record rec{rank, {}, {}};
    int64_t count = 1;
    for (int32_t k = 0; k < rank; k++) {
      rec.extents.push_back(at<int64_t>(b, 24 + 8 * k));
      count *= rec.extents.back();
    }
    const size_t want = 24 + 8 * (size_t)rank + 8 * (size_t)(count < 0 ? 0 : count);
    if (count < 0 || b.size() != want)
      refuse(tog() + path + " is " + std::to_string(b.size()) + " bytes, its header says " + std::to_string(want));
    if (!g_programs.count(key))
      refuse(tog() + path + " records key " + std::to_string(key) + ", which programs.bin does not list");
    for (int64_t k = 0; k < count; k++) rec.values.push_back(at<int64_t>(b, 24 + 8 * rank + 8 * k));
    g_records[{key, id}] = std::move(rec);
  }
}

extern "C" const int64_t* togsim_runtime_attribute(EmitCtx*, int32_t id, int32_t* rank, const int64_t** extents) {
  if (!g_in_item) refuse("an attribute read outside a work-item");
  auto r = g_records.find({g_key, id});
  if (r == g_records.end())
    refuse(tog() + "work-item " + std::to_string(g_item) + " (key " + std::to_string(g_key) + ") reads attribute " +
           std::to_string(id) + ", which " + g_dir + " does not record for that key");
  *rank = r->second.rank;
  *extents = r->second.extents.data();
  printf("{\"op\": \"attr\", \"id\": %d", id);
  arr("extents", r->second.extents.data(), r->second.rank);
  iv_out();
  printf("}\n");
  return r->second.values.data();
}

extern "C" void togsim_dma(EmitCtx*, int32_t dir, int32_t arg_id, uint64_t offset, int32_t ndim, const int64_t* dims,
                           const int64_t* strides, int32_t elem_bits, int32_t is_async, uint64_t sem,
                           const int64_t* read_bufs, int32_t n_read, const int64_t* write_bufs, int32_t n_write,
                           const int64_t* element_offsets, int64_t n_offsets) {
  printf("{\"op\": \"dma\", \"dir\": %d, \"arg\": %d, \"offset\": %llu, \"elem_bits\": %d, \"async\": %d, "
         "\"sem\": %llu",
         dir, arg_id, (unsigned long long)offset, elem_bits, is_async, (unsigned long long)sem);
  arr("dims", dims, ndim);
  arr("strides", strides, strides ? ndim : 0);
  arr("read", read_bufs, n_read);
  arr("write", write_bufs, n_write);
  if (element_offsets) arr("offsets", element_offsets, n_offsets);
  iv_out();
  printf("}\n");
}

extern "C" void togsim_compute(EmitCtx*, uint64_t tile_id, int32_t compute_type, int32_t ndim, const int64_t* dims,
                               const int64_t* read_bufs, int32_t n_read, const int64_t* write_bufs, int32_t n_write) {
  printf("{\"op\": \"compute\", \"tile\": %llu, \"type\": %d", (unsigned long long)tile_id, compute_type);
  arr("dims", dims, ndim);
  arr("read", read_bufs, n_read);
  arr("write", write_bufs, n_write);
  iv_out();
  printf("}\n");
}

extern "C" void togsim_memory_barrier(EmitCtx*, uint64_t sem, int64_t expected) {
  printf("{\"op\": \"barrier\", \"sem\": %llu, \"expected\": %lld", (unsigned long long)sem, (long long)expected);
  iv_out();
  printf("}\n");
}

// The work-item scope is the call, as in TOGSim: bracketed here, run under the key it is dispatched with, and
// numbered by dispatch order as TOGSim numbers its items.
extern "C" void togsim_dispatch(EmitCtx* ctx, togsim_tile_fn fn, int64_t* iv, int32_t n_iv, int64_t key) {
  g_iv.assign(iv, iv + n_iv);
  printf("{\"op\": \"dispatch\", \"key\": %lld", (long long)key);
  iv_out();
  printf("}\n");
  if (g_declared > 0 && !g_dispatched.insert(key).second)
    refuse(tog() + "two work-items are dispatched under key " + std::to_string(key));
  g_key = key;
  g_in_item = true;
  fn(ctx, iv, n_iv);
  g_in_item = false;
  g_iv.clear();
  ++g_item;
}

// The run's and the recording's key sets are one: a key only one of them names is a launch the other did not make.
static void check_programs() {
  if (g_declared == 0) return;
  for (int64_t k : g_dispatched)
    if (!g_programs.count(k))
      refuse(tog() + "work-item key " + std::to_string(k) + " was dispatched and " + g_dir +
             "/programs.bin does not list it");
  for (int64_t k : g_programs)
    if (!g_dispatched.count(k))
      refuse(tog() + g_dir + "/programs.bin lists key " + std::to_string(k) +
             ", which no work-item was dispatched under");
}

int main(int argc, char** argv) {
  int a = 1;
  if (a + 1 < argc && std::string(argv[a]) == "--attributes") {
    g_dir = argv[a + 1];
    a += 2;
  }
  if (argc - a < 1) {
    fprintf(stderr, "usage: tog_record [--attributes DIR] <tog.so> [shape_arg ...]\n");
    return 2;
  }
  g_so = argv[a];
  void* h = dlopen(argv[a], RTLD_NOW | RTLD_LOCAL);
  if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
  auto fn = (void (*)(EmitCtx*, int64_t*, int32_t))dlsym(h, "togsim_kernel");
  if (!fn) { fprintf(stderr, "dlsym togsim_kernel: %s\n", dlerror()); return 2; }
  auto abi = (const int32_t*)dlsym(h, "togsim_abi_version");
  if (!abi || *abi != ABI)
    refuse("refusing TOG " + g_so + ": it was built for TOG ABI " + (abi ? std::to_string(*abi) : "none") +
           ", this recorder is " + std::to_string(ABI));
  auto count = (const int32_t*)dlsym(h, "togsim_runtime_attribute_count");
  g_rank = (const int32_t*)dlsym(h, "togsim_runtime_attribute_rank");
  if (!count || !g_rank)
    refuse("refusing TOG " + g_so +
           ": it states no runtime attribute table (togsim_runtime_attribute_count / togsim_runtime_attribute_rank)");
  g_declared = *count;
  printf("{\"op\": \"attributes\", \"ranks\": [");
  for (int32_t i = 0; i < g_declared; i++) printf(i ? ",%d" : "%d", g_rank[i]);
  printf("]}\n");
  load_attributes();
  std::vector<int64_t> shape;
  for (int i = a + 1; i < argc; i++) shape.push_back(strtoll(argv[i], nullptr, 0));
  fn(nullptr, shape.empty() ? nullptr : shape.data(), (int32_t)shape.size());
  check_programs();
  return 0;
}
