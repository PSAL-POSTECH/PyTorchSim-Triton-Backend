/* DRAM mapped from PYTORCHSIM_DRAM_BASE to the last tensor, each tensor at its address, the placement
   file's or back to back; a read argument alone in its pages is its .raw mapped there, others copied in.
   Pages are taken before a Spike launch (paths given), never under gem5. Failures exit 3, named. */
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PYTORCHSIM_PAGE 4096ULL
#define PYTORCHSIM_PAD 512ULL

static uint64_t pytorchsim_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

static long pytorchsim_sys(long n, long a, long b, long c, long d, long e, long f) {
  register long a0 asm("a0") = a, a1 asm("a1") = b, a2 asm("a2") = c, a3 asm("a3") = d;
  register long a4 asm("a4") = e, a5 asm("a5") = f, a7 asm("a7") = n;
  asm volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a7) : "memory");
  return a0;
}

static void pytorchsim_cannot_place(const char *name, uint64_t at, uint64_t n, long err) {
  fprintf(stderr, "torchsim wrapper: cannot place %s at 0x%llx, %llu bytes: errno %ld\n", name,
          (unsigned long long)at, (unsigned long long)n, -err);
  exit(3);
}

static void pytorchsim_refuse(const char *what, const char *name) {
  fprintf(stderr, "torchsim wrapper: %s %s\n", what, name);
  exit(3);
}

/* mmap(at, n, read and write, private and fixed, fd), anonymous when fd is -1; refused by name otherwise. */
static void pytorchsim_map(const char *name, uint64_t at, uint64_t n, int fd) {
  long r = pytorchsim_sys(222, (long)at, (long)n, 3, fd == -1 ? 0x32 : 0x12, fd, 0);
  if (r != (long)at) pytorchsim_cannot_place(name, at, n, r < 0 ? r : -1);
}

/* The .raw at `path` opened, refused unless n bytes long: pk asserts on a page wholly past a file's end. */
static int pytorchsim_open_raw(const char *path, uint64_t n) {
  int fd = open(path, O_RDONLY);
  if (fd == -1) pytorchsim_refuse("cannot open", path);
  if (lseek(fd, 0, SEEK_END) != (off_t)n) pytorchsim_refuse("is not as long as its argument:", path);
  return fd;
}

/* Whether a and b are the same string, read through volatile so gcc neither inlines strcmp nor vectorizes the
   loop: either brings in vfirst.m, which Spike here asserts on. */
static int pytorchsim_same(const char *a, const char *b) {
  const volatile char *x = a, *y = b;
  for (; *x && *x == *y; ++x, ++y) {
  }
  return *x == *y;
}

/* The placement file's addresses into addr, one line per argument in order, refused unless each line names its
   argument with its size and no line is left over (TOGSim refuses the same); 0 when there is no file. */
static int pytorchsim_read_placement(int n, const char *const *names, const uint64_t *nbytes,
                                     uint64_t *addr) {
  int fd = open(PYTORCHSIM_PLACEMENT_FILE, O_RDONLY);
  if (fd == -1) return 0;
  off_t len = lseek(fd, 0, SEEK_END);
  char *text = malloc(len + 1);
  if (!text) pytorchsim_refuse("has no memory to read", PYTORCHSIM_PLACEMENT_FILE);
  lseek(fd, 0, SEEK_SET);
  for (off_t got = 0, r; got < len; got += r)
    if ((r = read(fd, text + got, len - got)) <= 0)
      pytorchsim_refuse("cannot read", PYTORCHSIM_PLACEMENT_FILE);
  close(fd);
  text[len] = 0;
  char *s = text;
  for (int i = 0; i < n; ++i) {
    char name[256];
    unsigned long long a, b, align;
    char role[16];
    int used = 0;
    if (sscanf(s, " %255s %llu %llu %llu %15s%n", name, &a, &b, &align, role, &used) != 5)
      pytorchsim_refuse("has no line `name addr bytes align role` for", names[i]);
    if (!pytorchsim_same(name, names[i])) pytorchsim_refuse("names another argument in the line of", names[i]);
    if (b != nbytes[i]) pytorchsim_refuse("gives another size in the line of", names[i]);
    addr[i] = a;
    s += used;
  }
  for (; *s; ++s)
    if (*s != ' ' && *s != '\t' && *s != '\r' && *s != '\n')
      pytorchsim_refuse("lists more arguments than the kernel has:", PYTORCHSIM_PLACEMENT_FILE);
  free(text);
  return 1;
}

/* The bytes of `path` copied to `to`, through a map of it at `scratch` that is taken down again. */
static void pytorchsim_copy_raw(const char *name, uint64_t to, uint64_t n, const char *path,
                                uint64_t scratch) {
  int fd = pytorchsim_open_raw(path, n);
  pytorchsim_map(name, scratch, n, fd);
  close(fd);
  memcpy((void *)to, (const void *)scratch, n);
  pytorchsim_sys(215, (long)scratch, (long)n, 0, 0, 0, 0);
}

/* The region from args[order[k]] on, every argument whose pages meet it: its end in *hi, its last k. */
static int pytorchsim_region(int n, const int *order, int k, const uint64_t *addr, const uint64_t *span,
                             uint64_t *hi) {
  *hi = pytorchsim_up(addr[order[k]] + span[order[k]], PYTORCHSIM_PAGE);
  while (k + 1 < n && addr[order[k + 1]] / PYTORCHSIM_PAGE * PYTORCHSIM_PAGE < *hi) {
    ++k;
    uint64_t end = pytorchsim_up(addr[order[k]] + span[order[k]], PYTORCHSIM_PAGE);
    if (end > *hi) *hi = end;
  }
  return k;
}

/* at[i], where argument i sits, mapped; `paths` are the .raw files in argument order, or NULL under gem5. */
void pytorchsim_place_tensors(int n, const char *const *names, const uint64_t *nbytes,
                              const unsigned char *is_read, char *const *paths, void **at) {
  uint64_t addr[n], span[n], from_zero = 0, top = 0;
  int order[n], m = 0;
  int given = pytorchsim_read_placement(n, names, nbytes, addr);
  for (int i = 0; i < n; ++i) {
    span[i] = pytorchsim_up(nbytes[i], PYTORCHSIM_PAD);
    if (!given) addr[i] = from_zero;
    from_zero = pytorchsim_up(from_zero + span[i], PYTORCHSIM_FROM_ZERO_ALIGNMENT);
    at[i] = (void *)(uintptr_t)(PYTORCHSIM_DRAM_BASE + addr[i]);
    if (!span[i]) continue;
    uint64_t end = pytorchsim_up(addr[i] + span[i], PYTORCHSIM_PAGE);
    if (end > top) top = end;
    int k = m++;
    for (; k > 0 && addr[order[k - 1]] > addr[i]; --k) order[k] = order[k - 1];
    order[k] = i;
  }
  uint64_t base = PYTORCHSIM_DRAM_BASE, scratch = base + top;
  fprintf(stderr, "torchsim wrapper: %d tensors %s\n", n,
          given ? "at the addresses " PYTORCHSIM_PLACEMENT_FILE " gives" : "laid out back to back");
  if (top) pytorchsim_map("the DRAM below the last tensor", base, top, -1);
  for (int k = 0; k < m; ++k) {
    uint64_t hi;
    int first = k, i = order[k];
    k = pytorchsim_region(m, order, k, addr, span, &hi);
    uint64_t lo = addr[i] / PYTORCHSIM_PAGE * PYTORCHSIM_PAGE;
    if (first == k && paths && is_read[i] && lo == addr[i]) {
      int fd = pytorchsim_open_raw(paths[i], nbytes[i]);
      pytorchsim_map(names[i], base + lo, nbytes[i], fd);
      close(fd);
    } else {
      for (int j = 0; j < n; ++j)
        if (paths && is_read[j] && span[j] && addr[j] >= lo && addr[j] < hi)
          pytorchsim_copy_raw(names[j], base + addr[j], nbytes[j], paths[j], scratch);
    }
    if (paths)
      for (uint64_t p = base + lo; p < base + hi; p += PYTORCHSIM_PAGE) {
        volatile char *v = (volatile char *)(uintptr_t)p;
        *v = *v;
      }
  }
  for (int i = 0; i < n; ++i)
    fprintf(stderr, "torchsim wrapper: %s at 0x%llx, %llu bytes\n", names[i],
            (unsigned long long)(uintptr_t)at[i], (unsigned long long)nbytes[i]);
}
