/* The runtime attribute records a kernel writes, streamed through pk's file calls to one file: a header
   (magic, version, the lowered module's table hash), then per record its program key, id, byte count and
   bytes, padded to 8. What the bytes mean is the manifest's; any failure is named and exits 1. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern const uint64_t torchsim_runtime_attribute_table_hash;
uint64_t torchsim_runtime_attribute_key;
static FILE *torchsim_runtime_attribute_file;

static void torchsim_runtime_attribute_fail(const char *what) {
  fputs("torchsim runtime attribute: ", stderr);
  fputs(what, stderr);
  fputs("\n", stderr);
  exit(1);
}

static void torchsim_runtime_attribute_put(const void *data, uint64_t nbytes) {
  if (nbytes && fwrite(data, 1, nbytes, torchsim_runtime_attribute_file) != nbytes)
    torchsim_runtime_attribute_fail("cannot write the record file");
}

void torchsim_runtime_attribute_open(const char *path) {
  static const uint32_t version = TORCHSIM_RUNTIME_ATTRIBUTE_VERSION;
  enum { BUFFER = 1 << 22 };
  if (!path) torchsim_runtime_attribute_fail("no record file path follows the tensor paths");
  if (torchsim_runtime_attribute_file) torchsim_runtime_attribute_fail("the record file is open already");
  torchsim_runtime_attribute_file = fopen(path, "wb");
  if (!torchsim_runtime_attribute_file) torchsim_runtime_attribute_fail("cannot open the record file");
  char *buffer = malloc(BUFFER);
  if (!buffer || setvbuf(torchsim_runtime_attribute_file, buffer, _IOFBF, BUFFER))
    torchsim_runtime_attribute_fail("cannot give the record file its buffer");
  torchsim_runtime_attribute_put("TSRA", 4);
  torchsim_runtime_attribute_put(&version, 4);
  torchsim_runtime_attribute_put(&torchsim_runtime_attribute_table_hash, 8);
}

void torchsim_runtime_attribute_write(uint32_t id, const void *data, uint64_t nbytes) {
  static const uint8_t pad[8];
  uint32_t head[4];
  if (!torchsim_runtime_attribute_file) torchsim_runtime_attribute_fail("a record before the record file is open");
  if (nbytes > UINT32_MAX) torchsim_runtime_attribute_fail("a record of more than 4 GiB");
  memcpy(head, &torchsim_runtime_attribute_key, 8);
  head[2] = id;
  head[3] = (uint32_t)nbytes;
  torchsim_runtime_attribute_put(head, sizeof head);
  torchsim_runtime_attribute_put(data, nbytes);
  torchsim_runtime_attribute_put(pad, -nbytes & 7);
}

void torchsim_runtime_attribute_close(void) {
  if (!torchsim_runtime_attribute_file) torchsim_runtime_attribute_fail("closing a record file never opened");
  if (fclose(torchsim_runtime_attribute_file)) torchsim_runtime_attribute_fail("cannot flush the record file");
  torchsim_runtime_attribute_file = 0;
}
