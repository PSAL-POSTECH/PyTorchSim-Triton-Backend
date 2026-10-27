/* The runtime attribute records a kernel writes, streamed through pk's file calls to one file: a header
   (magic, version, the lowered module's table hash), then per record its program key, id, byte count and
   bytes, padded to 8. What the bytes mean is the manifest's; any failure is named and exits 1. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern const uint64_t pytorchsim_runtime_attribute_table_hash;
uint64_t pytorchsim_runtime_attribute_key;
static FILE *pytorchsim_runtime_attribute_file;

static void pytorchsim_runtime_attribute_fail(const char *what) {
  fputs("pytorchsim runtime attribute: ", stderr);
  fputs(what, stderr);
  fputs("\n", stderr);
  exit(1);
}

static void pytorchsim_runtime_attribute_put(const void *data, uint64_t nbytes) {
  if (nbytes && fwrite(data, 1, nbytes, pytorchsim_runtime_attribute_file) != nbytes)
    pytorchsim_runtime_attribute_fail("cannot write the record file");
}

void pytorchsim_runtime_attribute_open(const char *path) {
  static const uint32_t version = PYTORCHSIM_RUNTIME_ATTRIBUTE_VERSION;
  enum { BUFFER = 1 << 22 };
  if (!path) pytorchsim_runtime_attribute_fail("no record file path follows the tensor paths");
  if (pytorchsim_runtime_attribute_file) pytorchsim_runtime_attribute_fail("the record file is open already");
  pytorchsim_runtime_attribute_file = fopen(path, "wb");
  if (!pytorchsim_runtime_attribute_file) pytorchsim_runtime_attribute_fail("cannot open the record file");
  char *buffer = malloc(BUFFER);
  if (!buffer || setvbuf(pytorchsim_runtime_attribute_file, buffer, _IOFBF, BUFFER))
    pytorchsim_runtime_attribute_fail("cannot give the record file its buffer");
  pytorchsim_runtime_attribute_put("TSRA", 4);
  pytorchsim_runtime_attribute_put(&version, 4);
  pytorchsim_runtime_attribute_put(&pytorchsim_runtime_attribute_table_hash, 8);
}

void pytorchsim_runtime_attribute_write(uint32_t id, const void *data, uint64_t nbytes) {
  static const uint8_t pad[8];
  uint32_t head[4];
  if (!pytorchsim_runtime_attribute_file) pytorchsim_runtime_attribute_fail("a record before the record file is open");
  if (nbytes > UINT32_MAX) pytorchsim_runtime_attribute_fail("a record of more than 4 GiB");
  memcpy(head, &pytorchsim_runtime_attribute_key, 8);
  head[2] = id;
  head[3] = (uint32_t)nbytes;
  pytorchsim_runtime_attribute_put(head, sizeof head);
  pytorchsim_runtime_attribute_put(data, nbytes);
  pytorchsim_runtime_attribute_put(pad, -nbytes & 7);
}

void pytorchsim_runtime_attribute_close(void) {
  if (!pytorchsim_runtime_attribute_file) pytorchsim_runtime_attribute_fail("closing a record file never opened");
  if (fclose(pytorchsim_runtime_attribute_file)) pytorchsim_runtime_attribute_fail("cannot flush the record file");
  pytorchsim_runtime_attribute_file = 0;
}
