
#include <cassert>
#include <cmath>
#include <common.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <faiss/Index.h>
#include <memory>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
void ivecs_save(const char *fname, size_t d, size_t n, const faiss::idx_t *x) {
  FILE *f = fopen(fname, "w");
  if (!f) {
    fprintf(stderr, "could not open %s for writing\n", fname);
    perror("");
    abort();
  }
  auto temp_int = std::unique_ptr<int[]>(new int[d]);
  for (size_t i = 0; i < n; i++) {
    fwrite(&d, 1, sizeof(int), f);
    const faiss::idx_t *xi = x + i * d;
    for (size_t j = 0; j < d; j++) {
      temp_int[j] = xi[j];
    }
    fwrite(temp_int.get(), d, sizeof(int), f);
  }
  fflush(f);
  fclose(f);
}

std::unique_ptr<float[]> fvecs_read(const char *fname, size_t *d_out,
                                    size_t *n_out) {
  FILE *f = fopen(fname, "r");
  if (!f) {
    fprintf(stderr, "could not open %s\n", fname);
    perror("");
    abort();
  }
  int d;
  auto r = fread(&d, 1, sizeof(int), f);
  if (r == 0) {
    fprintf(stderr, "could not read vector dimension in %s\n", fname);
    perror("");
    abort();
  }
  assert((d > 0 && d < 1000000) || !"unreasonable dimension");
  fseek(f, 0, SEEK_SET);
  struct stat st;
  fstat(fileno(f), &st);
  size_t sz = st.st_size;
  assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
  size_t n = sz / ((d + 1) * 4);

  *d_out = d;
  *n_out = n;
  auto x = std::unique_ptr<float[]>(new float[n * (d + 1)]);
  size_t nr = fread(x.get(), sizeof(float), n * (d + 1), f);
  if (nr != n * (d + 1)) {
    fprintf(stderr, "could not read whole file\n");
    perror("");
    abort();
  }

  assert(nr == n * (d + 1) || !"could not read whole file");

  // shift array to remove row headers
  for (size_t i = 0; i < n; i++)
    memmove(x.get() + i * d, x.get() + 1 + i * (d + 1), d * sizeof(*x.get()));

  fclose(f);
  return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
std::unique_ptr<int[]> ivecs_read(const char *fname, size_t *d_out,
                                  size_t *n_out) {
  return std::unique_ptr<int[]>(
      reinterpret_cast<int *>(fvecs_read(fname, d_out, n_out).release()));
}
