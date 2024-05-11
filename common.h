#include <faiss/Index.h>
#include <memory>

void ivecs_save(const char *fname, size_t d, size_t n, const faiss::idx_t *x);
std::unique_ptr<float[]> fvecs_read(const char *fname, size_t *d_out,
                                    size_t *n_out);
std::unique_ptr<int[]> ivecs_read(const char *fname, size_t *d_out,
                                  size_t *n_out);