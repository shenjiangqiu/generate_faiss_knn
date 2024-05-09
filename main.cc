/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <CLI11.hpp>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <faiss/AutoTune.h>
#include <faiss/index_factory.h>

/**
 * To run this demo, please download the ANN_SIFT1M dataset from
 *
 *   http://corpus-texmex.irisa.fr/
 *
 * and unzip it to the sudirectory sift1M.
 **/

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

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/
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
  if (nr != n * (d + 1)){
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

double elapsed() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}
// const char *TRAIN = "dataset/sift_100M/sift_100M_query_100K.fvecs";
// const char *BASE = "dataset/sift_100M/sift_100M_train_1M.fvecs";
// const char *QUERY = "dataset/sift_100M/sift_100M_query_100K.fvecs";
int main(int argc, char **argv) {

  CLI::App app("Faiss build knn");
  argv = app.ensure_utf8(argv);
  std::string train;
  app.add_option("-t,--train", train, "train file path");
  std::string base;
  app.add_option("-b,--base", base, "base file path");
  std::string query;
  app.add_option("-q,--query", query, "query file path");
  std::string ground_truth;
  app.add_option("-g,--ground_truth", ground_truth, "ground truth file path");
  std::string output;
  app.add_option("-o,--output", output, "output file path");

  CLI11_PARSE(app, argc, argv);

  std::cout << "train: " << train << std::endl;
  std::cout << "base: " << base << std::endl;
  std::cout << "query: " << query << std::endl;
  std::cout << "ground_truth: " << ground_truth << std::endl;
  std::cout << "output: " << output << std::endl;

  double t0 = elapsed();

  // this is typically the fastest one.
  const char *index_key = "OPQ64_128,IVF65536(IVF256,PQ64x4fs,RFlat),PQ64";
  // const char *index_key = "IVF512,Flat";

  // these ones have better memory usage
  // const char *index_key = "Flat";
  // const char *index_key = "PQ32";
  // const char *index_key = "PCA80,Flat";
  // const char *index_key = "IVF4096,PQ8+16";
  // const char *index_key = "IVF4096,PQ32";
  // const char *index_key = "IMI2x8,PQ32";
  // const char *index_key = "IMI2x8,PQ8+16";
  // const char *index_key = "OPQ16_64,IMI2x8,PQ8+16";

  faiss::Index *index;

  size_t d;

  {
    printf("[%.3f s] Loading train set\n", elapsed() - t0);

    size_t nt;
    auto xt = fvecs_read(train.c_str(), &d, &nt);

    printf("[%.3f s] Preparing index \"%s\" d=%ld\n", elapsed() - t0, index_key,
           d);
    index = faiss::index_factory(d, index_key);

    printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nt);

    index->train(nt, xt.get());
  }
  // add base
  {
    printf("[%.3f s] Loading database\n", elapsed() - t0);

    size_t nb, d2;
    auto xb = fvecs_read(base.c_str(), &d2, &nb);
    assert(d == d2 || !"dataset does not have same dimension as train set");

    printf("[%.3f s] Indexing database, size %ld*%ld\n", elapsed() - t0, nb, d);

    index->add(nb, xb.get());
  }

  // read query
  auto total = index->ntotal;
  printf("total: %ld\n", total);
  size_t nq;
  std::unique_ptr<float[]> xq;
  {
    printf("[%.3f s] Loading queries\n", elapsed() - t0);

    size_t d2;
    xq = fvecs_read(query.c_str(), &d2, &nq);
    assert(d == d2 || !"query does not have same dimension as train set");
  }

  size_t k; // nb of results per query in the GT
  std::unique_ptr<faiss::idx_t[]>
      gt; // nq * k matrix of ground-truth nearest-neighbors
  // read ground truth
  {
    printf("[%.3f s] Loading ground truth for %ld queries\n", elapsed() - t0,
           nq);

    // load ground-truth and convert int to long
    size_t nq2;
    auto gt_int = ivecs_read(ground_truth.c_str(), &k, &nq2);
    assert(nq2 == nq || !"incorrect nb of ground truth entries");

    gt = std::unique_ptr<faiss::idx_t[]>(new faiss::idx_t[k * nq]);
    for (size_t i = 0; i < k * nq; i++) {
      gt.get()[i] = gt_int[i];
    }
  }
  std::string selected_params;
  { // run auto-tuning

    printf("[%.3f s] Preparing auto-tune criterion 1-recall at 1 "
           "criterion, with k=%ld nq=%ld\n",
           elapsed() - t0, k, nq);

    faiss::OneRecallAtRCriterion crit(nq, 1);
    crit.set_groundtruth(k, nullptr, gt.get());
    crit.nnn = k; // by default, the criterion will request only 1 NN

    printf("[%.3f s] Preparing auto-tune parameters\n", elapsed() - t0);

    faiss::ParameterSpace params;
    params.initialize(index);

    printf("[%.3f s] Auto-tuning over %ld parameters (%ld combinations)\n",
           elapsed() - t0, params.parameter_ranges.size(),
           params.n_combinations());

    faiss::OperatingPoints ops;
    params.explore(index, nq, xq.get(), crit, &ops);

    printf("[%.3f s] Found the following operating points: \n", elapsed() - t0);

    ops.display();

    // keep the first parameter that obtains > 0.5 1-recall@1
    for (size_t i = 0; i < ops.optimal_pts.size(); i++) {
      std::cout << i << " : " << ops.optimal_pts[i].key
                << " ,t: " << ops.optimal_pts[i].t
                << " , perf: " << ops.optimal_pts[i].perf << std::endl;
    }
    std::cout << "select one: ";
    int select;
    std::cin >> select;
    selected_params = ops.optimal_pts[select].key;
    assert(selected_params.size() >= 0 ||
           !"could not find good enough op point");
  }

  { // Use the found configuration to perform a search

    faiss::ParameterSpace params;

    params.set_index_parameters(index, "nprobe=32");

    printf("[%.3f s] Perform a search on %ld queries\n", elapsed() - t0, nq);

    // output buffers
    auto I = std::unique_ptr<faiss::idx_t[]>(new faiss::idx_t[nq * k]);
    auto D = std::unique_ptr<float[]>(new float[nq * k]);

    index->search(nq, xq.get(), k, D.get(), I.get());

    printf("[%.3f s] Compute recalls\n", elapsed() - t0);

    // evaluate result by hand.
    int n_1 = 0, n_10 = 0, n_100 = 0;
    for (size_t i = 0; i < nq; i++) {
      auto gt_nn = gt[i * k];
      for (size_t j = 0; j < k; j++) {
        if (I[i * k + j] == gt_nn) {
          if (j < 1)
            n_1++;
          if (j < 10)
            n_10++;
          if (j < 100)
            n_100++;
        }
      }
    }
    printf("R@1 = %.4f\n", n_1 / float(nq));
    printf("R@10 = %.4f\n", n_10 / float(nq));
    printf("R@100 = %.4f\n", n_100 / float(nq));
  }
  // build knn for all base and save ivecs
  {
    auto labels = std::unique_ptr<faiss::idx_t[]>(new faiss::idx_t[total * k]);
    auto distances = std::unique_ptr<float[]>(new float[total * k]);
    printf("[%.3f s] Loading database\n", elapsed() - t0);
    size_t nb, d2;
    auto xb = fvecs_read(base.c_str(), &d2, &nb);
    assert(d == d2 || !"dataset does not have same dimension as train set");
    index->search(nb, xb.get(), k, distances.get(), labels.get());
    ivecs_save(output.c_str(), k, total, labels.get());
  }
  delete index;
  return 0;
}