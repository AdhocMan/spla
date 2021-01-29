#include <mpi.h>
#include <complex>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include "CLI/CLI.hpp"
#include "mpi_util/mpi_init_handle.hpp"
#include "spla/context.hpp"
#include "spla/exceptions.hpp"
#include "spla/matrix_distribution.hpp"
#include "spla/spla.hpp"
#include "spla/types.h"
#include "timing/rt_graph.hpp"
#include "timing/timing.hpp"
#include "memory/buffer.hpp"
#include "memory/mpi_allocator.hpp"

#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
#include "memory/pinned_allocator.hpp"
#include "memory/gpu_allocator.hpp"
#endif

template <typename T, typename ALLOCATOR>
void run_pgemm_ssb(spla::Context& ctx, int k, int blacsBlockSize, int numRepeats) {
  int worldRank, worldSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  const int gridRows = std::sqrt(worldSize);
  const int gridCols = worldSize / gridRows;

  const int m = gridRows * blacsBlockSize;
  const int n = gridCols * blacsBlockSize;

  const int maxRowsPerRank = (k + worldSize - 1) / worldSize;
  const int localNumRows = std::min(k - worldRank * maxRowsPerRank, maxRowsPerRank);
  const int maxRowsC = (m / (blacsBlockSize * gridRows) + 1) * blacsBlockSize;
  const int maxColsC = (n / (blacsBlockSize * gridCols) + 1) * blacsBlockSize;

  spla::Buffer<ALLOCATOR> A;
  spla::Buffer<ALLOCATOR> B;
  spla::Buffer<ALLOCATOR> C;
  A.template resize<T>(maxRowsPerRank * m);
  B.template resize<T>(maxRowsPerRank * n);
  C.template resize<T>(maxRowsC * maxColsC);

  const T alpha = 2.0;
  const T beta = 0.0;

  rt_graph::Timer timer;

  auto arrayDesc = spla::MatrixDistribution::create_blacs_block_cyclic(
      MPI_COMM_WORLD, 'R', gridRows, gridCols, blacsBlockSize, blacsBlockSize);

  // run once to warm up
  START_TIMING("warmup");
  spla::pgemm_ssb(m, n, localNumRows, SPLA_OP_CONJ_TRANSPOSE, alpha, A.template data<T>(),
                  localNumRows, B.template data<T>(), localNumRows, beta, C.template data<T>(),
                  maxRowsC, 0, 0, arrayDesc, ctx);
  STOP_TIMING("warmup");

  START_TIMING("benchmark");
  for (int r = 0; r < numRepeats; ++r) {
    SCOPED_TIMING("pgemm_ssb");
    spla::pgemm_ssb(m, n, localNumRows, SPLA_OP_CONJ_TRANSPOSE, alpha, A.template data<T>(),
                    localNumRows, B.template data<T>(), localNumRows, beta, C.template data<T>(),
                    maxRowsC, 0, 0, arrayDesc, ctx);
  }
  STOP_TIMING("benchmark");
}

int main(int argc, char** argv) {
  spla::MPIInitHandle initHandle(argc, argv, true);

  int worldRank, worldSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);


  int repeats = 100;
  int numThreads = 6;
  int lengthTarget = 256;
  std::string procName;
  std::string outputFileName;
  std::string typeName;
  std::string funcName;
  std::vector<int> kValues;
  std::vector<int> blockSizes;

  CLI::App app{"spla benchmark"};
  app.add_option("-r", repeats, "Number of repeats")->default_val("100");
  app.add_option("-l", lengthTarget, "Length target")->default_val("1024");
  app.add_option("-k", kValues, "Number of k in A and B")->required();
  app.add_option("-o", outputFileName, "Output file name")->default_val("timers.json");
  app.add_option("-b,--blocksize", blockSizes, "ScaLAPACK block size of C")->required();
  app.add_option("-t,--threads", numThreads, "Number of threads")->default_val("-1");
  app.add_set("--type", typeName, std::set<std::string>{"scalar", "complex"}, "Data type")
      ->default_val("complex");
  app.add_set("-p", procName, std::set<std::string>{"cpu", "gpu", "gpu-gpu"}, "Processing unit")->required();
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }


  SplaProcessingUnit pu = procName == "cpu" ? SplaProcessingUnit::SPLA_PU_HOST : SplaProcessingUnit::SPLA_PU_GPU;
  spla::Context ctx(pu);
  ctx.set_tile_size_host(lengthTarget);
  ctx.set_num_threads(numThreads);
  ctx.set_tile_size_gpu(4096);

  if (worldRank == 0) {
    std::cout << "k = ";
    for (const auto &k : kValues) {
      std::cout << k << ", ";
    }
    std::cout << std::endl;
    std::cout << "tile length = " << lengthTarget << std::endl;
    std::cout << "block size = ";
    for (const auto &blacsBlockSize : blockSizes) {
      std::cout << blacsBlockSize << ", ";
    }
    std::cout << std::endl;
    std::cout << "repeats = " << repeats << std::endl;
    std::cout << "proc = " << procName << std::endl;
    std::cout << "type = " << typeName << std::endl;
    std::cout << "threads = " << ctx.num_threads() << std::endl;
  }

  for (const auto &k : kValues) {
    SCOPED_TIMING("k=" + std::to_string(k));
    for (const auto &blacsBlockSize : blockSizes) {
      SCOPED_TIMING("b=" + std::to_string(blacsBlockSize));
      for (auto useRingReduce : {false, true}) {
        SCOPED_TIMING(useRingReduce ? "ring" : "allreduce");
        spla::useRingReduceGlobal = useRingReduce;
        if (procName == "cpu") {
          if (typeName == "scalar")
            run_pgemm_ssb<double, spla::MPIAllocator>(ctx, k, blacsBlockSize,
                                                      repeats);
          else
            run_pgemm_ssb<std::complex<double>, spla::MPIAllocator>(
                ctx, k, blacsBlockSize, repeats);
        }
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
        else if (procName == "gpu") {
          if (typeName == "scalar")
            run_pgemm_ssb<double, spla::PinnedAllocator>(ctx, k, blacsBlockSize,
                                                         repeats);
          else
            run_pgemm_ssb<std::complex<double>, spla::PinnedAllocator>(
                ctx, k, blacsBlockSize, repeats);
        } else if (procName == "gpu-gpu") {
          if (typeName == "scalar")
            run_pgemm_ssb<double, spla::GPUAllocator>(ctx, k, blacsBlockSize,
                                                      repeats);
          else
            run_pgemm_ssb<std::complex<double>, spla::GPUAllocator>(
                ctx, k, blacsBlockSize, repeats);
        }
#else
        else {
          throw spla::GPUSupportError();
        }
#endif
      }
    }
  }

  if (worldRank == 0) {
    auto result = spla::timing::GlobalTimer.process();
    std::cout << result.print() << std::endl;
    std::ofstream file(outputFileName);
    file << result.json();
  }

  return 0;
}
