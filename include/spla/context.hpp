/*
 * Copyright (c) 2020 ETH Zurich, Simon Frasch
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef SPLA_CONTEXT_HPP
#define SPLA_CONTEXT_HPP

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "spla/config.h"
#include "spla/types.h"

/*! \cond PRIVATE */
namespace spla {
/*! \endcond */

class ContextInternal;

class MatrixDistribution;

/**
 * Context, which provides configuration settings and reusable resources.
 */
class SPLA_EXPORT Context {
public:
  /**
   * Constructor of Context with default configuration for given processing unit.
   *
   * @param[in] pu Processing unit to be used for computations.
   */
  explicit Context(SplaProcessingUnit pu);

  /**
   * Default move constructor.
   */
  Context(Context &&) = default;

  /**
   * Disabled copy constructor.
   */
  Context(const Context &) = delete;

  /**
   * Default move assignment operator.
   */
  Context &operator=(Context &&) = default;

  /**
   * Disabled copy assignment operator.
   */
  Context &operator=(const Context &) = delete;

  /**
   * Access a Context parameter.
   * @return Processing unit used.
   */
  SplaProcessingUnit processing_unit() const;

  /**
   * Access a Context parameter.
   * @return Number of tiles used to overlap computation and communication.
   */
  int num_tiles() const;

  /**
   * Access a Context parameter.
   * @return Size of tiles on host. Used for partitioning communication.
   */
  int tile_size_host() const;

  /**
   * Access a Context parameter.
   * @return Target size of tiles on GPU.
   */
  int tile_size_gpu() const;

  /**
   * Access a Context parameter.
   * @return Operations threshold, below which computation may be done on Host, even if processing
   * unit is set to GPU. For GEMM, the number of operations is estimatex as 2mnk.
   */
  int op_threshold_gpu() const;

  /**
   * Access a Context parameter.
   * @return Id of GPU used for computations. This is set as fixed parameter by query of device id
   * at context creation.
   */
  int gpu_device_id() const;

  /**
   * Access a Context parameter.
   * @return Total allocated memory on host in bytes used for internal buffers. Does not include
   * allocations through standard C++ allocators. May change with use of context.
   */
  std::uint_least64_t allocated_memory_host() const;

  /**
   * Access a Context parameter.
   * @return Total allocated pinned memory on host in bytes used for internal buffers. Does not include
   * allocations through standard C++ allocators. May change with with use of context.
   */
  std::uint_least64_t allocated_memory_pinned() const;

  /**
   * Access a Context parameter.
   * @return Total allocated memory on gpu in bytes used for internal buffers. Does not include
   * allocations by device libraries like cuBLAS / rocBLAS. May change with with use of context.
   */
  std::uint_least64_t allocated_memory_gpu() const;

  /**
   * Set the number of tiles.
   *
   * @param[in] numTiles Number of tiles.
   */
  void set_num_tiles(int numTiles);

  /**
   * Set the tile size used for computations on host and partitioning of communication.
   *
   * @param[in] tileSizeHost Tile size.
   */
  void set_tile_size_host(int tileSizeHost);

  /**
   * Set the operations threshold, below which computation may be done on Host, even if processing
   * unit is set to GPU. For GEMM, the number of operations is estimatex as 2mnk.
   *
   * @param[in] opThresholdGPU Threshold in number of operations.
   */
  void set_op_threshold_gpu(int opThresholdGPU);

  /**
   * Set the tile size used for computations on GPU.
   *
   * @param[in] tileSizeGPU Tile size on GPU.
   */
  void set_tile_size_gpu(int tileSizeGPU);

  /**
   * Set the allocation and deallocation functions for host memory. Internal default uses a memory
   * pool for better performance.
   *
   * @param[in] allocateFunc Function allocating given size in bytes.
   * @param[in] deallocateFunc Function to deallocate memory allocated using allocateFunc.
   */
  void set_alloc_host(std::function<void *(std::size_t)> allocateFunc,
                      std::function<void(void *)> deallocateFunc);

  /**
   * Set the allocation and deallocation functions for pinned host memory. Internal default uses a
   * memory pool for better performance.
   *
   * @param[in] allocateFunc Function allocating given size in bytes.
   * @param[in] deallocateFunc Function to deallocate memory allocated using allocateFunc.
   */
  void set_alloc_pinned(std::function<void *(std::size_t)> allocateFunc,
                        std::function<void(void *)> deallocateFunc);

  /**
   * Set the allocation and deallocation functions for gpu memory. Internal default uses a
   * memory pool for better performance.
   *
   * @param[in] allocateFunc Function allocating given size in bytes.
   * @param[in] deallocateFunc Function to deallocate memory allocated using allocateFunc.
   */
  void set_alloc_gpu(std::function<void *(std::size_t)> allocateFunc,
                     std::function<void(void *)> deallocateFunc);

private:
  /*! \cond PRIVATE */
  friend void pgemm_ssb(int m, int n, int kLocal, SplaOperation opA, float alpha, const float *A,
                        int lda, const float *B, int ldb, float beta, float *C, int ldc,
                        int cRowStart, int cColStart, MatrixDistribution &descC, Context &ctx);

  friend void pgemm_ssb(int m, int n, int kLocal, SplaOperation opA, double alpha, const double *A,
                        int lda, const double *B, int ldb, double beta, double *C, int ldc,
                        int cRowStart, int cColStart, MatrixDistribution &descC, Context &ctx);

  friend void pgemm_ssb(int m, int n, int kLocal, SplaOperation opA, std::complex<float> alpha,
                        const std::complex<float> *A, int lda, const std::complex<float> *B,
                        int ldb, std::complex<float> beta, std::complex<float> *C, int ldc,
                        int cRowStart, int cColStart, MatrixDistribution &descC, Context &ctx);

  friend void pgemm_ssb(int m, int n, int kLocal, SplaOperation opA, std::complex<double> alpha,
                        const std::complex<double> *A, int lda, const std::complex<double> *B,
                        int ldb, std::complex<double> beta, std::complex<double> *C, int ldc,
                        int cRowStart, int cColStart, MatrixDistribution &descC, Context &ctx);

  friend void pgemm_sbs(int mLocal, int n, int k, float alpha, const float *A, int lda,
                        const float *B, int ldb, int bRowOffset, int bColOffset,
                        MatrixDistribution &descB, float beta, float *C, int ldc, Context &ctx);

  friend void pgemm_sbs(int mLocal, int n, int k, double alpha, const double *A, int lda,
                        const double *B, int ldb, int bRowOffset, int bColOffset,
                        MatrixDistribution &descB, double beta, double *C, int ldc, Context &ctx);

  friend void pgemm_sbs(int mLocal, int n, int k, std::complex<float> alpha,
                        const std::complex<float> *A, int lda, const std::complex<float> *B,
                        int ldb, int bRowOffset, int bColOffset, MatrixDistribution &descB,
                        std::complex<float> beta, std::complex<float> *C, int ldc, Context &ctx);

  friend void pgemm_sbs(int mLocal, int n, int k, std::complex<double> alpha,
                        const std::complex<double> *A, int lda, const std::complex<double> *B,
                        int ldb, int bRowOffset, int bColOffset, MatrixDistribution &descB,
                        std::complex<double> beta, std::complex<double> *C, int ldc, Context &ctx);

  friend void pgemm_ssbtr(int m, int n, int kLocal, SplaOperation opA, float alpha, const float *A,
                          int lda, const float *B, int ldb, float beta, float *C, int ldc,
                          int cRowStart, int cColStart, SplaFillMode cFillMode,
                          MatrixDistribution &descC, Context &ctx);

  friend void pgemm_ssbtr(int m, int n, int kLocal, SplaOperation opA, double alpha,
                          const double *A, int lda, const double *B, int ldb, double beta,
                          double *C, int ldc, int cRowStart, int cColStart, SplaFillMode cFillMode,
                          MatrixDistribution &descC, Context &ctx);

  friend void pgemm_ssbtr(int m, int n, int kLocal, SplaOperation opA, std::complex<float> alpha,
                          const std::complex<float> *A, int lda, const std::complex<float> *B,
                          int ldb, std::complex<float> beta, std::complex<float> *C, int ldc,
                          int cRowStart, int cColStart, SplaFillMode cFillMode,
                          MatrixDistribution &descC, Context &ctx);

  friend void pgemm_ssbtr(int m, int n, int kLocal, SplaOperation opA, std::complex<double> alpha,
                          const std::complex<double> *A, int lda, const std::complex<double> *B,
                          int ldb, std::complex<double> beta, std::complex<double> *C, int ldc,
                          int cRowStart, int cColStart, SplaFillMode cFillMode,
                          MatrixDistribution &descC, Context &ctx);

  friend void gemm(SplaOperation opA, SplaOperation opB, int m, int n, int k, float alpha,
                   const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc,
                   Context &ctx);

  friend void gemm(SplaOperation opA, SplaOperation opB, int m, int n, int k, double alpha,
                   const double *A, int lda, const double *B, int ldb, double beta, double *C,
                   int ldc, Context &ctx);

  friend void gemm(SplaOperation opA, SplaOperation opB, int m, int n, int k,
                   std::complex<float> alpha, const std::complex<float> *A, int lda,
                   const std::complex<float> *B, int ldb, std::complex<float> beta,
                   std::complex<float> *C, int ldc, Context &ctx);

  friend void gemm(SplaOperation opA, SplaOperation opB, int m, int n, int k,
                   std::complex<double> alpha, const std::complex<double> *A, int lda,
                   const std::complex<double> *B, int ldb, std::complex<double> beta,
                   std::complex<double> *C, int ldc, Context &ctx);

  std::shared_ptr<ContextInternal> ctxInternal_;
  /*! \endcond */
};

/*! \cond PRIVATE */
}  // namespace spla
/*! \endcond */

#endif
