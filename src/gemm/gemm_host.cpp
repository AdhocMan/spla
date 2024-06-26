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

#include "gemm/gemm_host.hpp"

#include <algorithm>
#include <complex>

#include "memory/host_array_const_view.hpp"
#include "memory/host_array_view.hpp"
#include "spla/config.h"
#include "spla/context_internal.hpp"
#include "spla/types.h"
#include "util/blas_interface.hpp"
#include "util/check_gemm_param.hpp"

namespace spla {

static auto map_op_to_host_blas(SplaOperation op) -> blas::Operation {
  switch (op) {
    case SplaOperation::SPLA_OP_TRANSPOSE:
      return blas::Operation::TRANS;
    case SplaOperation::SPLA_OP_CONJ_TRANSPOSE:
      return blas::Operation::CONJ_TRANS;
    default:
      return blas::Operation::NONE;
  }
}

template <typename T>
void gemm_host(SplaOperation opA, SplaOperation opB, IntType m, IntType n, IntType k, T alpha,
               const T *A, IntType lda, const T *B, IntType ldb, T beta, T *C, IntType ldc) {
  if (m == 0 || n == 0) {
    return;
  }
  check_gemm_param(opA, opB, m, n, k, A, lda, B, ldb, C, ldc);

  const auto opBlasA = map_op_to_host_blas(opA);
  const auto opBlasB = map_op_to_host_blas(opB);

  // Some blas libraries like MKL do not accept 0 as ld, even if m, n or k is 0
  if (lda < 1) lda = 1;
  if (ldb < 1) ldb = 1;
  if (ldc < 1) ldc = 1;

  blas::gemm(opBlasA, opBlasB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template auto gemm_host<float>(SplaOperation opA, SplaOperation opB, IntType m, IntType n,
                               IntType k, float alpha, const float *A, IntType lda, const float *B,
                               IntType ldb, float beta, float *C, IntType ldc) -> void;

template auto gemm_host<double>(SplaOperation opA, SplaOperation opB, IntType m, IntType n,
                                IntType k, double alpha, const double *A, IntType lda,
                                const double *B, IntType ldb, double beta, double *C, IntType ldc)
    -> void;

template auto gemm_host<std::complex<float>>(SplaOperation opA, SplaOperation opB, IntType m,
                                             IntType n, IntType k, std::complex<float> alpha,
                                             const std::complex<float> *A, IntType lda,
                                             const std::complex<float> *B, IntType ldb,
                                             std::complex<float> beta, std::complex<float> *C,
                                             IntType ldc) -> void;

template auto gemm_host<std::complex<double>>(SplaOperation opA, SplaOperation opB, IntType m,
                                              IntType n, IntType k, std::complex<double> alpha,
                                              const std::complex<double> *A, IntType lda,
                                              const std::complex<double> *B, IntType ldb,
                                              std::complex<double> beta, std::complex<double> *C,
                                              IntType ldc) -> void;

}  // namespace spla
