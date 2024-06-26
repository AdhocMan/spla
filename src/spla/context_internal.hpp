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
#ifndef SPLA_CONTEXT_INTERNAL_HPP
#define SPLA_CONTEXT_INTERNAL_HPP

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <deque>
#include <memory>
#include <type_traits>

#include "memory/allocator_collection.hpp"
#include "mpi_util/mpi_check_status.hpp"
#include "spla/config.h"
#include "spla/context.hpp"
#include "spla/exceptions.hpp"
#include "util/common_types.hpp"

#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
#include "gpu_util/gpu_blas_handle.hpp"
#include "gpu_util/gpu_event_handle.hpp"
#include "gpu_util/gpu_stream_handle.hpp"
#endif

namespace spla {

class ContextInternal {
public:
  explicit ContextInternal(SplaProcessingUnit pu)
      : pu_(pu),
        numTiles_(4),
        tileSizeHost_(pu == SplaProcessingUnit::SPLA_PU_HOST ? 500 : 1500),
        tileSizeGPU_(2048),
        opThresholdGPU_(2000000),
        gpuDeviceId_(0) {
    if (pu == SplaProcessingUnit::SPLA_PU_GPU) {
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
      gpu::check_status(gpu::get_device(&gpuDeviceId_));
#else
      throw GPUSupportError();
#endif
    } else if (pu != SplaProcessingUnit::SPLA_PU_HOST) {
      throw InvalidParameterError();
    }
  }

#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
  inline auto gpu_blas_handles(IntType numHandles) -> std::deque<GPUBlasHandle>& {
    const IntType numMissing = numHandles - static_cast<IntType>(gpuBlasHandles_.size());
    if (static_cast<IntType>(gpuBlasHandles_.size()) < numHandles) {
      gpuBlasHandles_.resize(numHandles);
    }
    return gpuBlasHandles_;
  }

  inline auto gpu_event_handles(IntType numHandles) -> std::deque<GPUEventHandle>& {
    const IntType numMissing = numHandles - static_cast<IntType>(gpuEventHandles_.size());
    if (static_cast<IntType>(gpuEventHandles_.size()) < numHandles) {
      gpuEventHandles_.resize(numHandles);
    }
    return gpuEventHandles_;
  }

  inline auto gpu_stream_handles(IntType numHandles) -> std::deque<GPUStreamHandle>& {
    const IntType numMissing = numHandles - static_cast<IntType>(gpuStreamHandles_.size());
    if (static_cast<IntType>(gpuStreamHandles_.size()) < numHandles) {
      gpuStreamHandles_.resize(numHandles);
    }
    return gpuStreamHandles_;
  }
#endif

  // Get methods

  inline auto processing_unit() const -> SplaProcessingUnit { return pu_; }

  inline auto num_tiles() const -> IntType { return numTiles_; }

  inline auto tile_size_host() const -> IntType { return tileSizeHost_; }

  inline auto tile_size_gpu() const -> IntType { return tileSizeGPU_; }

  inline auto op_threshold_gpu() const -> IntType { return opThresholdGPU_; }

  inline auto gpu_device_id() const -> int { return gpuDeviceId_; }

  inline auto allocators() const -> const AllocatorCollection& { return allocators_; }

  inline auto allocators() -> AllocatorCollection& { return allocators_; }

  // Set methods

  inline auto set_num_tiles(IntType numTiles) -> void {
    if (numTiles < 1) throw InvalidParameterError();
    numTiles_ = numTiles;
  }

  inline auto set_tile_size_host(IntType tileSizeHost) -> void {
    if (tileSizeHost < 1) throw InvalidParameterError();
    tileSizeHost_ = tileSizeHost;
  }

  inline auto set_tile_size_gpu(IntType tileSizeGPU) -> void {
    if (tileSizeGPU < 1) throw InvalidParameterError();
    tileSizeGPU_ = tileSizeGPU;
  }

  inline auto set_op_threshold_gpu(IntType opThresholdGPU) -> void {
    if (opThresholdGPU < 0) throw InvalidParameterError();
    opThresholdGPU_ = opThresholdGPU;
  }

private:
  SplaProcessingUnit pu_;
  IntType numTiles_;
  IntType tileSizeHost_;
  IntType tileSizeGPU_;
  IntType opThresholdGPU_;
  int gpuDeviceId_;

  AllocatorCollection allocators_;
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
  std::deque<GPUBlasHandle> gpuBlasHandles_;
  std::deque<GPUEventHandle> gpuEventHandles_;
  std::deque<GPUStreamHandle> gpuStreamHandles_;
#endif
};
}  // namespace spla

#endif
