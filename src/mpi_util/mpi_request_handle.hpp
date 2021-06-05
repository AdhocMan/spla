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
#ifndef SPLA_MPI_REQUEST_HANDLE_HPP
#define SPLA_MPI_REQUEST_HANDLE_HPP

#include <mpi.h>

#include <memory>
#include <vector>

#include "mpi_util/mpi_check_status.hpp"
#include "spla/config.h"

namespace spla {

// Storage for MPI datatypes
class MPIRequestHandle {
public:
  MPIRequestHandle() = default;

  MPIRequestHandle(const MPIRequestHandle&) = delete;

  MPIRequestHandle(MPIRequestHandle&& handle) { *this = std::move(handle); }

  auto operator=(const MPIRequestHandle&) -> MPIRequestHandle& = delete;

  auto operator=(MPIRequestHandle&& handle) -> MPIRequestHandle& {
    mpiRequest_ = handle.mpiRequest_;
    handle.mpiRequest_ = MPI_REQUEST_NULL;
    return *this;
  }

  inline auto get() -> MPI_Request* { return &mpiRequest_; }

  inline auto wait() -> void { mpi_check_status(MPI_Wait(&mpiRequest_, MPI_STATUS_IGNORE)); }

private:
  MPI_Request mpiRequest_ = MPI_REQUEST_NULL;
};


class PersistantMPIRequestHandle {
public:
  PersistantMPIRequestHandle() = default;

  PersistantMPIRequestHandle(const MPIRequestHandle&) = delete;

  PersistantMPIRequestHandle(PersistantMPIRequestHandle&& handle) {
    *this = std::move(handle);
  }

  auto operator=(const PersistantMPIRequestHandle& other) -> PersistantMPIRequestHandle& = delete;

  auto operator=(PersistantMPIRequestHandle&& handle) -> PersistantMPIRequestHandle& {
    if (active_) MPI_Request_free(req_.get());
    req_ = std::move(handle.req_);
    active_ = handle.active_;
    handle.active_ = false;
    return *this;
  }

  ~PersistantMPIRequestHandle() {
    if (active_) MPI_Request_free(req_.get());
  }

  inline auto init_send(const void* buf, int count, MPI_Datatype dataType, int dest, int tag,
                        MPI_Comm comm) -> void {
    if (active_) MPI_Request_free(req_.get());
    mpi_check_status(MPI_Send_init(buf, count, dataType, dest, tag, comm, req_.get()));
    active_ = true;
  }

  inline auto init_recv(void* buf, int count, MPI_Datatype dataType, int source, int tag,
                        MPI_Comm comm) -> void {
    if (active_) MPI_Request_free(req_.get());
    mpi_check_status(MPI_Recv_init(buf, count, dataType, source, tag, comm, req_.get()));
    active_ = true;
  }

  inline auto wait() -> void { req_.wait(); }

  inline auto start() -> void {
    if (active_) mpi_check_status(MPI_Start(req_.get()));
  }

private:
  bool active_ = false;
  MPIRequestHandle req_;
};

}  // namespace spla

#endif
