
!  Copyright (c) 2019 ETH Zurich, Simon Frasch
!
!  Redistribution and use in source and binary forms, with or without
!  modification, are permitted provided that the following conditions are met:
!
!  1. Redistributions of source code must retain the above copyright notice,
!     this list of conditions and the following disclaimer.
!  2. Redistributions in binary form must reproduce the above copyright
!     notice, this list of conditions and the following disclaimer in the
!     documentation and/or other materials provided with the distribution.
!  3. Neither the name of the copyright holder nor the names of its contributors
!     may be used to endorse or promote products derived from this software
!     without specific prior written permission.
!
!  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
!  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
!  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
!  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
!  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
!  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
!  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
!  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
!  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
!  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
!  POSSIBILITY OF SUCH DAMAGE.

module spla

use iso_c_binding
implicit none

! Constants
integer(c_int), parameter ::                  &
    SPLA_DIST_BLACS_BLOCK_CYCLIC        = 0,  &
    SPLA_DIST_MIRROR                    = 1,  &

    SPLA_PU_HOST                        = 0,  &
    SPLA_PU_GPU                         = 1,  &

    SPLA_OP_NONE                        = 0,  &
    SPLA_OP_TRANSPOSE                   = 1,  &
    SPLA_OP_CONJ_TRANSPOSE              = 2,  &

    SPLA_FILL_MODE_FULL                 = 0,  &
    SPLA_FILL_MODE_UPPER                = 1,  &
    SPLA_FILL_MODE_LOWER                = 2,  &

    SPFFT_SUCCESS                       = 0,  &
    SPFFT_UNKNOWN_ERROR                 = 1,  &
    SPFFT_INTERNAL_ERROR                = 2,  &
    SPFFT_INVALID_PARAMETER_ERROR       = 3,  &
    SPFFT_INVALID_POINTER_ERROR         = 4,  &
    SPFFT_INVALID_HANDLE_ERROR          = 5,  &
    SPFFT_MPI_ERROR                     = 6,  &
    SPFFT_MPI_ALLOCATION_ERROR          = 7,  &
    SPFFT_MPI_THREAD_SUPPORT_ERROR      = 8,  &
    SPFFT_GPU_ERROR                     = 9,  &
    SPFFT_GPU_SUPPORT_ERROR             = 10, &
    SPFFT_GPU_ALLOCATION_ERROR          = 11, &
    SPFFT_GPU_LAUNCH_ERROR              = 12, &
    SPFFT_GPU_NO_DEVICE_ERROR           = 13, &
    SPFFT_GPU_INVALID_VALUE_ERROR       = 14, &
    SPFFT_GPU_INVALID_DEVICE_PTR_ERROR  = 15, &
    SPFFT_GPU_BLAS_ERROR                = 16

interface

  !--------------------------
  !          Context
  !--------------------------
  integer(c_int) function spla_ctx_create(ctx, pu) bind(C)
    use iso_c_binding
    type(c_ptr), intent(out) :: ctx
    integer(c_int), value :: pu
  end function

  integer(c_int) function spla_ctx_destroy(ctx) bind(C)
    use iso_c_binding
    type(c_ptr), intent(inout) :: ctx
  end function

  integer(c_int) function spla_ctx_processing_unit(ctx, processingUnit) bind(C)
    use iso_c_binding
    type(c_ptr), value :: ctx
    integer(c_int), intent(out) :: processingUnit
  end function

  integer(c_int) function spla_ctx_num_threads(ctx, numThreads) bind(C)
    use iso_c_binding
    type(c_ptr), value :: ctx
    integer(c_int), intent(out) :: numThreads
  end function

  integer(c_int) function spla_ctx_num_tiles(ctx, numTiles) bind(C)
    use iso_c_binding
    type(c_ptr), value :: ctx
    integer(c_int), intent(out) :: numTiles
  end function

  integer(c_int) function spla_ctx_tile_size_host(ctx, tileSizeHost) bind(C)
    use iso_c_binding
    type(c_ptr), value :: ctx
    integer(c_int), intent(out) :: tileSizeHost
  end function

  integer(c_int) function spla_ctx_tile_size_gpu(ctx, tileSizeGPU) bind(C)
    use iso_c_binding
    type(c_ptr), value :: ctx
    integer(c_int), intent(out) :: tileSizeGPU
  end function

  integer(c_int) function spla_ctx_op_threshold_gpu(ctx, opThresholdGPU) bind(C)
    use iso_c_binding
    type(c_ptr), value :: ctx
    integer(c_int), intent(out) :: opThresholdGPU
  end function

  integer(c_int) function spla_ctx_gpu_device_id(ctx, deviceId) bind(C)
    use iso_c_binding
    type(c_ptr), value :: ctx
    integer(c_int), intent(out) :: deviceId
  end function

  integer(c_int) function spla_ctx_set_num_threads(ctx, numThreads) bind(C)
    use iso_c_binding
    type(c_ptr), value :: ctx
    integer(c_int), value :: numThreads
  end function

  integer(c_int) function spla_ctx_set_num_tiles(ctx, numTiles) bind(C)
    use iso_c_binding
    type(c_ptr), value :: ctx
    integer(c_int), value :: numTiles
  end function

  integer(c_int) function spla_ctx_set_tile_size_host(ctx, tileSizeHost) bind(C)
    use iso_c_binding
    type(c_ptr), value :: ctx
    integer(c_int), value :: tileSizeHost
  end function

  integer(c_int) function spla_ctx_set_tile_threshold_gpu(ctx, opThresholdGPU) bind(C)
    use iso_c_binding
    type(c_ptr), value :: ctx
    integer(c_int), value :: opThresholdGPU
  end function

  integer(c_int) function spla_ctx_set_tile_size_gpu(ctx, tileSizeGPU) bind(C)
    use iso_c_binding
    type(c_ptr), value :: ctx
    integer(c_int), value :: tileSizeGPU
  end function


  !--------------------------
  !    Matrix Distribution
  !--------------------------
  integer(c_int) function spla_mat_dis_create_block_cyclic(matDis, commFortran, order, &
                                                           procGridRows, procGridCols, &
                                                           rowBlockSize, colBlockSize) &
      bind(C, name='spla_mat_dis_create_block_cyclic_fortran')
    use iso_c_binding
    type(c_ptr), intent(out) :: matDis
    integer(c_int), value :: commFortran
    character(c_char), value :: order
    integer(c_int), value :: procGridRows
    integer(c_int), value :: procGridCols
    integer(c_int), value :: rowBlockSize
    integer(c_int), value :: colBlockSize
  end function

  integer(c_int) function spla_mat_dis_create_blacs_block_cyclic_from_mapping(matDis, &
                                                           commFortran, mapping, &
                                                           procGridRows, procGridCols, &
                                                           rowBlockSize, colBlockSize) &
      bind(C, name='spla_mat_dis_create_blacs_block_cyclic_from_mapping_fortran')
    use iso_c_binding
    type(c_ptr), intent(out) :: matDis
    integer(c_int), value :: commFortran
    integer(c_int), dimension(*), intent(in) :: mapping
    integer(c_int), value :: procGridRows
    integer(c_int), value :: procGridCols
    integer(c_int), value :: rowBlockSize
    integer(c_int), value :: colBlockSize
  end function

  integer(c_int) function spla_create_mirror(matDis, commFortran) &
      bind(C, name='spla_mat_dis_create_mirror_fortran')
    use iso_c_binding
    type(c_ptr), intent(out) :: matDis
    integer(c_int), value :: commFortran
  end function

  integer(c_int) function spla_mat_dis_destroy(matDis) bind(C)
    use iso_c_binding
    type(c_ptr), intent(inout) :: matDis
  end function

  integer(c_int) function spla_mat_dis_proc_grid_rows(ctx, rows) bind(C)
    use iso_c_binding
    type(c_ptr), value :: ctx
    integer(c_int), intent(out) :: rows
  end function

  integer(c_int) function spla_mat_dis_proc_grid_cols(ctx, cols) bind(C)
    use iso_c_binding
    type(c_ptr), value :: ctx
    integer(c_int), intent(out) :: cols
  end function

  integer(c_int) function spla_mat_dis_row_block_size(ctx, rowBlockSize) bind(C)
    use iso_c_binding
    type(c_ptr), value :: ctx
    integer(c_int), intent(out) :: rowBlockSize
  end function

  integer(c_int) function spla_mat_dis_col_block_size(ctx, colBlockSize) bind(C)
    use iso_c_binding
    type(c_ptr), value :: ctx
    integer(c_int), intent(out) :: colBlockSize
  end function

  integer(c_int) function spla_mat_dis_type(ctx, disType) bind(C)
    use iso_c_binding
    type(c_ptr), value :: ctx
    integer(c_int), intent(out) :: disType
  end function

  integer(c_int) function spla_mat_dis_comm(ctx, commFortran) &
      bind(C, name='spla_mat_dis_comm_fortran')
    use iso_c_binding
    type(c_ptr), value :: ctx
    integer(c_int), intent(out) :: commFortran
  end function


  !--------------------------
  !         pgemm_ssb
  !--------------------------

  integer(c_int) function spla_psgemm_ssb(m, n, kLocal, opA, &
                                          alpha, A, lda, B, &
                                          ldb, beta, C, ldc, cRowOffset, &
                                          cColOffset, distC, &
                                          ctx) bind(C)
    use iso_c_binding
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: kLocal
    integer(c_int), value :: opA
    real(c_float), value :: alpha
    real(c_float), dimension(*), intent(in) :: A
    integer(c_int), value :: lda
    real(c_float), dimension(*), intent(in) :: B
    integer(c_int), value :: ldb
    real(c_float), value :: beta
    real(c_float), dimension(*), intent(inout) ::C
    integer(c_int), value :: ldc
    integer(c_int), value :: cRowOffset
    integer(c_int), value :: cColOffset
    type(c_ptr), value :: distC
    type(c_ptr), value :: ctx
  end function

  integer(c_int) function spla_pdgemm_ssb(m, n, kLocal, opA, &
                                          alpha, A, lda, B, &
                                          ldb, beta, C, ldc, cRowOffset, &
                                          cColOffset, distC, &
                                          ctx) bind(C)
    use iso_c_binding
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: kLocal
    integer(c_int), value :: opA
    real(c_double), value :: alpha
    real(c_double), dimension(*), intent(in) :: A
    integer(c_int), value :: lda
    real(c_double), dimension(*), intent(in) :: B
    integer(c_int), value :: ldb
    real(c_double), value :: beta
    real(c_double), dimension(*), intent(inout) ::C
    integer(c_int), value :: ldc
    integer(c_int), value :: cRowOffset
    integer(c_int), value :: cColOffset
    type(c_ptr), value :: distC
    type(c_ptr), value :: ctx
  end function

  integer(c_int) function spla_pcgemm_ssb(m, n, kLocal, opA, &
                                          alpha, A, lda, B, &
                                          ldb, beta, C, ldc, cRowOffset, &
                                          cColOffset, distC, &
                                          ctx) bind(C)
    use iso_c_binding
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: kLocal
    integer(c_int), value :: opA
    complex(c_float), intent(in) :: alpha
    complex(c_float), dimension(*), intent(in) :: A
    integer(c_int), value :: lda
    complex(c_float), dimension(*), intent(in) :: B
    integer(c_int), value :: ldb
    complex(c_float), intent(in) :: beta
    complex(c_float), dimension(*), intent(inout) ::C
    integer(c_int), value :: ldc
    integer(c_int), value :: cRowOffset
    integer(c_int), value :: cColOffset
    type(c_ptr), value :: distC
    type(c_ptr), value :: ctx
  end function

  integer(c_int) function spla_pzgemm_ssb(m, n, kLocal, opA, &
                                          alpha, A, lda, B, &
                                          ldb, beta, C, ldc, cRowOffset, &
                                          cColOffset, distC, &
                                          ctx) bind(C)
    use iso_c_binding
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: kLocal
    integer(c_int), value :: opA
    complex(c_double), intent(in) :: alpha
    complex(c_double), dimension(*), intent(in) :: A
    integer(c_int), value :: lda
    complex(c_double), dimension(*), intent(in) :: B
    integer(c_int), value :: ldb
    complex(c_double), intent(in) :: beta
    complex(c_double), dimension(*), intent(inout) ::C
    integer(c_int), value :: ldc
    integer(c_int), value :: cRowOffset
    integer(c_int), value :: cColOffset
    type(c_ptr), value :: distC
    type(c_ptr), value :: ctx
  end function



  !--------------------------
  !         pgemm_sbs
  !--------------------------

  integer(c_int) function spla_psgemm_sbs(mLocal, n, k, alpha, A, &
                                          lda, B, ldb, bRowOffset, &
                                          bColOffset, distB, beta, &
                                          C, ldc, ctx) bind(C)
    use iso_c_binding
    integer(c_int), value :: mLocal
    integer(c_int), value :: n
    integer(c_int), value :: k
    real(c_float), value :: alpha
    real(c_float), dimension(*), intent(in) :: A
    integer(c_int), value :: lda
    real(c_float), dimension(*), intent(in) :: B
    integer(c_int), value :: ldb
    integer(c_int), value :: bRowOffset
    integer(c_int), value :: bColOffset
    type(c_ptr), value :: distB
    real(c_float), value :: beta
    real(c_float), dimension(*), intent(inout) ::C
    integer(c_int), value :: ldc
    type(c_ptr), value :: ctx
  end function

  integer(c_int) function spla_pdgemm_sbs(mLocal, n, k, alpha, A, &
                                          lda, B, ldb, bRowOffset, &
                                          bColOffset, distB, beta, &
                                          C, ldc, ctx) bind(C)
    use iso_c_binding
    integer(c_int), value :: mLocal
    integer(c_int), value :: n
    integer(c_int), value :: k
    real(c_double), value :: alpha
    real(c_double), dimension(*), intent(in) :: A
    integer(c_int), value :: lda
    real(c_double), dimension(*), intent(in) :: B
    integer(c_int), value :: ldb
    integer(c_int), value :: bRowOffset
    integer(c_int), value :: bColOffset
    type(c_ptr), value :: distB
    real(c_double), value :: beta
    real(c_double), dimension(*), intent(inout) ::C
    integer(c_int), value :: ldc
    type(c_ptr), value :: ctx
  end function

  integer(c_int) function spla_pcgemm_sbs(mLocal, n, k, alpha, A, &
                                          lda, B, ldb, bRowOffset, &
                                          bColOffset, distB, beta, &
                                          C, ldc, ctx) bind(C)
    use iso_c_binding
    integer(c_int), value :: mLocal
    integer(c_int), value :: n
    integer(c_int), value :: k
    complex(c_float), intent(in) :: alpha
    complex(c_float), dimension(*), intent(in) :: A
    integer(c_int), value :: lda
    complex(c_float), dimension(*), intent(in) :: B
    integer(c_int), value :: ldb
    integer(c_int), value :: bRowOffset
    integer(c_int), value :: bColOffset
    type(c_ptr), value :: distB
    complex(c_float), intent(in) :: beta
    complex(c_float), dimension(*), intent(inout) ::C
    integer(c_int), value :: ldc
    type(c_ptr), value :: ctx
  end function

  integer(c_int) function spla_pzgemm_sbs(mLocal, n, k, alpha, A, &
                                          lda, B, ldb, bRowOffset, &
                                          bColOffset, distB, beta, &
                                          C, ldc, ctx) bind(C)
    use iso_c_binding
    integer(c_int), value :: mLocal
    integer(c_int), value :: n
    integer(c_int), value :: k
    complex(c_double), intent(in) :: alpha
    complex(c_double), dimension(*), intent(in) :: A
    integer(c_int), value :: lda
    complex(c_double), dimension(*), intent(in) :: B
    integer(c_int), value :: ldb
    integer(c_int), value :: bRowOffset
    integer(c_int), value :: bColOffset
    type(c_ptr), value :: distB
    complex(c_double), intent(in) :: beta
    complex(c_double), dimension(*), intent(inout) ::C
    integer(c_int), value :: ldc
    type(c_ptr), value :: ctx
  end function


  !--------------------------
  !         pgemm_sbs
  !--------------------------

  integer(c_int) function spla_sgemm(opA, opB, m, n, k, &
                                     alpha, A, lda, B, ldb, &
                                     beta, C, ldc, ctx) bind(C)
    use iso_c_binding
    integer(c_int), value :: opA
    integer(c_int), value :: opB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    real(c_float), value :: alpha
    real(c_float), dimension(*), intent(in) :: A
    integer(c_int), value :: lda
    real(c_float), dimension(*), intent(in) :: B
    integer(c_int), value :: ldb
    real(c_float), value :: beta
    real(c_float), dimension(*), intent(inout) ::C
    integer(c_int), value :: ldc
    type(c_ptr), value :: ctx
  end function

  integer(c_int) function spla_dgemm(opA, opB, m, n, k, &
                                     alpha, A, lda, B, ldb, &
                                     beta, C, ldc, ctx) bind(C)
    use iso_c_binding
    integer(c_int), value :: opA
    integer(c_int), value :: opB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    real(c_double), value :: alpha
    real(c_double), dimension(*), intent(in) :: A
    integer(c_int), value :: lda
    real(c_double), dimension(*), intent(in) :: B
    integer(c_int), value :: ldb
    real(c_double), value :: beta
    real(c_double), dimension(*), intent(inout) ::C
    integer(c_int), value :: ldc
    type(c_ptr), value :: ctx
  end function

  integer(c_int) function spla_cgemm(opA, opB, m, n, k, &
                                     alpha, A, lda, B, ldb, &
                                     beta, C, ldc, ctx) bind(C)
    use iso_c_binding
    integer(c_int), value :: opA
    integer(c_int), value :: opB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    complex(c_float), intent(in) :: alpha
    complex(c_float), dimension(*), intent(in) :: A
    integer(c_int), value :: lda
    complex(c_float), dimension(*), intent(in) :: B
    integer(c_int), value :: ldb
    complex(c_float), intent(in) :: beta
    complex(c_float), dimension(*), intent(inout) ::C
    integer(c_int), value :: ldc
    type(c_ptr), value :: ctx
  end function

  integer(c_int) function spla_zgemm(opA, opB, m, n, k, &
                                     alpha, A, lda, B, ldb, &
                                     beta, C, ldc, ctx) bind(C)
    use iso_c_binding
    integer(c_int), value :: opA
    integer(c_int), value :: opB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    complex(c_double), intent(in) :: alpha
    complex(c_double), dimension(*), intent(in) :: A
    integer(c_int), value :: lda
    complex(c_double), dimension(*), intent(in) :: B
    integer(c_int), value :: ldb
    complex(c_double), intent(in) :: beta
    complex(c_double), dimension(*), intent(inout) ::C
    integer(c_int), value :: ldc
    type(c_ptr), value :: ctx
  end function

end interface

end