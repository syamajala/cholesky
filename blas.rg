-- Copyright 2019 Stanford University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

import "regent"
local c = regentlib.c

terralib.linklibrary("libcblas.so")
local cblas = terralib.includec("cblas.h")

terralib.linklibrary("liblapacke.so")
local lapack = terralib.includec("lapacke.h")

function raw_ptr_factory(ty)
  local struct raw_ptr
  {
    ptr : &ty,
    offset : int,
  }
  return raw_ptr
end

local raw_ptr = raw_ptr_factory(double)

terra get_raw_ptr_2d(rect : rect2d,
                     pr   : c.legion_physical_region_t,
                     fld  : c.legion_field_id_t)
  var fa = c.legion_physical_region_get_field_accessor_array_2d(pr, fld)
  var subrect : c.legion_rect_2d_t
  var offsets : c.legion_byte_offset_t[2]
  var ptr = c.legion_accessor_array_2d_raw_rect_ptr(fa, rect, &subrect, offsets)
  return raw_ptr { ptr = [&double](ptr), offset = offsets[1].offset / sizeof(double) }
end

terra get_raw_ptr_1d(rect : rect1d,
                     pr   : c.legion_physical_region_t,
                     fld  : c.legion_field_id_t)
  var fa = c.legion_physical_region_get_field_accessor_array_1d(pr, fld)
  var subrect : c.legion_rect_1d_t
  var offsets : c.legion_byte_offset_t[1]
  var ptr = c.legion_accessor_array_1d_raw_rect_ptr(fa, rect, &subrect, offsets)
  return raw_ptr { ptr = [&double](ptr), offset = offsets[0].offset / sizeof(double) }
end

fspace Filled {
  filled   : int1d,
  sep      : int2d,
  interval : int1d,
  cluster  : int1d,
  bounds   : rect2d
}

terra dpotrf_terra(rect : rect2d, m:int,
                   pr   : c.legion_physical_region_t,
                   fld  : c.legion_field_id_t,
                   block : int2d,
                   level : int,
                   interval : int)
  var rawA = get_raw_ptr_2d(rect, pr, fld)
  var uplo : rawstring = 'L'
  if m ~= 0 then
    var start = c.legion_get_current_time_in_micros()

    lapack.LAPACKE_dpotrf(cblas.CblasColMajor, @uplo, m, rawA.ptr, rawA.offset)


    -- c.printf("BLAS: {'op': 'POTRF', 'M': %d, 'Time': %lu}\n", m, stop - start)
    var stop = c.legion_get_current_time_in_micros()
    c.printf("Timing: {'op': 'POTRF BLAS', 'Block': (%d, %d), 'Level': %d, 'Interval': %d, 'Time': %lu}\n",
             block.__ptr.x, block.__ptr.y, level, interval, stop - start)
  end
end

terra dtrsm_terra(rectA : rect2d, rectB : rect2d, m : int, n : int,
                  prA   : c.legion_physical_region_t,
                  fldA  : c.legion_field_id_t,
                  prB   : c.legion_physical_region_t,
                  fldB  : c.legion_field_id_t,
                  block : int2d,
                  level : int,
                  interval : int)
  var rawA = get_raw_ptr_2d(rectA, prA, fldA)
  var rawB = get_raw_ptr_2d(rectB, prB, fldB)
  var alpha = 1.0

  var start = c.legion_get_current_time_in_micros()

  cblas.cblas_dtrsm(cblas.CblasColMajor, cblas.CblasRight, cblas.CblasLower, cblas.CblasTrans, cblas.CblasNonUnit,
                    m, n, alpha, rawA.ptr, rawA.offset, rawB.ptr, rawB.offset)

  var stop = c.legion_get_current_time_in_micros()
  c.printf("Timing: {'op': 'TRSM BLAS', 'Block': (%d, %d), 'Level': %d, 'Interval': %d, 'Time': %lu}\n",
           block.__ptr.x, block.__ptr.y, level, interval, stop - start)
end

terra dgemm_terra(rectA : rect2d, rectB : rect2d, rectC : rect2d, m : int, n : int, k : int,
                  prA   : c.legion_physical_region_t,
                  fldA  : c.legion_field_id_t,
                  prB   : c.legion_physical_region_t,
                  fldB  : c.legion_field_id_t,
                  prC   : c.legion_physical_region_t,
                  fldC  : c.legion_field_id_t,
                  block : int2d,
                  level : int,
                  interval : int)

  var alpha = -1.0
  var beta = 1.0

  var rawA = get_raw_ptr_2d(rectA, prA, fldA)
  var rawB = get_raw_ptr_2d(rectB, prB, fldB)
  var rawC = get_raw_ptr_2d(rectC, prC, fldC)

  var start = c.legion_get_current_time_in_micros()

  cblas.cblas_dgemm(cblas.CblasColMajor, cblas.CblasNoTrans, cblas.CblasTrans, m, n, k,
                    alpha, rawA.ptr, rawA.offset,
                    rawB.ptr, rawB.offset,
                    beta, rawC.ptr, rawC.offset)

  -- c.printf("BLAS: {'op': 'GEMM', 'M': %d, 'N': %d, 'K': %d, 'Time': %lu}\n", m, n, k, stop - start)
  var stop = c.legion_get_current_time_in_micros()
  c.printf("Timing: {'op': 'GEMM BLAS', 'Block': (%d, %d), 'Level': %d, 'Interval': %d, 'Time': %lu}\n",
           block.__ptr.x, block.__ptr.y, level, interval, stop - start)
end

terra dsyrk_terra(rectA : rect2d, rectC : rect2d, n : int, k : int,
                  prA   : c.legion_physical_region_t,
                  fldA  : c.legion_field_id_t,
                  prC   : c.legion_physical_region_t,
                  fldC  : c.legion_field_id_t,
                  block : int2d,
                  level : int,
                  interval : int)

  var alpha = -1.0
  var beta = 1.0

  var rawA = get_raw_ptr_2d(rectA, prA, fldA)
  var rawC = get_raw_ptr_2d(rectC, prC, fldC)

  var start = c.legion_get_current_time_in_micros()

  cblas.cblas_dsyrk(cblas.CblasColMajor, cblas.CblasLower, cblas.CblasNoTrans, n, k,
                   alpha, rawA.ptr, rawA.offset,
                   beta, rawC.ptr, rawC.offset)

  -- c.printf("BLAS: {'op': 'SYRK', 'N': %d, 'K': %d, 'Time': %lu}\n", n, k, stop - start)
  var stop = c.legion_get_current_time_in_micros()
  c.printf("Timing: {'op': 'GEMM BLAS', 'Block': (%d, %d), 'Level': %d, 'Interva': %d, 'Time': %lu}\n",
           block.__ptr.x, block.__ptr.y, level, interval, stop - start)

end

terra dtrsv_terra(rectA : rect2d, rectB : rect1d, uplo : int, trans : int, n : int,
                  prA   : c.legion_physical_region_t,
                  fldA  : c.legion_field_id_t,
                  prB   : c.legion_physical_region_t,
                  fldB  : c.legion_field_id_t)

  var rawA = get_raw_ptr_2d(rectA, prA, fldA)
  var rawB = get_raw_ptr_1d(rectB, prB, fldB)

  cblas.cblas_dtrsv(cblas.CblasColMajor, uplo, trans, cblas.CblasNonUnit, n,
                    rawA.ptr, rawA.offset, rawB.ptr, rawB.offset)

end

__demand(__leaf)
task dtrsv(rA : region(ispace(int2d), double), rB : region(ispace(int1d), double), uplo : int, trans : int)
where
  reads(rA),
  reads writes(rB)
do
  var rectA = rA.bounds
  var sizeA:int2d = rectA.hi - rectA.lo + {1, 1}
  var rectB = rB.bounds

  var n = sizeA.x

  dtrsv_terra(rectA, rectB, uplo, trans, n,
              __physical(rA)[0], __fields(rA)[0],
              __physical(rB)[0], __fields(rB)[0])
end

terra dgemv_terra(rectA : rect2d, rectX : rect1d, rectY : rect1d, trans : int, m : int, n : int,
                  prA   : c.legion_physical_region_t,
                  fldA  : c.legion_field_id_t,
                  prX   : c.legion_physical_region_t,
                  fldX  : c.legion_field_id_t,
                  prY   : c.legion_physical_region_t,
                  fldY  : c.legion_field_id_t)

  var rawA = get_raw_ptr_2d(rectA, prA, fldA)
  var rawX = get_raw_ptr_1d(rectX, prX, fldX)
  var rawY = get_raw_ptr_1d(rectY, prY, fldY)

  var alpha = -1.0
  var beta = 1.0

  cblas.cblas_dgemv(cblas.CblasColMajor, trans, m, n, alpha,
                    rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)

end

__demand(__leaf)
task dgemv(rA    : region(ispace(int2d), double),
           rX    : region(ispace(int1d), double),
           rY    : region(ispace(int1d), double),
           trans : int)
where reads(rA, rX), reads writes(rY)
do
  var rectA = rA.bounds
  var sizeA:int2d = rectA.hi - rectA.lo + {1, 1}
  var rectX = rX.bounds
  var rectY = rY.bounds

  var m:int = 0
  var n:int = 0

  m = sizeA.x
  n = sizeA.y

  dgemv_terra(rectA, rectX, rectY, trans, m, n,
              __physical(rA)[0], __fields(rA)[0],
              __physical(rX)[0], __fields(rX)[0],
              __physical(rY)[0], __fields(rY)[0])
end

__demand(__leaf)
task fused_dpotrf(rA        : region(ispace(int2d), double),
                  filled_rA : region(ispace(ptr), Filled),
                  level     : int,
                  interval  : int,
                  debug     : bool)
where
  reads writes(rA), reads(filled_rA)
do
  var potrf_start = c.legion_get_current_time_in_micros()
  var block : int2d

  for i in filled_rA.ispace do
    var a = filled_rA[i]
    var color = int3d{a.sep.x, a.sep.y, a.cluster}
    var rectA = a.bounds
    var size:int2d = rectA.hi - rectA.lo + {1, 1}
    block = int2d{a.sep.x, a.sep.y}

    if debug then
      c.printf("POTRF: {'A': (%d, %d, %d), 'A_Lo': (%d, %d), 'A_Hi': (%d, %d), 'SizeA': (%d, %d), 'Block': (%d, %d), 'Level': %d, 'Interval': %d}\n",
               color.x, color.y, color.z, rectA.lo.x, rectA.lo.y, rectA.hi.x, rectA.hi.y, size.x, size.y,
               color.x, color.y, level, interval)
    end

    var start = c.legion_get_current_time_in_micros()

    dpotrf_terra(rectA, size.x, __physical(rA)[0], __fields(rA)[0], block, level, interval)

    var stop = c.legion_get_current_time_in_micros()
    c.printf("Timing: {'op': 'POTRF Terra', 'Block': (%d, %d), 'Level': %d, 'Interval': %d, 'Time': %lu}\n",
             block.x, block.y, level, interval, stop - start)
  end

  var potrf_stop = c.legion_get_current_time_in_micros()
  c.printf("Timing: {'op': 'POTRF Task', 'Block': (%d, %d), 'Level': %d, 'Interval': %d, 'Time': %lu}\n",
           block.x, block.y, level, interval, potrf_stop - potrf_start)
end

__demand(__leaf)
task fused_dtrsm(rA        : region(ispace(int2d), double),
                 rB        : region(ispace(int2d), double),
                 filled_rA : region(ispace(ptr), Filled),
                 filled_rB : region(ispace(ptr), Filled),
                 level     : int,
                 interval  : int,
                 debug     : bool)
where
  reads(rA, filled_rA, filled_rB), reads writes(rB)
do
  var trsm_start = c.legion_get_current_time_in_micros()
  var block : int2d

  for i in filled_rA.ispace do
    var a = filled_rA[i]
    var Acolor = int3d{a.sep.x, a.sep.y, a.cluster}
    var rectA = a.bounds
    var sizeA = rectA.hi - rectA.lo + {1, 1}
    for j in filled_rB.ispace do
      var b = filled_rB[j]
      var Bcolor = int3d{b.sep.x, b.sep.y, b.cluster}
      var rectB = b.bounds
      var sizeB:int2d = rectB.hi - rectB.lo + {1, 1}
      block = int2d{b.sep.x, b.sep.y}

      if debug then
        c.printf("TRSM: {'A': (%d, %d, %d), 'A_Lo': (%d, %d), 'A_Hi': (%d, %d), 'SizeA': (%d, %d), 'B': (%d, %d, %d), 'B_Lo': (%d, %d), 'B_Hi': (%d, %d), 'SizeB': (%d, %d), 'Block': (%d, %d), 'Level': %d, 'Interval': %d}\n",
                 Acolor.x, Acolor.y, Acolor.z, rectA.lo.x, rectA.lo.y, rectA.hi.x, rectA.hi.y, sizeA.x, sizeA.y,
                 Bcolor.x, Bcolor.y, Bcolor.z, rectB.lo.x, rectB.lo.y, rectB.hi.x, rectB.hi.y, sizeB.x, sizeB.y,
                 Bcolor.x, Bcolor.y, level, interval)
      end

      var start = c.legion_get_current_time_in_micros()

      dtrsm_terra(rectA, rectB, sizeB.x, sizeB.y,
                  __physical(rA)[0], __fields(rA)[0],
                  __physical(rB)[0], __fields(rB)[0],
                  block, level, interval)

      var stop = c.legion_get_current_time_in_micros()
      c.printf("Timing: {'op': 'TRSM Terra', 'Block': (%d, %d), 'Level': %d, 'Interval': %d, 'Time': %lu}\n",
               block.x, block.y,
               level, interval, stop - start)
    end
  end

  var trsm_stop = c.legion_get_current_time_in_micros()
  c.printf("Timing: {'op': 'TRSM Task', 'Block': (%d, %d), 'Level': %d, 'Interval': %d, 'Time': %lu}\n",
           block.x, block.y, level, interval, trsm_stop - trsm_start)
end

__demand(__leaf)
task fused_dsyrk(rA               : region(ispace(int2d), double),
                 rB               : region(ispace(int2d), double),
                 rC               : region(ispace(int2d), double),
                 filled_rA        : region(ispace(ptr), Filled),
                 filled_rB        : region(ispace(ptr), Filled),
                 filled_rC        : region(ispace(ptr), Filled),
                 col_cluster_size : int,
                 level            : int,
                 interval         : int,
                 debug            : bool)
where
  reads(rA, rB, filled_rA, filled_rB, filled_rC),
  reads writes(rC)
do
  var syrk_start = c.legion_get_current_time_in_micros()
  var block : int2d

  for i in filled_rA.ispace do
    var a = filled_rA[i]
    var Acolor = int3d{a.sep.x, a.sep.y, a.cluster}
    var row = Acolor.z
    var rectA = a.bounds
    var sizeA:int2d = rectA.hi - rectA.lo + {1, 1}

    for j in filled_rB.ispace do
      var b = filled_rB[j]
      var Bcolor = int3d{b.sep.x, b.sep.y, b.cluster}
      var col = Bcolor.z
      var rectB = b.bounds
      var sizeB:int2d = rectB.hi - rectB.lo + {1, 1}

      var Ccolor = int3d{Acolor.x, Bcolor.x, row*col_cluster_size+col}
      var rectC = rect2d{lo=int2d{0, 0}, hi={int2d{-1, -1}}}
      block = int2d{Ccolor.x, Ccolor.y}

      for k in filled_rC.ispace do
        var p = filled_rC[k]
        var clust = int3d{p.sep.x, p.sep.y, p.cluster}
        if clust == Ccolor then
          rectC = p.bounds
          break
        end
      end

      var vol = c.legion_domain_get_volume(c.legion_domain_from_rect_2d(rectC))
      if vol ~= 0 then

        var sizeC:int2d = rectC.hi - rectC.lo + {1, 1}

        if col < row then
          var m = sizeA.x
          var n = sizeB.x
          var k = sizeA.y

          if debug then
            c.printf("GEMM: {'A': (%d, %d, %d), 'A_Lo': (%d, %d), 'A_Hi': (%d, %d), 'sizeA': (%d, %d), 'B': (%d, %d, %d), 'B_Lo': (%d, %d), 'B_Hi': (%d, %d), 'sizeB': (%d, %d), 'C': (%d, %d, %d), 'C_Lo': (%d, %d), 'C_Hi': (%d, %d), 'sizeC': (%d, %d), 'Block': (%d, %d), 'Level': %d, 'Interval': %d}\n",
                     Acolor.x, Acolor.y, Acolor.z, rectA.lo.x, rectA.lo.y, rectA.hi.x, rectA.hi.y, sizeA.x, sizeA.y,
                     Bcolor.x, Bcolor.y, Bcolor.z, rectB.lo.x, rectB.lo.y, rectB.hi.x, rectB.hi.y, sizeB.x, sizeB.y,
                     Ccolor.x, Ccolor.y, Ccolor.z, rectC.lo.x, rectC.lo.y, rectC.hi.x, rectC.hi.y, sizeC.x, sizeC.y,
                     Ccolor.x, Ccolor.y, level, interval)
          end

          var start = c.legion_get_current_time_in_micros()

          dgemm_terra(rectA, rectB, rectC, m, n, k,
                      __physical(rA)[0], __fields(rA)[0],
                      __physical(rB)[0], __fields(rB)[0],
                      __physical(rC)[0], __fields(rC)[0],
                      block, level, interval)

          var stop = c.legion_get_current_time_in_micros()
          c.printf("Timing: {'op': 'GEMM Terra', 'Block': (%d, %d), 'Level': %d, 'Interval': %d, 'Time': %lu}\n",
                   block.x, block.y, level, interval, stop - start)

        elseif col == row then
          var n = sizeC.x
          var k = sizeA.y

          if debug then
            c.printf("GEMM: {'A': (%d, %d, %d), 'A_Lo': (%d, %d), 'A_Hi': (%d, %d), 'sizeA': (%d, %d), 'B': (%d, %d, %d), 'B_Lo': (%d, %d), 'B_Hi': (%d, %d), 'sizeB': (%d, %d), 'C': (%d, %d, %d), 'C_Lo': (%d, %d), 'C_Hi': (%d, %d), 'sizeC': (%d, %d), 'Block': (%d, %d), 'Level': %d, 'Interval': %d}\n",
                     Acolor.x, Acolor.y, Acolor.z, rectA.lo.x, rectA.lo.y, rectA.hi.x, rectA.hi.y, sizeA.x, sizeA.y,
                     Bcolor.x, Bcolor.y, Bcolor.z, rectB.lo.x, rectB.lo.y, rectB.hi.x, rectB.hi.y, sizeB.x, sizeB.y,
                     Ccolor.x, Ccolor.y, Ccolor.z, rectC.lo.x, rectC.lo.y, rectC.hi.x, rectC.hi.y, sizeC.x, sizeC.y,
                     Ccolor.x, Ccolor.y, level, interval)
          end

          var start = c.legion_get_current_time_in_micros()

          dsyrk_terra(rectA, rectC, n, k,
                      __physical(rA)[0], __fields(rA)[0],
                      __physical(rC)[0], __fields(rC)[0],
                      block, level, interval)

          var stop = c.legion_get_current_time_in_micros()
          c.printf("Timing: {'op': 'GEMM Terra', 'Block': (%d, %d), 'Level': %d, 'Interval': %d, 'Time': %lu}\n",
                   block.x, block.y, level, interval, stop - start)

        end
      end
    end
  end

  var syrk_stop = c.legion_get_current_time_in_micros()
  c.printf("Timing: {'op': 'GEMM Task', 'Block': (%d, %d), 'Level': %d, 'Interval': %d, 'Time': %lu}\n",
           block.x, block.y, level, interval, syrk_stop - syrk_start)
end

__demand(__leaf)
task fused_dgemm(rA               : region(ispace(int2d), double),
                 rB               : region(ispace(int2d), double),
                 rC               : region(ispace(int2d), double),
                 filled_rA        : region(ispace(ptr), Filled),
                 filled_rB        : region(ispace(ptr), Filled),
                 filled_rC        : region(ispace(ptr), Filled),
                 col_cluster_size : int,
                 level            : int,
                 interval         : int,
                 debug            : bool)
where
  reads(rA, rB, filled_rA, filled_rB, filled_rC),
  reads writes(rC)
do
  var gemm_start = c.legion_get_current_time_in_micros()
  var block : int2d

  for i in filled_rA.ispace do
    var a = filled_rA[i]
    var Acolor = int3d{a.sep.x, a.sep.y, a.cluster}
    var row = Acolor.z
    var rectA = a.bounds
    var sizeA:int2d = rectA.hi - rectA.lo + {1, 1}

    for j in filled_rB.ispace do
      var b = filled_rB[j]
      var Bcolor = int3d{b.sep.x, b.sep.y, b.cluster}
      var col = Bcolor.z

      var Ccolor = int3d{Acolor.x, Bcolor.x, row*col_cluster_size+col}
      var rectC = rect2d{lo=int2d{0, 0}, hi={int2d{-1, -1}}}
      block = int2d{Ccolor.x, Ccolor.y}

      for k in filled_rC.ispace do
        var p = filled_rC[k]
        var clust = int3d{p.sep.x, p.sep.y, p.cluster}
        if clust == Ccolor then
          rectC = p.bounds
          break
        end
      end

      var vol = c.legion_domain_get_volume(c.legion_domain_from_rect_2d(rectC))
      if vol ~= 0 then

        var sizeC:int2d = rectC.hi - rectC.lo + {1, 1}

        var rectB = b.bounds
        var sizeB:int2d = rectB.hi - rectB.lo + {1, 1}

        var m = sizeA.x
        var n = sizeB.x
        var k = sizeA.y

        if debug then
          c.printf("GEMM: {'A': (%d, %d, %d), 'A_Lo': (%d, %d), 'A_Hi': (%d, %d), 'sizeA': (%d, %d), 'B': (%d, %d, %d), 'B_Lo': (%d, %d), 'B_Hi': (%d, %d), 'sizeB': (%d, %d), 'C': (%d, %d, %d), 'C_Lo': (%d, %d), 'C_Hi': (%d, %d), 'sizeC': (%d, %d), 'Block': (%d, %d), 'Level': %d, 'Interval': %d}\n",
                   Acolor.x, Acolor.y, Acolor.z, rectA.lo.x, rectA.lo.y, rectA.hi.x, rectA.hi.y, sizeA.x, sizeA.y,
                   Bcolor.x, Bcolor.y, Bcolor.z, rectB.lo.x, rectB.lo.y, rectB.hi.x, rectB.hi.y, sizeB.x, sizeB.y,
                   Ccolor.x, Ccolor.y, Ccolor.z, rectC.lo.x, rectC.lo.y, rectC.hi.x, rectC.hi.y, sizeC.x, sizeC.y,
                   Ccolor.x, Ccolor.y, level, interval)
        end

        var start = c.legion_get_current_time_in_micros()

        dgemm_terra(rectA, rectB, rectC, m, n, k,
                    __physical(rA)[0], __fields(rA)[0],
                    __physical(rB)[0], __fields(rB)[0],
                    __physical(rC)[0], __fields(rC)[0],
                    block, level, interval)

        var stop = c.legion_get_current_time_in_micros()
        c.printf("Timing: {'op': 'GEMM Terra', 'Block': (%d, %d), 'Level': %d, 'Interval': %d, 'Time': %lu}\n",
                 block.x, block.y, level, interval, stop - start)
      end
    end
  end

  var gemm_stop = c.legion_get_current_time_in_micros()
  c.printf("Timing: {'op': 'GEMM Task', 'Block': (%d, %d), 'Level': %d, 'Interval': %d, 'Time': %lu}\n",
           block.x, block.y, level, interval, gemm_stop - gemm_start)
end
