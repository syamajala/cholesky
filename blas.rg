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

terra get_raw_ptr_2d(rect: rect2d,
                     pr : c.legion_physical_region_t,
                     fld : c.legion_field_id_t)
  var fa = c.legion_physical_region_get_field_accessor_array_2d(pr, fld)
  var subrect : c.legion_rect_2d_t
  var offsets : c.legion_byte_offset_t[2]
  var ptr = c.legion_accessor_array_2d_raw_rect_ptr(fa, rect, &subrect, offsets)
  return raw_ptr { ptr = [&double](ptr), offset = offsets[1].offset / sizeof(double) }
end

terra get_raw_ptr_1d(rect: rect1d,
                     pr : c.legion_physical_region_t,
                     fld : c.legion_field_id_t)
  var fa = c.legion_physical_region_get_field_accessor_array_1d(pr, fld)
  var subrect : c.legion_rect_1d_t
  var offsets : c.legion_byte_offset_t[1]
  var ptr = c.legion_accessor_array_1d_raw_rect_ptr(fa, rect, &subrect, offsets)
  return raw_ptr { ptr = [&double](ptr), offset = offsets[0].offset / sizeof(double) }
end

terra dpotrf_terra(rect: rect2d, m:int,
                   pr : c.legion_physical_region_t,
                   fld : c.legion_field_id_t)
  var rawA = get_raw_ptr_2d(rect, pr, fld)
  var uplo : rawstring = 'L'
  var info = lapack.LAPACKE_dpotrf(cblas.CblasColMajor, @uplo, m, rawA.ptr, rawA.offset)
  return info
end

__demand(__leaf)
task dpotrf(rA : region(ispace(int2d), double))
where reads writes(rA)
do
  var rect = rA.bounds
  var size:int2d = rect.hi - rect.lo + {1, 1}
  return dpotrf_terra(rect, size.x, __physical(rA)[0], __fields(rA)[0])
end

terra dtrsm_terra(rectA:rect2d, rectB:rect2d, m:int, n:int,
                  prA : c.legion_physical_region_t,
                  fldA : c.legion_field_id_t,
                  prB : c.legion_physical_region_t,
                  fldB : c.legion_field_id_t)
  var rawA = get_raw_ptr_2d(rectA, prA, fldA)
  var rawB = get_raw_ptr_2d(rectB, prB, fldB)
  var alpha = 1.0

  cblas.cblas_dtrsm(cblas.CblasColMajor, cblas.CblasRight, cblas.CblasLower, cblas.CblasTrans, cblas.CblasNonUnit, m, n, alpha,
                    rawA.ptr, rawA.offset, rawB.ptr, rawB.offset)
end

__demand(__leaf)
task dtrsm(rA : region(ispace(int2d), double),
           rB : region(ispace(int2d), double))
where reads(rA), reads writes(rB)
do
  var rectA = rA.bounds
  var rectB = rB.bounds
  var size:int2d = rectB.hi - rectB.lo + {1, 1}

  dtrsm_terra(rectA, rectB, size.x, size.y,
              __physical(rA)[0], __fields(rA)[0],
              __physical(rB)[0], __fields(rB)[0])
end

terra dgemm_terra(rectA:rect2d, rectB:rect2d, rectC:rect2d,
                  m:int, n:int, k:int,
                  prA : c.legion_physical_region_t,
                  fldA : c.legion_field_id_t,
                  prB : c.legion_physical_region_t,
                  fldB : c.legion_field_id_t,
                  prC : c.legion_physical_region_t,
                  fldC : c.legion_field_id_t)

  var alpha = -1.0
  var beta = 1.0

  var rawA = get_raw_ptr_2d(rectA, prA, fldA)
  var rawB = get_raw_ptr_2d(rectB, prB, fldB)
  var rawC = get_raw_ptr_2d(rectC, prC, fldC)

  cblas.cblas_dgemm(cblas.CblasColMajor, cblas.CblasNoTrans, cblas.CblasTrans, m, n, k,
                   alpha, rawA.ptr, rawA.offset,
                   rawB.ptr, rawB.offset,
                   beta, rawC.ptr, rawC.offset)
end

__demand(__leaf)
task dgemm(rA : region(ispace(int2d), double),
           rB : region(ispace(int2d), double),
           rC : region(ispace(int2d), double))
where reads(rA, rB), reads writes(rC)
do
  var rectA = rA.bounds
  var sizeA:int2d = rectA.hi - rectA.lo + {1, 1}
  var rectB = rB.bounds
  var sizeB:int2d = rectB.hi - rectB.lo + {1, 1}
  var rectC = rC.bounds
  var sizeC:int2d = rectC.hi - rectC.lo + {1, 1}

  var m = sizeC.x
  var n = sizeC.y
  var k = sizeA.y

  dgemm_terra(rectA, rectB, rectC, m, n, k,
              __physical(rA)[0], __fields(rA)[0],
              __physical(rB)[0], __fields(rB)[0],
              __physical(rC)[0], __fields(rC)[0])
end

terra dsyrk_terra(rectA:rect2d, rectC:rect2d,
                  n:int, k:int,
                  prA : c.legion_physical_region_t,
                  fldA : c.legion_field_id_t,
                  prC : c.legion_physical_region_t,
                  fldC : c.legion_field_id_t)

  var alpha = -1.0
  var beta = 1.0

  var rawA = get_raw_ptr_2d(rectA, prA, fldA)
  var rawC = get_raw_ptr_2d(rectC, prC, fldC)

  cblas.cblas_dsyrk(cblas.CblasColMajor, cblas.CblasLower, cblas.CblasNoTrans, n, k,
                   alpha, rawA.ptr, rawA.offset,
                   beta, rawC.ptr, rawC.offset)

end

__demand(__leaf)
task dsyrk(rA : region(ispace(int2d), double),
           rC : region(ispace(int2d), double))
where reads(rA), reads writes(rC)
do
  var rectA = rA.bounds
  var sizeA:int2d = rectA.hi - rectA.lo + {1, 1}
  var rectC = rC.bounds
  var sizeC:int2d = rectA.hi - rectA.lo + {1, 1}

  var n = sizeC.x
  var k = sizeA.y

  dsyrk_terra(rectA, rectC, n, k,
              __physical(rA)[0], __fields(rA)[0],
              __physical(rC)[0], __fields(rC)[0])
end


terra dtrsv_terra(rectA:rect2d, rectB:rect1d, uplo:int, trans:int, n:int,
                  prA : c.legion_physical_region_t,
                  fldA : c.legion_field_id_t,
                  prB : c.legion_physical_region_t,
                  fldB : c.legion_field_id_t)

  var rawA = get_raw_ptr_2d(rectA, prA, fldA)
  var rawB = get_raw_ptr_1d(rectB, prB, fldB)

  cblas.cblas_dtrsv(cblas.CblasColMajor, uplo, trans, cblas.CblasNonUnit, n,
                    rawA.ptr, rawA.offset, rawB.ptr, rawB.offset)

end

__demand(__leaf)
task dtrsv(rA : region(ispace(int2d), double), rB : region(ispace(int1d), double), uplo:int, trans:int)
where reads(rA), reads writes(rB)
do
  var rectA = rA.bounds
  var sizeA:int2d = rectA.hi - rectA.lo + {1, 1}
  var rectB = rB.bounds

  var n = sizeA.x

  dtrsv_terra(rectA, rectB, uplo, trans, n,
              __physical(rA)[0], __fields(rA)[0],
              __physical(rB)[0], __fields(rB)[0])
end

terra dgemv_terra(rectA:rect2d, rectX:rect1d, rectY:rect1d, trans:int, m:int, n:int,
                  prA : c.legion_physical_region_t,
                  fldA : c.legion_field_id_t,
                  prX : c.legion_physical_region_t,
                  fldX : c.legion_field_id_t,
                  prY : c.legion_physical_region_t,
                  fldY : c.legion_field_id_t)

  var rawA = get_raw_ptr_2d(rectA, prA, fldA)
  var rawX = get_raw_ptr_1d(rectX, prX, fldX)
  var rawY = get_raw_ptr_1d(rectY, prY, fldY)

  var alpha = -1.0
  var beta = 1.0

  cblas.cblas_dgemv(cblas.CblasColMajor, trans, m, n, alpha,
                    rawA.ptr, rawA.offset, rawX.ptr, rawX.offset, beta, rawY.ptr, rawY.offset)

end

__demand(__leaf)
task dgemv(rA : region(ispace(int2d), double),
           rX : region(ispace(int1d), double),
           rY : region(ispace(int1d), double),
          trans:int)
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
