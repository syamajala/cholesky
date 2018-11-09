import "regent"
local c = regentlib.c
terralib.includepath = terralib.includepath .. ";/usr/include"

terralib.linklibrary("/usr/lib/libcblas.so")
local blas = terralib.includec("cblas.h")

terralib.linklibrary("/usr/lib/liblapacke.so")
local lapack = terralib.includec("lapacke.h")

local struct __f2d { y : int, x : int }
local f2d = regentlib.index_type(__f2d, "f2d")

function raw_ptr_factory(ty)
  local struct raw_ptr
  {
    ptr : &ty,
    offset : int,
  }
  return raw_ptr
end

local raw_ptr = raw_ptr_factory(double)

terra get_raw_ptr(y : int, x : int, bn : int,
                  pr : c.legion_physical_region_t,
                  fld : c.legion_field_id_t)
  var fa = c.legion_physical_region_get_field_accessor_array_2d(pr, fld)
  var rect : c.legion_rect_2d_t
  var subrect : c.legion_rect_2d_t
  var offsets : c.legion_byte_offset_t[2]
  rect.lo.x[0] = y * bn
  rect.lo.x[1] = x * bn
  rect.hi.x[0] = (y + 1) * bn - 1
  rect.hi.x[1] = (x + 1) * bn - 1
  var ptr = c.legion_accessor_array_2d_raw_rect_ptr(fa, rect, &subrect, offsets)
  return raw_ptr { ptr = [&double](ptr), offset = offsets[1].offset / sizeof(double) }
end

terra dpotrf_terra(x : int, bn : int,
                   pr : c.legion_physical_region_t,
                   fld : c.legion_field_id_t)
  var rawA = get_raw_ptr(x, x, bn, pr, fld)
  var uplo : rawstring = 'L'
  var info = lapack.LAPACKE_dpotrf(blas.CblasColMajor, @uplo, bn, rawA.ptr, rawA.offset)
end

task dpotrf(x : int, bn : int, rA : region(ispace(f2d), double))
where reads writes(rA)
do
  dpotrf_terra(x, bn, __physical(rA)[0], __fields(rA)[0])
end

terra dtrsm_terra(x : int, y : int, bn : int,
                  prA : c.legion_physical_region_t,
                  fldA : c.legion_field_id_t,
                  prB : c.legion_physical_region_t,
                  fldB : c.legion_field_id_t)

  var rawA = get_raw_ptr(y, x, bn, prA, fldA)
  var rawB = get_raw_ptr(x, x, bn, prB, fldB)
  var alpha = 1.0

  blas.cblas_dtrsm(blas.CblasColMajor, blas.CblasRight, blas.CblasLower, blas.CblasTrans, blas.CblasNonUnit, bn, bn, alpha,
                   rawB.ptr, rawB.offset, rawA.ptr, rawA.offset)
end

task dtrsm(x : int, y : int, bn : int,
           rA : region(ispace(f2d), double),
           rB : region(ispace(f2d), double))
where reads writes(rA), reads(rB)
do
  dtrsm_terra(x, y, bn,
              __physical(rA)[0], __fields(rA)[0],
              __physical(rB)[0], __fields(rB)[0])
end

terra dgemm_terra(x : int, y : int, k : int, bn : int,
                  prA : c.legion_physical_region_t,
                  fldA : c.legion_field_id_t,
                  prB : c.legion_physical_region_t,
                  fldB : c.legion_field_id_t,
                  prC : c.legion_physical_region_t,
                  fldC : c.legion_field_id_t)

  var alpha = -1.0
  var beta = 1.0

  var rawA = get_raw_ptr(y, k, bn, prA, fldA)
  var rawB = get_raw_ptr(y, x, bn, prB, fldB)
  var rawC = get_raw_ptr(k, x, bn, prC, fldC)

  blas.cblas_dgemm(blas.CblasColMajor, blas.CblasNoTrans, blas.CblasTrans, bn, bn, bn,
                   alpha, rawB.ptr, rawB.offset,
                   rawC.ptr, rawC.offset,
                   beta, rawA.ptr, rawA.offset)
end


task dgemm(x : int, y : int, k : int, bn : int,
           rA : region(ispace(f2d), double),
           rB : region(ispace(f2d), double),
           rC : region(ispace(f2d), double))
where reads writes(rA), reads(rB, rC)
do
  dgemm_terra(x, y, k, bn,
              __physical(rA)[0], __fields(rA)[0],
              __physical(rB)[0], __fields(rB)[0],
              __physical(rC)[0], __fields(rC)[0])
end

-- task main()

--   var n = 4
--   var np = 1
--   var mat = region(ispace(f2d, {x = n, y = n}), double)

--   fill(mat, 0)

--   mat[{0, 0}] = 4.0
--   mat[{0, 1}] = 3.0

--   mat[{1, 0}] = 6.0
--   mat[{1, 1}] = 3.0

--   var coloring = c.legion_domain_coloring_create()
--   var bounds = rect2d{{0, 0}, {1, 1}}
--   c.legion_domain_coloring_color_domain(coloring, 0, bounds)
--   var mat_part = partition(disjoint, mat, coloring)

--   var bn = n / np
--   var x = 0
--   dpotrf(0, bn, mat_part[0])

--   for i = 0, n do
--     for j = 0, n do
--       c.printf("%f ", mat[{i, j}])
--     end
--     c.printf('\n')
--   end
-- end


-- regentlib.start(main)
