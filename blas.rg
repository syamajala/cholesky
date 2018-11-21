import "regent"
local c = regentlib.c
terralib.includepath = terralib.includepath .. ";/usr/include"

terralib.linklibrary("/usr/lib/libcblas.so")
local blas = terralib.includec("cblas.h")

terralib.linklibrary("/usr/lib/liblapacke.so")
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

terra get_raw_ptr(rect: rect2d,
                  pr : c.legion_physical_region_t,
                  fld : c.legion_field_id_t)
  var fa = c.legion_physical_region_get_field_accessor_array_2d(pr, fld)
  var subrect : c.legion_rect_2d_t
  var offsets : c.legion_byte_offset_t[2]
  var ptr = c.legion_accessor_array_2d_raw_rect_ptr(fa, rect, &subrect, offsets)
  return raw_ptr { ptr = [&double](ptr), offset = offsets[1].offset / sizeof(double) }
end

terra dpotrf_terra(rect: rect2d, m:int,
                   pr : c.legion_physical_region_t,
                   fld : c.legion_field_id_t)
  var rawA = get_raw_ptr(rect, pr, fld)
  var uplo : rawstring = 'L'
  var info = lapack.LAPACKE_dpotrf(blas.CblasColMajor, @uplo, m, rawA.ptr, rawA.offset)
  return info
end

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
  var rawA = get_raw_ptr(rectA, prA, fldA)
  var rawB = get_raw_ptr(rectB, prB, fldB)
  var alpha = 1.0

  blas.cblas_dtrsm(blas.CblasColMajor, blas.CblasRight, blas.CblasLower, blas.CblasTrans, blas.CblasNonUnit, m, n, alpha,
                   rawB.ptr, rawB.offset, rawA.ptr, rawA.offset)
end

task dtrsm(rA : region(ispace(int2d), double),
           rB : region(ispace(int2d), double))
where reads writes(rA), reads(rB)
do
  var rectA = rA.bounds
  var rectB = rB.bounds
  var size:int2d = rectA.hi - rectA.lo + {1, 1}

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

  var rawA = get_raw_ptr(rectA, prA, fldA)
  var rawB = get_raw_ptr(rectB, prB, fldB)
  var rawC = get_raw_ptr(rectC, prC, fldC)

  blas.cblas_dgemm(blas.CblasColMajor, blas.CblasNoTrans, blas.CblasTrans, m, n, k,
                   alpha, rawB.ptr, rawB.offset,
                   rawC.ptr, rawC.offset,
                   beta, rawA.ptr, rawA.offset)
end


task dgemm(rA : region(ispace(int2d), double),
           rB : region(ispace(int2d), double),
           rC : region(ispace(int2d), double))
where reads writes(rA), reads(rB, rC)
do
  var rectA = rA.bounds
  var sizeA:int2d = rectA.hi - rectA.lo + {1, 1}
  var rectB = rB.bounds
  var sizeB:int2d = rectB.hi - rectB.lo + {1, 1}
  var rectC = rC.bounds

  var m = sizeA.x
  var n = sizeB.y
  var k = sizeA.y

  dgemm_terra(rectA, rectB, rectC, m, n, k,
              __physical(rA)[0], __fields(rA)[0],
              __physical(rB)[0], __fields(rB)[0],
              __physical(rC)[0], __fields(rC)[0])
end

task transpose_copy(rSrc : region(ispace(int2d), double), rDst : region(ispace(int2d), double))
where reads(rSrc), writes(rDst)
do
  for p in rSrc.ispace do
    rDst[{ x = p.y, y = p.x }] = rSrc[p]
  end
end

-- task main()

--   var n = 4
--   var np = 2
--   var mat = region(ispace(int2d, {x = n, y = n}), double)

  -- fill(mat, 0)
  -- mat[{0, 0}] = 4.0
  -- mat[{0, 1}] = 3.0
  -- mat[{1, 0}] = 6.0
  -- mat[{1, 1}] = 3.0

  -- mat[{2, 2}] = 8.0
  -- mat[{2, 3}] = 6.0
  -- mat[{3, 2}] = 12.0
  -- mat[{3, 3}] = 6.0

--   var coloring = c.legion_domain_point_coloring_create()

--   var bounds = rect2d{{0, 0}, {1, 1}}
--   var color:int2d = {0, 0}
--   c.legion_domain_point_coloring_color_domain(coloring, color:to_domain_point(), c.legion_domain_from_rect_2d(bounds))

--   bounds = rect2d{{2, 2}, {3, 3}}
--   color = {1, 1}
--   c.legion_domain_point_coloring_color_domain(coloring, color:to_domain_point(), c.legion_domain_from_rect_2d(bounds))

--   var colors = ispace(int2d, {2, 2}, {0, 0})
--   var mat_part = partition(disjoint, mat, coloring, colors)

--   var bn = n / np

--   var part = mat_part[{0,0}]
--   fill(part, 0)
--   var lo = part.bounds.lo
--   part[lo + {0, 0}] = 4.0
--   part[lo + {0, 1}] = 0.0
--   part[lo + {1, 0}] = 6.0
--   part[lo + {1, 1}] = 3.0

--   var info = dpotrf(mat_part[{0, 0}])
--   c.printf("Info %d\n", info)

--   var part2 = mat_part[{1, 1}]
--   fill(part2, 0)
--   var lo2 = part2.bounds.lo
--   part2[lo2 + {0, 0}] = 4.0*2
--   part2[lo2 + {0, 1}] = 0.0
--   part2[lo2 + {1, 0}] = 6.0*2
--   part2[lo2 + {1, 1}] = 3.0*2

--   dpotrf(mat_part[{1, 1}])

--   for color in colors do
--     var part = mat_part[color]
--     var lo = part.bounds.lo
--     var vol = c.legion_domain_get_volume(c.legion_domain_from_rect_2d(part.bounds))
--     var size = part.bounds.hi - part.bounds.lo + {1, 1}
--     c.printf("Color: %d Vol: %d\n", color, vol)
--     for i = 0, size.y do
--       for j = 0, size.x do
--         c.printf("%f ", part[lo + {i, j}])
--       end
--       c.printf('\n')
--     end
--   end

--   for i = 0, n do
--     for j = 0, n do
--       c.printf("%f ", mat[{i, j}])
--     end
--     c.printf('\n')
--   end

-- end

-- regentlib.start(main)
