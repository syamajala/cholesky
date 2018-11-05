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

terra get_raw_ptr(pr : c.legion_physical_region_t,
                  fld : c.legion_field_id_t)
  var fa = c.legion_physical_region_get_field_accessor_array_2d(pr, fld)
  var rect : c.legion_rect_2d_t
  var subrect : c.legion_rect_2d_t
  var offsets : c.legion_byte_offset_t[2]
  rect.lo.x[0] = 0
  rect.lo.x[1] = 0
  rect.hi.x[0] = 1
  rect.hi.x[1] = 1
  var ptr = c.legion_accessor_array_2d_raw_rect_ptr(fa, rect, &subrect, offsets)
  return raw_ptr { ptr = [&double](ptr), offset = offsets[1].offset / sizeof(double) }
end


terra dgetrf_terra(n:int,
                   pr:c.legion_physical_region_t,
                   fld: c.legion_field_id_t)
  var rawA = get_raw_ptr(pr, fld)
  c.printf("offset: %d\n", rawA.offset)
  var ipiv:int[2]
  var d:int[1]
  d[0] = n
  var info:int[1]
  lapack.dgetrf_(d, d, rawA.ptr, d, ipiv, info)
  c.printf("info: %d\n", info[0])
end

task dgetrf(n:int, rA : region(ispace(int2d), double))
where reads writes(rA)
do
  c.printf("dgetrf\n")
  dgetrf_terra(n, __physical(rA)[0], __fields(rA)[0])
end


task main()

  var n = 2
  var mat = region(ispace(int2d, {n, n}), double)

  fill(mat, 0)

  mat[{0, 0}] = 4.0
  mat[{0, 1}] = 3.0

  mat[{1, 0}] = 6.0
  mat[{1, 1}] = 3.0

  var coloring = c.legion_domain_coloring_create()
  var bounds = rect2d{{0, 0}, {1, 1}}
  c.legion_domain_coloring_color_domain(coloring, 0, bounds)
  var mat_part = partition(disjoint, mat, coloring)

  dgetrf(2, mat_part[0])

  for i = 0, n do
    for j = 0, n do
      c.printf("%f ", mat[{i, j}])
    end
    c.printf('\n')
  end
end


regentlib.start(main)
