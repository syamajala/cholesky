import "regent"
local c = regentlib.c
terralib.includepath = terralib.includepath .. ";."

terralib.linklibrary("mmio.so")
local mmio = terralib.includec("mmio.h")

-- print(mmio.mm_read_banner.type)

struct MMatBanner {
  M:int
  N:int
  NZ:int
}

terra read_matrix_banner(file:&int8)
  var f = c.fopen(file, 'r')
  var matcode : mmio.MM_typecode[1]
  var ret:int

  ret = mmio.mm_read_banner(f, matcode)

  if ret ~= 0 then
    c.printf("Unable to read banner.\n")
    return MMatBanner{0, 0, 0}
  end

  var M : int[1]
  var N : int[1]
  var nz : int[1]
  ret = mmio.mm_read_mtx_crd_size(f, M, N, nz)

  if ret ~= 0 then
    c.printf("Unable to read matrix size.\n")
    return MMatBanner{0, 0, 0}
  end

  c.fclose(f)
  return MMatBanner{M[0], N[0], nz[0]}
end

-- terra read_matrix()
  -- var I = [&int](c.malloc(sizeof(int) * nz[0]))
  -- var J = [&int](c.malloc(sizeof(int) * nz[0]))
  -- var val = [&double](c.malloc(sizeof(double) * nz[0]))

  -- for i = 0, nz[0] do
  --   c.fscanf(f, "%d %d %lg\n", &(I[i]), &(J[i]), &(val[i]))
  --   I[i] = I[i] - 1
  --   J[i] = J[i] - 1
  -- end

  -- for i = 0, 10 do
  --   c.printf("I[%d]: %d J[%d]: %d val[%d]: %f\n", i, I[i], i, J[i], i, val[i])
  -- end
-- end

terra get_raw_1d_ptr(pr : c.legion_physical_region_t[1],
                     fld : c.legion_field_id_t[1],
                     len: int)
  var fa = c.legion_physical_region_get_field_accessor_array_1d(pr[0], fld[0])
  var rect : c.legion_rect_1d_t
  var subrect : c.legion_rect_1d_t
  var offsets : c.legion_byte_offset_t[2]
  rect.lo.x[0] = 0
  rect.hi.x[1] = len
  var p = c.legion_accessor_array_1d_raw_rect_ptr(fa, rect, &subrect, offsets)
  return [&int](p)
end


fspace MatrixMarket {
  Idx: int2d,
  Sep: int
}

fspace Matrix {
  Val: double
}


task main()

  var file = "lapl_20_2.mtx"
  var banner = read_matrix_banner(file)
  c.printf("M: %d N: %d nz: %d\n", banner.M, banner.N, banner.NZ)

  var mmat = region(ispace(int1d, banner.NZ), MatrixMarket)
  var mat = region(ispace(int2d, {x = banner.M, y = banner.N}), Matrix)

  var p1 = get_raw_1d_ptr(__physical(mmat.Sep), __fields(mmat.Sep), banner.NZ)
  var p2 = get_raw_1d_ptr(__physical(mmat.Idx), __fields(mmat.Idx), banner.NZ)
end



regentlib.start(main)
