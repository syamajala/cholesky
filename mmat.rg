import "regent"
local c = regentlib.c
terralib.includepath = terralib.includepath .. ";."

terralib.linklibrary("mmio.so")
local mmio = terralib.includec("mmio.h")
local mnd = terralib.includec("mnd.h")

-- print(mmio.mm_read_banner.type)

struct MMatBanner {
  M:int
  N:int
  NZ:int
}

terra read_matrix_banner(file:&c.FILE)
  var matcode : mmio.MM_typecode[1]
  var ret:int

  ret = mmio.mm_read_banner(file, matcode)

  if ret ~= 0 then
    c.printf("Unable to read banner.\n")
    return MMatBanner{0, 0, 0}
  end

  var M : int[1]
  var N : int[1]
  var nz : int[1]
  ret = mmio.mm_read_mtx_crd_size(file, M, N, nz)

  if ret ~= 0 then
    c.printf("Unable to read matrix size.\n")
    return MMatBanner{0, 0, 0}
  end

  return MMatBanner{M[0], N[0], nz[0]}
end

struct MatrixEntry {
  I:int
  J:int
  Val:double
}

terra read_matrix(file:&c.FILE, nz:int)
  var entries = [&MatrixEntry](c.malloc(sizeof(MatrixEntry) * nz))

  for i = 0, nz do
    var entry = entries[i]
    c.fscanf(file, "%d %d %lg\n", &(entry.I), &(entry.J), &(entry.Val))
    entry.I = entry.I - 1
    entry.J = entry.J - 1
    entries[i] = entry
  end

  return entries
end

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
  var matrix_file = c.fopen("lapl_20_2.mtx", 'r')
  var banner = read_matrix_banner(matrix_file)
  c.printf("M: %d N: %d nz: %d\n", banner.M, banner.N, banner.NZ)

  var separator_file = "lapl_20_2_ord_5.txt"
  var separators = mnd.read_separators(separator_file, banner.M)
  var idx_to_sep =  mnd.row_to_separator(separators, banner.M)

  c.printf("levels: %d\n", separators[0][0])
  c.printf("separators: %d\n", separators[0][1])

  var mmat = region(ispace(int1d, banner.NZ), MatrixMarket)
  var mat = region(ispace(int2d, {x = banner.M, y = banner.N}), Matrix)

  var entries = read_matrix(matrix_file, banner.NZ)

  for i = 0, banner.NZ do
    var entry = entries[i]
    c.printf("I[%d]: %d J[%d]: %d val[%d]: %f\n", i, entry.I, i, entry.J, i, entry.Val)
  end

  c.fclose(matrix_file)

end

regentlib.start(main)
