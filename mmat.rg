import "regent"
local c = regentlib.c
terralib.includepath = terralib.includepath .. ";."

terralib.linklibrary("mmio.so")
local mmio = terralib.includec("mmio.h")
local mnd = terralib.includec("mnd.h")
local math = terralib.includec("math.h")
--print(mnd.read_separators)

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
  OrigIdx: int2d,
  NewIdx: int2d,
  Sep: int1d
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
  var idx_to_sep = mnd.row_to_separator(separators, banner.M)
  var tree = mnd.build_separator_tree(separators)

  var levels = separators[0][0]
  var num_separators = separators[0][1]

  c.printf("levels: %d\n", levels)
  c.printf("separators: %d\n", num_separators)

  var mmat = region(ispace(int1d, banner.NZ), MatrixMarket)
  var mat = region(ispace(int2d, {x = banner.M-1, y = banner.N-1}), Matrix)

  var entries = read_matrix(matrix_file, banner.NZ)

  for i = 0, banner.NZ do
    var entry = entries[i]
    mmat[i].OrigIdx = {entry.I, entry.J}
    mmat[i].Sep = idx_to_sep[entry.J]
  end

  var coloring = c.legion_domain_coloring_create()
  var prev_size = int2d{x = banner.M-1, y = banner.N-1}

  var separator_bounds : rect2d[32]

  for level = 0, levels do
    for sep_idx = 0, [int](math.pow(2, level)) do
      var sep = tree[level][sep_idx]
      var size = separators[sep][0]-1
      var bounds = rect2d { prev_size - {0, size}, prev_size - {size, 0} }

      separator_bounds[sep] = bounds

      c.printf("level: %d sep: %d size: %d ", level, sep, size)
      c.printf("prev_size: %d %d bounds.lo: %d %d, bounds.hi: %d %d\n",
               prev_size.x, prev_size.y,
               bounds.lo.x, bounds.lo.y,
               bounds.hi.x, bounds.hi.y)

      c.legion_domain_coloring_color_domain(coloring, sep, bounds)
      prev_size = prev_size - {size+1, size+1}

      var par_idx:int = sep_idx
      for par_level = level-1, -1, -1 do
        par_idx = par_idx/2
        var par_sep = tree[par_level][par_idx]
        var par_size = separators[par_sep][0]-1
        var par_bounds = separator_bounds[par_sep]

        var child_bounds = rect2d{ {x = par_bounds.lo.x, y = bounds.lo.y},
          {x = par_bounds.hi.x, y = bounds.hi.y }}

        c.printf("block: %d %d bounds.lo: %d %d bounds.hi: %d %d\n", sep, par_sep,
                 child_bounds.lo.x, child_bounds.lo.y, child_bounds.hi.x, child_bounds.hi.y)
        c.legion_domain_coloring_color_domain(coloring, par_sep, child_bounds)
      end

    end
  end

  -- var mat_part = partition(disjoint, mat, coloring)
  -- var colors = ispace(int1d, num_separators)
  -- var mmat_part = partition(mmat.Sep, colors)
  -- var mat_part = image(mat, mmat_part, mmat.OrigIdx)

  c.fclose(matrix_file)
  c.free(entries)
  c.free(idx_to_sep)
  for i = 0, num_separators+1 do
    c.free(separators[i])
  end
  c.free(separators)
  for i = 0, levels do
    c.free(tree[i])
  end
  c.free(tree)

end

regentlib.start(main)
