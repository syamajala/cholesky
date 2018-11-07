import "regent"
local c = regentlib.c
terralib.includepath = terralib.includepath .. ";."

terralib.linklibrary("/home/seshu/dev/cholesky/mmio.so")
local mmio = terralib.includec("mmio.h")
local mnd = terralib.includec("mnd.h")
local math = terralib.includec("math.h")
--print(mnd.read_separators)

local blas = require("blas")
local struct __f2d { y : int, x : int }
local f2d = regentlib.index_type(__f2d, "f2d")

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

fspace MatrixMarket {
  OrigIdx: int2d,
  NewIdx: int2d,
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
  var mat = region(ispace(f2d, {x = banner.M, y = banner.N}), double)

  var entries = read_matrix(matrix_file, banner.NZ)

  for i = 0, banner.NZ do
    var entry = entries[i]
    mmat[i].OrigIdx = {entry.I, entry.J}
  end

  var coloring = c.legion_domain_point_coloring_create()
  var prev_size = int2d{x = banner.M-1, y = banner.N-1}

  -- TODO FIX THIS
  var separator_bounds : rect2d[32]

  for level = 0, levels do
    for sep_idx = 0, [int](math.pow(2, level)) do
      var sep = tree[level][sep_idx]
      var size = separators[sep][0]-1
      var bounds = rect2d { prev_size - {size, size}, prev_size }

      separator_bounds[sep] = bounds

      c.printf("level: %d sep: %d size: %d ", level, sep, size+1)
      c.printf("prev_size: %d %d bounds.lo: %d %d, bounds.hi: %d %d vol: %d\n",
               prev_size.x, prev_size.y,
               bounds.lo.x, bounds.lo.y,
               bounds.hi.x, bounds.hi.y,
               c.legion_domain_get_volume(c.legion_domain_from_rect_2d(bounds)))

      var color:int2d = {x = sep, y = sep}
      c.legion_domain_point_coloring_color_domain(coloring, color:to_domain_point(), c.legion_domain_from_rect_2d(bounds))
      prev_size = prev_size - {size+1, size+1}

      var par_idx:int = sep_idx
      for par_level = level-1, -1, -1 do
        par_idx = par_idx/2
        var par_sep = tree[par_level][par_idx]
        var par_size = separators[par_sep][0]-1
        var par_bounds = separator_bounds[par_sep]

        var child_bounds = rect2d{ {x = par_bounds.lo.x, y = bounds.lo.y},
                                   {x = par_bounds.hi.x, y = bounds.hi.y } }

        c.printf("block: %d %d bounds.lo: %d %d bounds.hi: %d %d\n", sep, par_sep,
                 child_bounds.lo.x, child_bounds.lo.y, child_bounds.hi.x, child_bounds.hi.y)

        var color:int2d = {sep, par_sep}
        c.legion_domain_point_coloring_color_domain(coloring, color:to_domain_point(), c.legion_domain_from_rect_2d(child_bounds))
      end

    end
  end

  var colors = ispace(int2d, {num_separators, num_separators}, {1, 1})
  var mat_part = partition(disjoint, mat, coloring, colors)
  c.printf('\n')

  for color in colors do
    var part = mat_part[color]
    var vol = c.legion_domain_get_volume(c.legion_domain_from_rect_2d(part.bounds))
    if vol ~= 0 then
      -- c.printf("color: %d %d vol: %d ", color.x, color.y, vol)
      -- c.printf("bounds.lo: %d %d bounds.hi: %d %d\n", part.bounds.lo.x, part.bounds.lo.y, part.bounds.hi.x, part.bounds.hi.y)

      fill(part, 0)
    end
  end

  c.printf("done fill")

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
