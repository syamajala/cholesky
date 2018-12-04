import "regent"
local c = regentlib.c
terralib.includepath = terralib.includepath .. ";."

terralib.linklibrary("/home/seshu/dev/cholesky/mmio.so")
local mmio = terralib.includec("mmio.h")
local mnd = terralib.includec("mnd.h")
local math = terralib.includec("math.h")
--print(mnd.read_separators)

local blas = require("blas")

terralib.linklibrary("/usr/lib/libcblas.so")
local cblas = terralib.includec("cblas.h")

struct MMatBanner {
  M:int
  N:int
  NZ:int
  typecode:mmio.MM_typecode
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

  return MMatBanner{M[0], N[0], nz[0], matcode[0]}
end

struct MatrixEntry {
  I:int
  J:int
  Val:double
}


terra read_matrix(file:&c.FILE, nz:int)
  var entries = [&MatrixEntry](c.malloc(sizeof(MatrixEntry) * nz+1))

  for i = 0, nz do
    var entry = entries[i]
    c.fscanf(file, "%d %d %lg\n", &(entry.I), &(entry.J), &(entry.Val))
    entry.I = entry.I - 1
    entry.J = entry.J - 1
    entries[i] = entry
  end

  return entries
end

terra read_b(file:rawstring, n:int)
  var entries = [&double](c.malloc(n*sizeof(double)))

  var entries_file = c.fopen(file, 'r')

  for i = 0, n+3 do
    if i < 3 then
      var buff:int8[1024]
      c.fgets(buff, 1024, entries_file)
    else
      c.fscanf(entries_file, "%lg\n", &(entries[i-3]))
    end
  end

  c.fclose(entries_file)
  return entries
end

task write_matrix(mat: region(ispace(int2d), double),
                  mat_part: partition(disjoint, mat, ispace(int2d)),
                  file:regentlib.string,
                  banner:MMatBanner)
where
  reads(mat)
do
  var matrix_file = c.fopen(file, 'w')

  var nnz = 0

  for color in mat_part.colors do
    var part = mat_part[color]
    var vol = c.legion_domain_get_volume(c.legion_domain_from_rect_2d(part.bounds))

    if vol ~= 0 then
      var size = part.bounds.hi - part.bounds.lo + {1, 1}
      for i = 0, size.x do
        for j = 0, size.y do
          var idx = part.bounds.lo + {i, j}
          var val = part[idx]
          if val ~= 0 then
            nnz += 1
          end
        end
      end
    end
  end

  mmio.mm_write_banner(matrix_file, banner.typecode)
  mmio.mm_write_mtx_crd_size(matrix_file, banner.M, banner.N, nnz)

  for color in mat_part.colors do
    var part = mat_part[color]
    var vol = c.legion_domain_get_volume(c.legion_domain_from_rect_2d(part.bounds))

    if vol ~= 0 then
      var size = part.bounds.hi - part.bounds.lo + {1, 1}
      for i = 0, size.x do
        for j = 0, size.y do
          var idx = part.bounds.lo + {i, j}
          var val = part[idx]
          idx += {1, 1}
          if val ~= 0 then
            c.fprintf(matrix_file, "%d %d %0.5g\n", idx.x, idx.y, val)
          end
        end
      end
    end
  end

  c.fclose(matrix_file)
end

terra gen_filename(level:int, Ax:int, Ay:int, Bx:int, By:int, Cx:int, Cy:int, operation:rawstring, mm:bool)
  var filename:int8[255]

  if operation == "POTRF" then
    c.sprintf(filename, "steps/potrf_lvl%d_a%d%d", level, Ax, Ay)
  elseif operation == "TRSM" then
    c.sprintf(filename, "steps/trsm_lvl%d_a%d%d_b%d%d", level, Ax, Ay, Bx, By)
  elseif operation == "GEMM" then
    c.sprintf(filename, "steps/gemm_lvl%d_a%d%d_b%d%d_c%d%d", level, Ax, Ay, Bx, By, Cx, Cy)
  end

  var ext:int8[5]

  if mm then
    c.strcpy(ext, ".mtx")
  else
    c.strcpy(ext, ".txt")
  end

  c.strcat(filename, ext)
  var f:rawstring = filename
  c.printf("filename: %s\n", f)
  return f
end


task write_blocks(mat: region(ispace(int2d), double), mat_part: partition(disjoint, mat, ispace(int2d)),
                  level:int, A:int2d, B:int2d, C:int2d, operation:rawstring, banner:MMatBanner)
where
  reads(mat)
do
  var matrix_filename:regentlib.string = gen_filename(level, A.x, A.y, B.x, B.y, C.x, C.y, operation, true)
  write_matrix(mat, mat_part, matrix_filename, banner)

  var block_filename = gen_filename(level, A.x, A.y, B.x, B.y, C.x, C.y, operation, false)
  var file = c.fopen(block_filename, 'w')

  if operation == "POTRF" then
    c.fprintf(file, "Level: %d POTRF A=(%d, %d)\n", level, A.x, A.y)
  elseif operation == "TRSM" then
    c.fprintf(file, "Level: %d TRSM A=(%d, %d) B=(%d, %d)\n",
              level, A.x, A.y, B.x, B.y)
  elseif operation == "GEMM" then
    c.fprintf(file, "Level: %d GEMM A=(%d, %d) B=(%d, %d) C=(%d, %d)\n",
              level, A.x, A.y, B.x, B.y, C.x, C.y)
  end

  for color in mat_part.colors do
    var part = mat_part[color]
    var vol = c.legion_domain_get_volume(c.legion_domain_from_rect_2d(part.bounds))

    if vol ~= 0 then
      var size = part.bounds.hi - part.bounds.lo + {1, 1}
      c.fprintf(file, "Color: %d %d size: %dx%d bounds.lo: %d %d bounds.hi: %d %d vol: %d\n",
                color.x, color.y, size.x, size.y, part.bounds.lo.x, part.bounds.lo.y, part.bounds.hi.x, part.bounds.hi.y, vol)
      for i = 0, size.x do
        for j = 0, size.y do
          var frmt_str:rawstring
          var val = part[part.bounds.lo + {i, j}]
          if val < 0 then
            frmt_str = "%0.2f, "
          else
            frmt_str = " %0.2f, "
          end
          c.fprintf(file, frmt_str, val)
        end
        c.fprintf(file, "\n")
      end
    end
  end
end


task print_blocks(mat: region(ispace(int2d), double), mat_part: partition(disjoint, mat, ispace(int2d)))
where
  reads(mat)
do
  for color in mat_part.colors do
    var part = mat_part[color]
    var vol = c.legion_domain_get_volume(c.legion_domain_from_rect_2d(part.bounds))

    if vol ~= 0 then
      var size = part.bounds.hi - part.bounds.lo + {1, 1}
      c.printf("Color: %d %d size: %dx%d bounds.lo: %d %d bounds.hi: %d %d vol: %d\n",
               color.x, color.y, size.x, size.y, part.bounds.lo.x, part.bounds.lo.y, part.bounds.hi.x, part.bounds.hi.y, vol)
      for i = 0, size.x do
        for j = 0, size.y do
          var frmt_str:rawstring
          var val = part[part.bounds.lo + {i, j}]
          if val < 0 then
            frmt_str = "%0.2f, "
          else
            frmt_str = " %0.2f, "
          end
          c.printf(frmt_str, val)
        end
        c.printf("\n")
      end
    end
  end
end

task main()
  var args = c.legion_runtime_get_input_args()

  var matrix_file_path = ""
  var separator_file = ""
  var clusters_file = ""
  var b_file = ""
  var solution_file = ""
  var factor_file = ""
  var permuted_matrix_file = ""

  for i = 0, args.argc do
    if c.strcmp(args.argv[i], "-i") == 0 then
      matrix_file_path = args.argv[i+1]
    elseif c.strcmp(args.argv[i], "-s") == 0 then
      separator_file = args.argv[i+1]
    elseif c.strcmp(args.argv[i], "-c") == 0 then
      clusters_file = args.argv[i+1]
    elseif c.strcmp(args.argv[i], "-m") == 0 then
      factor_file = args.argv[i+1]
    elseif c.strcmp(args.argv[i], "-p") == 0 then
      permuted_matrix_file = args.argv[i+1]
    elseif c.strcmp(args.argv[i], "-o") == 0 then
      solution_file = args.argv[i+1]
    elseif c.strcmp(args.argv[i], "-b") == 0 then
      b_file = args.argv[i+1]
    end
  end

  var matrix_file = c.fopen(matrix_file_path, 'r')

  var banner = read_matrix_banner(matrix_file)
  c.printf("M: %d N: %d nz: %d typecode: %s\n", banner.M, banner.N, banner.NZ, banner.typecode)

  var separators = mnd.read_separators(separator_file, banner.M)
  var tree = mnd.build_separator_tree(separators)

  var clusters = mnd.read_clusters(clusters_file, banner.M)

  var levels = separators[0][0]
  var num_separators = separators[0][1]

  c.printf("levels: %d\n", levels)
  c.printf("separators: %d\n", num_separators)

  var mat = region(ispace(int2d, {x = banner.M, y = banner.N}), double)

  var matrix_entries = read_matrix(matrix_file, banner.NZ)

  var coloring = c.legion_domain_point_coloring_create()
  var prev_size = int2d{x = banner.M-1, y = banner.N-1}

  var separator_bounds = region(ispace(int1d, num_separators, 1), rect2d)

  for level = 0, levels do
    for sep_idx = 0, [int](math.pow(2, level)) do
      var sep = tree[level][sep_idx]
      var size = separators[sep][0]
      var bounds = rect2d { prev_size - {size-1, size-1}, prev_size }

      separator_bounds[sep] = bounds

      c.printf("level: %d sep: %d size: %d ", level, sep, size)
      c.printf("prev_size: %d %d bounds.lo: %d %d, bounds.hi: %d %d vol: %d\n",
               prev_size.x, prev_size.y,
               bounds.lo.x, bounds.lo.y,
               bounds.hi.x, bounds.hi.y,
               c.legion_domain_get_volume(c.legion_domain_from_rect_2d(bounds)))

      var color:int2d = {x = sep, y = sep}
      c.legion_domain_point_coloring_color_domain(coloring, color:to_domain_point(), c.legion_domain_from_rect_2d(bounds))
      prev_size = prev_size - {size, size}

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
  c.legion_domain_point_coloring_destroy(coloring)

  var nz = 0
  for color in colors do
    var part = mat_part[color]
    var vol = c.legion_domain_get_volume(c.legion_domain_from_rect_2d(part.bounds))

    if vol ~= 0 then
      var sep1 = separators[color.x]
      var sep2 = separators[color.y]
      var sep1_size = sep1[0]
      var sep2_size = sep2[0]

      fill(part, 0)

      var lo = part.bounds.lo

      for i = 0, sep1_size do
        var idxi = sep1[i+1]

        for j = 0, sep2_size do
          var idxj = sep2[j+1]

          if color.x == color.y and i <= j then
            for n = 0, banner.NZ do
              var entry = matrix_entries[n]
              var idx = lo + {j, i}

              if entry.I == idxi and entry.J == idxj then
                part[idx] = entry.Val
                nz += 1
                break

              elseif entry.I == idxj and entry.J == idxi then
                part[idx] = entry.Val
                nz += 1
                break
              end
            end

          elseif color.x ~= color.y then
            for n = 0, banner.NZ do
              var entry = matrix_entries[n]
              var idx = lo + {j, i}

              if entry.I == idxi and entry.J == idxj then
                part[idx] = entry.Val
                nz += 1
                break

              elseif entry.I == idxj and entry.J == idxi then
                part[idx] = entry.Val
                nz += 1
                break
              end
            end
          end
        end
      end
    end
  end

  c.printf("saving permuted matrix\n")
  write_matrix(mat, mat_part, "steps/permuted_matrix.mtx", banner)

  c.printf("done fill: %d %d\n", nz, banner.NZ)

  var interval = 0

  for level = levels-1, -1, -1 do
    for sep_idx = 0, [int](math.pow(2, level)) do
      var sep = tree[level][sep_idx]
      var pivot = mat_part[{sep, sep}]
      var sizeA = pivot.bounds.hi - pivot.bounds.lo + {1, 1}
      c.printf("Level: %d POTRF A=(%d, %d)\nSize: %dx%d Lo: %d %d Hi: %d %d\n\n",
               level, sep, sep, sizeA.x, sizeA.y,
               pivot.bounds.lo.x, pivot.bounds.lo.y, pivot.bounds.hi.x, pivot.bounds.hi.y)
      dpotrf(pivot)

      --write_blocks(mat, mat_part, level, int2d{sep, sep}, int2d{0, 0}, int2d{0, 0}, "POTRF", banner)

      -- we should make an empty partition and accumulate the partitions we make during the TRSM by taking the union
      -- so we can reuse them when we do the GEMM, but right now we just partition twice

      var par_idx = sep_idx
      for par_level = level-1, -1, -1 do

        par_idx = par_idx/2
        var par_sep = tree[par_level][par_idx]
        var off_diag = mat_part[{sep, par_sep}]
        var sizeB = off_diag.bounds.hi - off_diag.bounds.lo + {1, 1}

        c.printf("\tLevel: %d TRSM A=(%d, %d) B=(%d, %d)\n\tSizeA: %dx%d Lo: %d %d Hi: %d %d SizeB: %dx%d Lo: %d %d Hi: %d %d\n\n",
                 par_level, sep, sep, sep, par_sep,
                 sizeA.x, sizeA.y, pivot.bounds.lo.x, pivot.bounds.lo.y, pivot.bounds.hi.x, pivot.bounds.hi.y,
                 sizeB.x, sizeB.y, off_diag.bounds.lo.x, off_diag.bounds.lo.y, off_diag.bounds.hi.x, off_diag.bounds.hi.y)

        var row_cluster = clusters[par_sep][interval]
        var row_cluster_size = row_cluster[0]

        var col_cluster = clusters[sep][interval]
        var col_cluster_size = col_cluster[0]
        var prev_lo = off_diag.bounds.lo

        var off_diag_coloring = c.legion_domain_point_coloring_create()

        c.printf("\t\tPartitioning B=(%d, %d) Cluster: %d Rows: %d Cols: %d\n",
                 sep, par_sep, interval, row_cluster_size-1, col_cluster_size-1)

        for row = 1, row_cluster_size do

          var left = row_cluster[row]
          var right = row_cluster[row+1]

          for i = interval-1, -1, -1 do
            left = clusters[par_sep][i][left+1]
            right = clusters[par_sep][i][right+1]
          end

          for col = 1, col_cluster_size do

            var top = col_cluster[col]
            var bottom = col_cluster[col+1]

            for i = interval-1, -1, -1 do
              top = clusters[sep][i][top+1]
              bottom = clusters[sep][i][bottom+1]
            end

            var part_size = int2d{x = right - left - 1, y = bottom - top - 1}
            var color:int3d = {x = sep, y = par_sep, z = (row-1)*(col_cluster_size-1)+(col-1)}
            var bounds = rect2d { prev_lo, prev_lo + part_size }
            var size = bounds.hi - bounds.lo + {1, 1}

            c.legion_domain_point_coloring_color_domain(off_diag_coloring, color:to_domain_point(),
                                                        c.legion_domain_from_rect_2d(bounds))

            c.printf("\t\tcolor: %d %d %d bounds.lo: %d %d, bounds.hi: %d %d size: %d %d vol: %d\n",
                     color.x, color.y, color.z,
                     bounds.lo.x, bounds.lo.y,
                     bounds.hi.x, bounds.hi.y,
                     size.x, size.y,
                     c.legion_domain_get_volume(c.legion_domain_from_rect_2d(bounds)))

            prev_lo = prev_lo + int2d{0, bottom-top}
          end
          prev_lo = int2d{prev_lo.x + right-left, off_diag.bounds.lo.y}
          c.printf("\n")
        end

        var off_diag_colors = ispace(int3d, {1, 1, (row_cluster_size-1)*(col_cluster_size-1)}, {sep, par_sep, 0})
        var off_diag_part = partition(disjoint, off_diag, off_diag_coloring, off_diag_colors)
        c.legion_domain_point_coloring_destroy(off_diag_coloring)

        for color in off_diag_colors do
          var part = off_diag_part[color]

          c.printf("\t\tTRSM B=(%d, %d, %d) A=(%d, %d)\n", color.x, color.y, color.z, sep, sep)
          dtrsm(pivot, part)
        end

        --write_blocks(mat, mat_part, par_level, int2d{sep, sep}, int2d{sep, par_sep}, int2d{0, 0}, "TRSM", banner)

        c.printf("\n")
      end

      par_idx = sep_idx
      for par_level = level-1, -1, -1 do
        par_idx = par_idx/2
        var par_sep = tree[par_level][par_idx]

        var grandpar_idx = par_idx
        for grandpar_level = par_level, -1, -1 do
          var grandpar_sep = tree[grandpar_level][grandpar_idx]

          var A = mat_part[{sep, grandpar_sep}] -- ex: 16, 28
          var sizeA = A.bounds.hi - A.bounds.lo + {1, 1}
          var B = mat_part[{sep, par_sep}] -- ex: 16, 24
          var sizeB = B.bounds.hi - B.bounds.lo + {1, 1}
          var C = mat_part[{par_sep, grandpar_sep}] -- ex: 24, 28
          var sizeC = C.bounds.hi - C.bounds.lo + {1, 1}

          c.printf("\tLevel: %d GEMM A=(%d, %d) B=(%d, %d) C=(%d, %d)\n\tSizeA: %dx%d Lo: %d %d Hi: %d %d SizeB: %dx%d Lo: %d %d Hi: %d %d SizeC: %dx%d Lo: %d %d Hi: %d %d\n\n",
                   grandpar_level,
                   sep, grandpar_sep,
                   sep, par_sep,
                   par_sep, grandpar_sep,
                   sizeA.x, sizeA.y, A.bounds.lo.x, A.bounds.lo.y, A.bounds.hi.x, A.bounds.hi.y,
                   sizeB.x, sizeB.y, B.bounds.lo.x, B.bounds.lo.y, B.bounds.hi.x, B.bounds.hi.y,
                   sizeC.x, sizeC.y, C.bounds.lo.x, C.bounds.lo.y, C.bounds.hi.x, C.bounds.hi.y)

          -- partition A (should be done in TRSM above) ex: 16, 28
          var row_cluster = clusters[grandpar_sep][interval]
          var row_cluster_size = row_cluster[0]

          var col_cluster = clusters[sep][interval]
          var col_cluster_size = col_cluster[0]
          var prev_lo = A.bounds.lo

          var A_coloring = c.legion_domain_point_coloring_create()

          c.printf("\t\tPartitioning A=(%d, %d) Cluster: %d Rows: %d Cols: %d\n",
                   sep, grandpar_sep, interval, row_cluster_size-1, col_cluster_size-1)

          for row = 1, row_cluster_size do

            var left = row_cluster[row]
            var right = row_cluster[row+1]

            for i = interval-1, -1, -1 do
              left = clusters[grandpar_sep][i][left+1]
              right = clusters[grandpar_sep][i][right+1]
            end

            for col = 1, col_cluster_size do

              var top = col_cluster[col]
              var bottom = col_cluster[col+1]

              for i = interval-1, -1, -1 do
                top = clusters[sep][i][top+1]
                bottom = clusters[sep][i][bottom+1]
              end

              var part_size = int2d{x = right - left - 1, y = bottom - top - 1}
              var color:int3d = {x = sep, y = grandpar_sep, z = (row-1)*(col_cluster_size-1)+(col-1)}
              var bounds = rect2d { prev_lo, prev_lo + part_size }
              var size = bounds.hi - bounds.lo + {1, 1}

              c.legion_domain_point_coloring_color_domain(A_coloring, color:to_domain_point(),
                                                          c.legion_domain_from_rect_2d(bounds))

              c.printf("\t\tcolor: %d %d %d bounds.lo: %d %d, bounds.hi: %d %d size: %d %d vol: %d\n",
                       color.x, color.y, color.z,
                       bounds.lo.x, bounds.lo.y,
                       bounds.hi.x, bounds.hi.y,
                       size.x, size.y,
                       c.legion_domain_get_volume(c.legion_domain_from_rect_2d(bounds)))
              prev_lo = prev_lo + int2d{0, bottom-top}
            end
            prev_lo = int2d{prev_lo.x + right-left, A.bounds.lo.y}
            c.printf("\n")
          end

          var A_colors = ispace(int3d, {1, 1, (row_cluster_size-1)*(col_cluster_size-1)}, {sep, grandpar_sep, 0})
          var A_part = partition(disjoint, A, A_coloring, A_colors)
          c.legion_domain_point_coloring_destroy(A_coloring)

          -- partition B (should be done in TRSM above) ex: 16, 24
          row_cluster = clusters[par_sep][interval]
          row_cluster_size = row_cluster[0]

          col_cluster = clusters[sep][interval]
          col_cluster_size = col_cluster[0]
          prev_lo = B.bounds.lo

          var B_coloring = c.legion_domain_point_coloring_create()

          c.printf("\t\tPartitioning B=(%d, %d) Cluster: %d Rows: %d Cols: %d\n",
                   sep, par_sep, interval, row_cluster_size-1, col_cluster_size-1)

          for row = 1, row_cluster_size do

            var left = row_cluster[row]
            var right = row_cluster[row+1]

            for i = interval-1, -1, -1 do
              left = clusters[par_sep][i][left+1]
              right = clusters[par_sep][i][right+1]
            end

            for col = 1, col_cluster_size do

              var top = col_cluster[col]
              var bottom = col_cluster[col+1]

              for i = interval-1, -1, -1 do
                top = clusters[sep][i][top+1]
                bottom = clusters[sep][i][bottom+1]
              end

              var part_size = int2d{x = right - left - 1, y = bottom - top - 1}
              var color:int3d = {x = sep, y = par_sep, z = (row-1)*(col_cluster_size-1)+(col-1)}
              var bounds = rect2d { prev_lo, prev_lo + part_size }
              var size = bounds.hi - bounds.lo + {1, 1}
              c.legion_domain_point_coloring_color_domain(B_coloring, color:to_domain_point(),
                                                          c.legion_domain_from_rect_2d(bounds))

              c.printf("\t\tcolor: %d %d %d bounds.lo: %d %d, bounds.hi: %d %d size: %d %d vol: %d\n",
                       color.x, color.y, color.z,
                       bounds.lo.x, bounds.lo.y,
                       bounds.hi.x, bounds.hi.y,
                       size.x, size.y,
                       c.legion_domain_get_volume(c.legion_domain_from_rect_2d(bounds)))
              prev_lo = prev_lo + int2d{0, bottom-top}
            end
            prev_lo = int2d{prev_lo.x + right-left, B.bounds.lo.y}
            c.printf("\n")
          end

          var B_colors = ispace(int3d, {1, 1, (row_cluster_size-1)*(col_cluster_size-1)}, {sep, par_sep, 0})
          var B_part = partition(disjoint, B, B_coloring, B_colors)
          c.legion_domain_point_coloring_destroy(B_coloring)

          -- partition C  ex: 24, 28
          row_cluster = clusters[grandpar_sep][interval]
          row_cluster_size = row_cluster[0]

          col_cluster = clusters[par_sep][interval]
          col_cluster_size = col_cluster[0]
          prev_lo = C.bounds.lo

          var C_coloring = c.legion_domain_point_coloring_create()

          c.printf("\t\tPartitioning C=(%d, %d) Cluster: %d Rows: %d Cols: %d\n",
                   par_sep, grandpar_sep, interval, row_cluster_size-1, col_cluster_size-1)

          for row = 1, row_cluster_size do

            var left = row_cluster[row]
            var right = row_cluster[row+1]

            for i = interval-1, -1, -1 do
              left = clusters[grandpar_sep][i][left+1]
              right = clusters[grandpar_sep][i][right+1]
            end

            for col = 1, col_cluster_size do

              var top = col_cluster[col]
              var bottom = col_cluster[col+1]

              for i = interval-1, -1, -1 do
                top = clusters[par_sep][i][top+1]
                bottom = clusters[par_sep][i][bottom+1]
              end

              var part_size = int2d{x = right - left - 1, y = bottom - top - 1}
              var color:int3d = {x = par_sep, y = grandpar_sep, z = (row-1)*(col_cluster_size-1)+(col-1)}
              var bounds = rect2d { prev_lo, prev_lo + part_size }
              var size = bounds.hi - bounds.lo + {1, 1}
              c.legion_domain_point_coloring_color_domain(C_coloring, color:to_domain_point(),
                                                          c.legion_domain_from_rect_2d(bounds))

              c.printf("\t\tcolor: %d %d %d bounds.lo: %d %d, bounds.hi: %d %d size: %d %d vol: %d\n",
                       color.x, color.y, color.z,
                       bounds.lo.x, bounds.lo.y,
                       bounds.hi.x, bounds.hi.y,
                       size.x, size.y,
                       c.legion_domain_get_volume(c.legion_domain_from_rect_2d(bounds)))
              prev_lo = prev_lo + int2d{0, bottom-top}
            end
            prev_lo = int2d{prev_lo.x + right-left, C.bounds.lo.y}
            c.printf("\n")
          end

          var C_colors = ispace(int3d, {1, 1, (row_cluster_size-1)*(col_cluster_size-1)}, {par_sep, grandpar_sep, 0})
          var C_part = partition(disjoint, C, C_coloring, C_colors)
          c.legion_domain_point_coloring_destroy(C_coloring)

          c.printf("\t\tA Vol: %d B Vol: %d C Vol: %d\n\n", A_colors.volume, B_colors.volume, C_colors.volume)

          var colors_volume = min(A_colors.volume, min(B_colors.volume, C_colors.volume))

          if grandpar_sep == par_sep then
            for Acolor in A_colors do
              var row = Acolor.z
              var ABlock = A_part[Acolor]

              for Bcolor in A_colors do
                var col = Bcolor.z

                var Ccolor = int3d{par_sep, grandpar_sep, row*(col_cluster_size-1)+col}
                var CBlock = C_part[Ccolor]

                if col < row then
                  var BBlock = A_part[Bcolor]

                  c.printf("\t\tGEMM C=(%d, %d, %d) A=(%d, %d, %d) B=(%d, %d, %d)\n",
                           Ccolor.x, Ccolor.y, Ccolor.z,
                           Acolor.x, Acolor.y, Acolor.z,
                           Bcolor.x, Bcolor.y, Bcolor.z)

                  dgemm(ABlock, BBlock, CBlock)

                elseif col == row then

                  c.printf("\t\tSYRK C=(%d, %d, %d) A=(%d, %d, %d)\n",
                           Ccolor.x, Ccolor.y, Ccolor.z,
                           Acolor.x, Acolor.y, Acolor.z)
                  dsyrk(ABlock, CBlock)

                end
              end
            end
          else
            for Acolor in A_colors do
              var ABlock = A_part[Acolor]
              var row = Acolor.z

              for Bcolor in B_colors do
                var BBlock = B_part[Bcolor]
                var col = Bcolor.z

                var Ccolor = int3d{par_sep, grandpar_sep, row*(col_cluster_size-1)+col}
                var CBlock = C_part[Ccolor]

                c.printf("\t\tGEMM C=(%d, %d, %d) A=(%d, %d, %d) B=(%d, %d, %d)\n",
                         Ccolor.x, Ccolor.y, Ccolor.z,
                         Acolor.x, Acolor.y, Acolor.z,
                         Bcolor.x, Bcolor.y, Bcolor.z)

                dgemm(ABlock, BBlock, CBlock)
              end
            end
          end

          -- write_blocks(mat, mat_part, grandpar_level,
          --              int2d{sep, grandpar_sep}, int2d{sep, par_sep}, int2d{par_sep, grandpar_sep}, "GEMM", banner)

          c.printf("\n")

          grandpar_idx = grandpar_idx/2
        end
      end

      c.printf("\n")

    end
    interval += 1
  end

  if c.strcmp(factor_file, '') ~= 0 then
    c.printf("saving factored matrix to: %s\n\n", factor_file)
    write_matrix(mat, mat_part, factor_file, banner)
  end

  if c.strcmp(b_file, '') == 0 then
    c.exit(0)
  end

  var Bentries = read_b(b_file, banner.N)
  var B = region(ispace(int1d, banner.N), double)
  var Bcoloring = c.legion_domain_point_coloring_create()
  var Bprev_size = 0
  var X = region(ispace(int1d, banner.N), double)

  for sep = 1, num_separators+1 do
    var size = separators[sep][0]
    var bounds = rect1d { Bprev_size, Bprev_size + size - 1}
    c.printf("Separator: %d Lo: %d Hi: %d Size: %d 1\n", sep, bounds.lo, bounds.hi, bounds.hi - bounds.lo + 1)
    var color:int1d = int1d{ sep }
    c.legion_domain_point_coloring_color_domain(Bcoloring, color:to_domain_point(), c.legion_domain_from_rect_1d(bounds))
    Bprev_size = Bprev_size + size
  end

  c.printf("\n")

  var Bcolors = ispace(int1d, num_separators, 1)
  var Bpart = partition(disjoint, B, Bcoloring, Bcolors)
  c.legion_domain_point_coloring_destroy(Bcoloring)

  for Bcolor in Bcolors do
    var part = Bpart[Bcolor]

    var lo = part.bounds
    var ib:int = [int](Bcolor)
    var sep = separators[ib]
    var sep_size = sep[0]

    for i = 0, sep_size do
      var idxi = sep[i+1]
      part[part.bounds.lo + i] = Bentries[idxi]
    end
  end

  c.printf("Forward Substitution\n")
  for level = levels-1, -1, -1 do
    for sep_idx = [int](math.pow(2, level))-1, -1, -1 do
      var sep = tree[level][sep_idx]
      var pivot = mat_part[{sep, sep}]
      var bp = Bpart[sep]

      var sizeA = pivot.bounds.hi - pivot.bounds.lo + {1, 1}
      var sizeB = bp.bounds.hi - bp.bounds.lo + 1

      c.printf("Level: %d TRSV A=(%d, %d) B=(%d)\nSizeA: %dx%d Lo: %d %d Hi: %d %d SizeB: %d Lo: %d Hi: %d\n\n",
               level, sep, sep, sep,
               sizeA.x, sizeA.y, pivot.bounds.lo.x, pivot.bounds.lo.y, pivot.bounds.hi.x, pivot.bounds.hi.y,
               sizeB, bp.bounds.lo, bp.bounds.hi)

      dtrsv(pivot, bp, cblas.CblasLower, cblas.CblasNoTrans)

      var par_idx = sep_idx
      for par_level = level-1, -1, -1 do
        par_idx = par_idx/2
        var par_sep = tree[par_level][par_idx]

        var A = mat_part[{sep, par_sep}]
        var sizeA = A.bounds.hi - A.bounds.lo + {1, 1}
        var X = Bpart[sep]
        var sizeX = X.bounds.hi - X.bounds.lo + 1
        var Y = Bpart[par_sep]
        var sizeY = Y.bounds.hi - Y.bounds.lo + 1

        c.printf("\tLevel: %d GEMV A=(%d, %d) X=(%d) Y=(%d)\n\tSizeA: %dx%d Lo: %d %d Hi: %d %d SizeX: %d Lo: %d Hi: %d SizeY: %d Lo: %d Hi: %d\n\n",
                 par_level, sep, par_sep, sep, par_sep,
                 sizeA.x, sizeA.y, A.bounds.lo.x, A.bounds.lo.y, A.bounds.hi.x, A.bounds.hi.y,
                 sizeX, X.bounds.lo, X.bounds.hi,
                 sizeY, Y.bounds.lo, Y.bounds.hi)

        dgemv(A, X, Y, cblas.CblasNoTrans)
      end
    end
  end

  c.printf("Backward Substitution\n")
  for par_level = 0, levels do
    for par_idx = 0, [int](math.pow(2, par_level)) do
      var par_sep = tree[par_level][par_idx]
      var pivot = mat_part[{par_sep, par_sep}]
      var bp = Bpart[par_sep]

      var sizeA = pivot.bounds.hi - pivot.bounds.lo + {1, 1}
      var sizeB = bp.bounds.hi - bp.bounds.lo + {1, 1}

      c.printf("Level: %d TRSV A=(%d, %d) B=(%d)\nSizeA: %dx%d Lo: %d %d Hi: %d %d SizeB: %d Lo: %d Hi: %d\n\n",
               par_level, par_sep, par_sep, par_sep,
               sizeA.x, sizeA.y, pivot.bounds.lo.x, pivot.bounds.lo.y, pivot.bounds.hi.x, pivot.bounds.hi.y,
               sizeB, bp.bounds.lo, bp.bounds.hi)

      dtrsv(pivot, bp, cblas.CblasLower, cblas.CblasTrans)

      for level = par_level+1, levels do
        for sep_idx = [int](math.pow(2, level))-1, -1, -1 do
          var sep = tree[level][sep_idx]

          var A = mat_part[{sep, par_sep}]
          var sizeA = A.bounds.hi - A.bounds.lo + {1, 1}
          var X = Bpart[par_sep]
          var sizeX = X.bounds.hi - X.bounds.lo + 1
          var Y = Bpart[sep]
          var sizeY = Y.bounds.hi - Y.bounds.lo + 1

          var vol = c.legion_domain_get_volume(c.legion_domain_from_rect_2d(A.bounds))
          if vol ~= 0 then
            c.printf("\tLevel: %d GEMV A=(%d, %d) X=(%d) Y=(%d)\n\tSizeA: %dx%d Lo: %d %d Hi: %d %d SizeX: %d Lo: %d Hi: %d SizeY: %d Lo: %d Hi: %d\n\n",
                     level, sep, par_sep, par_sep, sep,
                     sizeA.x, sizeA.y, A.bounds.lo.x, A.bounds.lo.y, A.bounds.hi.x, A.bounds.hi.y,
                     sizeX, X.bounds.lo, X.bounds.hi,
                     sizeY, Y.bounds.lo, Y.bounds.hi)

            dgemv(A, X, Y, cblas.CblasTrans)
          end
        end
      end
    end
  end

  var j = 0
  for sep = 1, num_separators+1 do
    var size = separators[sep][0]
    for i = 1, size+1 do
      var idxi = separators[sep][i]
      X[idxi] = B[j]
      j += 1
    end
  end

  if c.strcmp(solution_file, '') ~= 0 then
    c.printf("Saving solution to: %s\n", solution_file)
    var solution = c.fopen(solution_file, 'w')

    for i = 0, banner.N do
      c.fprintf(solution, "%0.5g\n", X[i])
    end

    c.fclose(solution)
  end

  c.fclose(matrix_file)
  c.free(matrix_entries)

  for i = 0, num_separators+1 do
    c.free(separators[i])
  end
  c.free(separators)

  for i = 0, levels do
    c.free(tree[i])
  end
  c.free(tree)

  c.free(Bentries)
end

regentlib.start(main)
