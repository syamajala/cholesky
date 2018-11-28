import "regent"
local c = regentlib.c
terralib.includepath = terralib.includepath .. ";."

terralib.linklibrary("/home/seshu/dev/cholesky/mmio.so")
local mmio = terralib.includec("mmio.h")
local mnd = terralib.includec("mnd.h")
local math = terralib.includec("math.h")
--print(mnd.read_separators)

local blas = require("blas")

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

task write_matrix(mat: region(ispace(int2d), double),
                  mat_part: partition(disjoint, mat, ispace(int2d)),
                  file:rawstring,
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
  var matrix_file = c.fopen("lapl_3_2.mtx", 'r')
  var banner = read_matrix_banner(matrix_file)
  c.printf("M: %d N: %d nz: %d typecode: %s\n", banner.M, banner.N, banner.NZ, banner.typecode)

  var separator_file = "lapl_3_2_ord_2.txt"
  var separators = mnd.read_separators(separator_file, banner.M)
  var tree = mnd.build_separator_tree(separators)

  var clusters_file = "lapl_3_2_clust_2.txt"
  var clusters = mnd.read_clusters(clusters_file, banner.M)

  var levels = separators[0][0]
  var num_separators = separators[0][1]

  c.printf("levels: %d\n", levels)
  c.printf("separators: %d\n", num_separators)

  var mat = region(ispace(int2d, {x = banner.M, y = banner.N}), double)

  var entries = read_matrix(matrix_file, banner.NZ)

  var coloring = c.legion_domain_point_coloring_create()
  var prev_size = int2d{x = banner.M-1, y = banner.N-1}

  var separator_bounds = region(ispace(int1d, num_separators, 1), rect2d)

  for level = 0, levels do
    for sep_idx = 0, [int](math.pow(2, level)) do
      var sep = tree[level][sep_idx]
      var size = separators[sep][0]
      var bounds = rect2d { prev_size - {size-1, size-1}, prev_size }

      separator_bounds[sep] = bounds

      -- c.printf("level: %d sep: %d size: %d ", level, sep, size)
      -- c.printf("prev_size: %d %d bounds.lo: %d %d, bounds.hi: %d %d vol: %d\n",
      --          prev_size.x, prev_size.y,
      --          bounds.lo.x, bounds.lo.y,
      --          bounds.hi.x, bounds.hi.y,
      --          c.legion_domain_get_volume(c.legion_domain_from_rect_2d(bounds)))

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

        -- c.printf("block: %d %d bounds.lo: %d %d bounds.hi: %d %d\n", sep, par_sep,
        --          child_bounds.lo.x, child_bounds.lo.y, child_bounds.hi.x, child_bounds.hi.y)

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
      var hi = part.bounds.hi

      for i = 0, sep1_size do
        var idxi = sep1[i+1]

        for j = 0, sep2_size do
          var idxj = sep2[j+1]

          if color.x == color.y and i <= j then
            for n = 0, banner.NZ do
              var entry = entries[n]
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
              var entry = entries[n]
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
  write_matrix(mat, mat_part, "permuted_matrix.mtx", banner)

  c.printf("done fill: %d %d\n", nz, banner.NZ)


  var interval = 0

  for level = levels-1, -1, -1 do
    for sep_idx = 0, [int](math.pow(2, level)) do
      var sep = tree[level][sep_idx]
      var pivot = mat_part[{sep, sep}]
      var sizeA = pivot.bounds.hi - pivot.bounds.lo + {1, 1}
      c.printf("Level: %d POTRF (%d, %d)\nSize: %dx%d Lo: %d %d Hi: %d %d\n\n",
               level, sep, sep, sizeA.x, sizeA.y,
               pivot.bounds.lo.x, pivot.bounds.lo.y, pivot.bounds.hi.x, pivot.bounds.hi.y)
      dpotrf(pivot)

      -- we should make an empty partition and accumulate the partitions we make during the TRSM by taking the union
      -- so we can reuse them when we do the GEMM, but right now we just partition twice

      var par_idx = sep_idx
      for par_level = level-1, -1, -1 do

        par_idx = par_idx/2
        var par_sep = tree[par_level][par_idx]
        var off_diag = mat_part[{sep, par_sep}]
        var sizeB = off_diag.bounds.hi - off_diag.bounds.lo + {1, 1}

        c.printf("\tLevel: %d TRSM B=(%d, %d) A=(%d, %d)\n\tSizeA: %dx%d Lo: %d %d Hi: %d %d SizeB: %dx%d Lo: %d %d Hi: %d %d\n\n",
                 par_level, sep, sep, sep, par_sep,
                 sizeA.x, sizeA.y, pivot.bounds.lo.x, pivot.bounds.lo.y, pivot.bounds.hi.x, pivot.bounds.hi.y,
                 sizeB.x, sizeB.y, off_diag.bounds.lo.x, off_diag.bounds.lo.y, off_diag.bounds.hi.x, off_diag.bounds.hi.y)

        var row_cluster = clusters[par_sep][interval]
        var row_cluster_size = row_cluster[0]

        var col_cluster = clusters[sep][interval]
        var col_cluster_size = col_cluster[0]
        var prev_lo = off_diag.bounds.lo

        var off_diag_coloring = c.legion_domain_point_coloring_create()

        c.printf("\t\tPartitioning A=(%d, %d) Cluster: %d Rows: %d Cols: %d\n",
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

          c.printf("\t\tTRSM B=(%d, %d) A=(%d, %d, %d)\n", sep, sep, color.x, color.y, color.z)
          dtrsm(pivot, part)
        end

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

          c.printf("\tLevel: %d GEMM C=(%d, %d) A=(%d, %d) B=(%d, %d)\n\tSizeA: %dx%d Lo: %d %d Hi: %d %d SizeB: %dx%d Lo: %d %d Hi: %d %d SizeC: %dx%d Lo: %d %d Hi: %d %d\n\n",
                   grandpar_level,
                   grandpar_sep, par_sep,
                   grandpar_sep, sep,
                   sep, par_sep,
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
                   grandpar_sep, sep, interval, row_cluster_size-1, col_cluster_size-1)

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

          c.printf("\t\tPartitioning B=(%d, %d) Cluster: %d Rows: %d Cols: %d\n", sep, par_sep, interval, row_cluster_size-1, col_cluster_size-1)

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
                   grandpar_sep, par_sep, interval, row_cluster_size-1, col_cluster_size-1)

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

          if colors_volume == A_colors.volume then
            for Acolor in A_colors do
              var idx = Acolor.z
              var BA = A_part[Acolor]
              var Bcolor = int3d{sep, par_sep, idx}
              var BB = B_part[Bcolor]
              var Ccolor = int3d{par_sep, grandpar_sep, idx*(col_cluster_size-1)+idx}
              var BC = C_part[Ccolor]

              c.printf("\t\tGEMM C=(%d, %d, %d) A=(%d, %d, %d) B=(%d, %d, %d)\n",
                       Ccolor.x, Ccolor.y, Ccolor.z,
                       Acolor.x, Acolor.y, Acolor.z,
                       Bcolor.x, Bcolor.y, Bcolor.z)

              dgemm(BA, BB, BC)
            end

          elseif colors_volume == B_colors.volume then
            for Bcolor in B_colors do
              var idx = Bcolor.z
              var Acolor = int3d{sep, grandpar_sep, idx}
              var BA = A_part[Acolor]
              var BB = B_part[Bcolor]
              var Ccolor = int3d{par_sep, grandpar_sep, idx*(col_cluster_size-1)+idx}
              var BC = C_part[Ccolor]

              c.printf("\t\tGEMM C=(%d, %d, %d) A=(%d, %d, %d) B=(%d, %d, %d)\n",
                       Ccolor.x, Ccolor.y, Ccolor.z,
                       Acolor.x, Acolor.y, Acolor.z,
                       Bcolor.x, Bcolor.y, Bcolor.z)

              dgemm(BA, BB, BC)
            end
          end
          c.printf("\n")

          grandpar_idx = grandpar_idx/2
        end
      end

      c.printf("\n")

    end
    interval += 1
  end

  c.printf("saving factored matrix\n")
  write_matrix(mat, mat_part, "factored_matrix.mtx", banner)

  c.fclose(matrix_file)
  c.free(entries)
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
