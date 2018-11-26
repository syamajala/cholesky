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
  c.printf("M: %d N: %d nz: %d\n", banner.M, banner.N, banner.NZ)

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

  c.printf("done fill: %d %d\n", nz, banner.NZ)

  var interval = 0

  for level = levels-1, -1, -1 do
    for sep_idx = 0, [int](math.pow(2, level)) do
      var sep = tree[level][sep_idx]
      var pivot = mat_part[{sep, sep}]
      var size = pivot.bounds.hi - pivot.bounds.lo + {1, 1}
      c.printf("Level: %d POTRF (%d, %d) Size: %dx%d\n", level, sep, sep, size.x, size.y)
      dpotrf(pivot)

      var par_idx = sep_idx
      for par_level = level-1, -1, -1 do

        par_idx = par_idx/2
        var par_sep = tree[par_level][par_idx]
        var off_diag = mat_part[{sep, par_sep}]
        size = off_diag.bounds.hi - off_diag.bounds.lo + {1, 1}
        c.printf("\tLevel: %d TRSM (%d, %d) Size: %dx%d Lo: %d %d Hi: %d %d\n",
                 par_level, sep, par_sep, size.x, size.y,
                 off_diag.bounds.lo.x, off_diag.bounds.lo.y,
                 off_diag.bounds.hi.x, off_diag.bounds.hi.y)

        -- var cluster_coloring = c.legion_domain_point_coloring_create()
        -- var cluster = clusters[par_sep][interval]
        -- var cluster_size = cluster[0]

        -- -- c.printf("\tCluster: %d Size: %d\n", interval, cluster_size)
        -- var prev_lo = off_diag.bounds.lo
        -- var prev_hi = off_diag.bounds.hi
        -- for dof_idx = 1, cluster_size do
        --   var left = cluster[dof_idx]
        --   var right = cluster[dof_idx+1]

        --   for i = interval-1, -1, -1 do
        --     left = clusters[par_sep][i][left+1]
        --     right = clusters[par_sep][i][right+1]
        --   end

        --   var part_size = int2d{x = right - left - 1, y = 0}
        --   var color:int3d = {x = sep, y = par_sep, z = dof_idx}
        --   var bounds = rect2d { prev_lo, {prev_lo.x, prev_hi.y} + part_size }
        --   c.legion_domain_point_coloring_color_domain(cluster_coloring, color:to_domain_point(),
        --                                               c.legion_domain_from_rect_2d(bounds))

        --   -- c.printf("\tcolor: %d %d %d left: %d right: %d bounds.lo: %d %d, bounds.hi: %d %d vol: %d dim: %d\n",
        --   --          color.x, color.y, color.z,
        --   --          left, right,
        --   --          bounds.lo.x, bounds.lo.y,
        --   --          bounds.hi.x, bounds.hi.y,
        --   --          c.legion_domain_get_volume(c.legion_domain_from_rect_2d(bounds)),
        --   --          c.legion_domain_from_rect_2d(bounds).dim)

        --   prev_lo = bounds.lo + part_size + {1, 0}
        -- end

        -- var cluster_colors = ispace(int3d, {1, 1, cluster_size-1}, {sep, par_sep, 1})
        -- var cluster_part = partition(disjoint, off_diag, cluster_coloring, cluster_colors)
        -- c.legion_domain_point_coloring_destroy(cluster_coloring)

        -- for color in cluster_colors do
        --   var part = cluster_part[color]
        --   size = part.bounds.hi - part.bounds.lo + {1, 1}
        --   c.printf("\t\tInterval: %d TRSM (%d, %d, %d) Size: %dx%d Lo: %d %d Hi: %d %d\n",
        --            interval, color.x, color.y, color.z,
        --            size.x, size.y,
        --            part.bounds.lo.x, part.bounds.lo.y,
        --            part.bounds.hi.x, part.bounds.hi.y)
        --   dtrsm(part, pivot)
        -- end
        dtrsm(pivot, off_diag)
      end

      par_idx = sep_idx
      for par_level = level-1, -1, -1 do
        par_idx = par_idx/2
        var par_sep = tree[par_level][par_idx]

        var grandpar_idx = par_idx
        for grandpar_level = par_level, -1, -1 do
          var grandpar_sep = tree[grandpar_level][grandpar_idx]

          c.printf("\tLevel: %d GEMM (%d, %d)\n", grandpar_level, grandpar_sep, par_sep)

          var grandchild_block = mat_part[{sep, grandpar_sep}]    -- A ex: 28, 16
          var child_block = mat_part[{sep, par_sep}]              -- B ex: 24, 16
          var grandpar_block = mat_part[{par_sep, grandpar_sep}]  -- C ex: 24, 28

          var grandpar_cluster_coloring = c.legion_domain_point_coloring_create()
          var grandchild_cluster_coloring = c.legion_domain_point_coloring_create()
          var cluster = clusters[grandpar_sep][interval]
          var cluster_size = cluster[0]

          var grandpar_prev_lo = grandpar_block.bounds.lo
          var grandpar_prev_hi = grandpar_block.bounds.hi
          var grandchild_prev_lo = grandchild_block.bounds.lo
          var grandchild_prev_hi = grandchild_block.bounds.hi

          for dof_idx = 1, cluster_size do
            var left = cluster[dof_idx]
            var right = cluster[dof_idx+1]

            for i = interval-1, -1, -1 do
              left = clusters[grandpar_sep][i][left+1]
              right = clusters[grandpar_sep][i][right+1]
            end

            var part_size = int2d{x = right - left - 1, y = 0}
            var grandpar_color:int3d = {x = par_sep, y = grandpar_sep, z = dof_idx}
            var grandpar_bounds = rect2d { grandpar_prev_lo, {grandpar_prev_lo.x, grandpar_prev_hi.y} + part_size }
            c.legion_domain_point_coloring_color_domain(grandpar_cluster_coloring, grandpar_color:to_domain_point(),
                                                        c.legion_domain_from_rect_2d(grandpar_bounds))

            -- c.printf("\tcolor: %d %d %d left: %d right: %d bounds.lo: %d %d, bounds.hi: %d %d vol: %d dim: %d\n",
            --          grandpar_color.x, grandpar_color.y, grandpar_color.z,
            --          left, right,
            --          grandpar_bounds.lo.x, grandpar_bounds.lo.y,
            --          grandpar_bounds.hi.x, grandpar_bounds.hi.y,
            --          c.legion_domain_get_volume(c.legion_domain_from_rect_2d(grandpar_bounds)),
            --          c.legion_domain_from_rect_2d(grandpar_bounds).dim)

            grandpar_prev_lo = grandpar_bounds.lo + part_size + {1, 0}

            var grandchild_color:int3d = {x = sep, y = grandpar_sep, z = dof_idx}
            var grandchild_bounds = rect2d { grandchild_prev_lo, {grandchild_prev_lo.x, grandchild_prev_hi.y} + part_size }
            c.legion_domain_point_coloring_color_domain(grandchild_cluster_coloring, grandchild_color:to_domain_point(),
                                                        c.legion_domain_from_rect_2d(grandchild_bounds))

            -- c.printf("\tcolor: %d %d %d left: %d right: %d bounds.lo: %d %d, bounds.hi: %d %d vol: %d dim: %d\n",
            --          grandchild_color.x, grandchild_color.y, grandchild_color.z,
            --          left, right,
            --          grandchild_bounds.lo.x, grandchild_bounds.lo.y,
            --          grandchild_bounds.hi.x, grandchild_bounds.hi.y,
            --          c.legion_domain_get_volume(c.legion_domain_from_rect_2d(grandchild_bounds)),
            --          c.legion_domain_from_rect_2d(grandchild_bounds).dim)

            grandchild_prev_lo = grandchild_bounds.lo + part_size + {1, 0}

          end

          var grandpar_cluster_colors = ispace(int3d, {1, 1, cluster_size-1}, {par_sep, grandpar_sep, 1})
          var grandpar_part = partition(disjoint, grandpar_block, grandpar_cluster_coloring, grandpar_cluster_colors)

          var grandchild_cluster_colors = ispace(int3d, {1, 1, cluster_size-1}, {sep, grandpar_sep, 1})
          var grandchild_part = partition(disjoint, grandchild_block, grandchild_cluster_coloring, grandchild_cluster_colors)

          for i = 1, cluster_size-1 do
            var A = grandchild_part[{sep, grandpar_sep, i}]
            var C = grandpar_part[{par_sep, grandpar_sep, i}]
            dgemm(A, child_block, C)
            -- var grandchild_block = mat_part[{sep, grandpar_sep}]    -- A ex: 28, 16
            -- var child_block = mat_part[{sep, par_sep}]              -- B ex: 24, 16
            -- var grandpar_block = mat_part[{par_sep, grandpar_sep}]  -- C ex: 24, 28
          end

          grandpar_idx = grandpar_idx/2
        end
      end
    end
    interval += 1
  end

  c.printf("matrix entries")
  print_blocks(mat, mat_part)

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
