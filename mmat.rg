-- Copyright 2018 Stanford University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

import "regent"
local c = regentlib.c
terralib.includepath = terralib.includepath .. ";."

terralib.linklibrary("mmio.so")
local mmio = terralib.includec("mmio.h")
local mnd = terralib.includec("mnd.h")
local math = terralib.includec("math.h")
-- print(c.fopen)

local blas = require("blas")

terralib.linklibrary("libcblas.so")
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

terra gen_filename(level:int, Ax:int, Ay:int, Bx:int, By:int, Cx:int, Cy:int, operation:rawstring, mm:bool, output_dir:rawstring)
  var filename:int8[1024]

  if operation == "POTRF" then
    c.sprintf(filename, "%s/potrf_lvl%d_a%d%d", output_dir, level, Ax, Ay)
  elseif operation == "TRSM" then
    c.sprintf(filename, "%s/trsm_lvl%d_a%d%d_b%d%d", output_dir, level, Ax, Ay, Bx, By)
  elseif operation == "GEMM" then
    c.sprintf(filename, "%s/gemm_lvl%d_a%d%d_b%d%d_c%d%d", output_dir, level, Ax, Ay, Bx, By, Cx, Cy)
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
                  level:int, A:int2d, B:int2d, C:int2d, operation:rawstring, banner:MMatBanner, output_dir:rawstring)
where
  reads(mat)
do
  var matrix_filename:regentlib.string = gen_filename(level, A.x, A.y, B.x, B.y, C.x, C.y, operation, true, output_dir)
  write_matrix(mat, mat_part, matrix_filename, banner)

  var block_filename = gen_filename(level, A.x, A.y, B.x, B.y, C.x, C.y, operation, false, output_dir)
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


__demand(__inline)
task partition_separator(row_sep:int, col_sep:int, interval:int, clusters:&&&int,
                         block:region(ispace(int2d), double), debug:bool)

  var row_cluster = clusters[row_sep]
  var row_cluster_size = row_cluster[interval][0]

  var col_cluster = clusters[col_sep]
  var col_cluster_size = col_cluster[interval][0]
  var prev_lo = block.bounds.lo

  var block_coloring = c.legion_domain_point_coloring_create()

  if debug then
    c.printf("\t\tPartitioning (%d, %d) Cluster: %d Rows: %d Cols: %d\n",
             col_sep, row_sep, interval, row_cluster_size-1, col_cluster_size-1)
  end

  for row = 1, row_cluster_size do

    var left = row_cluster[interval][row]
    var right = row_cluster[interval][row+1]

    for i = interval-1, -1, -1 do
      left = row_cluster[i][left+1]
      right = row_cluster[i][right+1]
    end

    for col = 1, col_cluster_size do

      var top = col_cluster[interval][col]
      var bottom = col_cluster[interval][col+1]

      for i = interval-1, -1, -1 do
        top = col_cluster[i][top+1]
        bottom = col_cluster[i][bottom+1]
      end

      var part_size = int2d{x = right - left - 1, y = bottom - top - 1}
      var color:int3d = {x = row_sep, y = col_sep, z = (row-1)*(col_cluster_size-1)+(col-1)}
      var bounds = rect2d { prev_lo, prev_lo + part_size }
      var size = bounds.hi - bounds.lo + {1, 1}

      c.legion_domain_point_coloring_color_domain(block_coloring, color:to_domain_point(),
                                                  c.legion_domain_from_rect_2d(bounds))

      if debug then
        c.printf("\t\tcolor: %d %d %d bounds.lo: %d %d, bounds.hi: %d %d size: %d %d vol: %d\n",
                 color.x, color.y, color.z,
                 bounds.lo.x, bounds.lo.y,
                 bounds.hi.x, bounds.hi.y,
                 size.x, size.y,
                 c.legion_domain_get_volume(c.legion_domain_from_rect_2d(bounds)))
      end

      prev_lo = prev_lo + int2d{0, bottom-top}
    end
    prev_lo = int2d{prev_lo.x + right-left, block.bounds.lo.y}

    if debug then
      c.printf("\n")
    end
  end

  var colors = ispace(int3d, {1, 1, (row_cluster_size-1)*(col_cluster_size-1)}, {row_sep, col_sep, 0})
  var part = partition(disjoint, block, block_coloring, colors)
  c.legion_domain_point_coloring_destroy(block_coloring)
  return part
end

__demand(__inline)
task fill_block(block:region(ispace(int2d), double), color:int3d, separators:&&int, NZ:int, matrix_entries:&MatrixEntry,
               filled_blocks:region(ispace(int3d), bool))
where
  reads writes(block, filled_blocks)
do
  var sep1 = separators[color.y]
  var sep2 = separators[color.x]
  var sep1_size = sep1[0]
  var sep2_size = sep2[0]
  var nz = 0

  fill(block, 0)

  var lo = block.bounds.lo
  var filled = false

  for i = 0, sep1_size do
    var idxi = sep1[i+1]

    for j = 0, sep2_size do
      var idxj = sep2[j+1]

      if color.x == color.y and i <= j then
        for n = 0, NZ do
          var entry = matrix_entries[n]
          var idx = lo + {j, i}

          if entry.I == idxi and entry.J == idxj then
            block[idx] = entry.Val
            filled = true
            nz += 1
            break
          elseif entry.I == idxj and entry.J == idxi then
            block[idx] = entry.Val
            filled = true
            nz += 1
            break
          end
        end
      elseif color.x ~= color.y then
        for n = 0, NZ do
          var entry = matrix_entries[n]
          var idx = lo + {j, i}

          if entry.I == idxi and entry.J == idxj then
            block[idx] = entry.Val
            filled = true
            nz += 1
            break
          elseif entry.I == idxj and entry.J == idxi then
            block[idx] = entry.Val
            filled = true
            nz += 1
            break
          end
        end
      end
    end
  end

  filled_blocks[color] = filled
  if filled then
    c.printf("Filled: %d %d %d with %d non-zeros\n", color.y, color.x, color.z, nz)
  else
    c.printf("Block %d %d %d empty\n", color.y, color.x, color.z)
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
  var debug_path = ""
  var debug = false

  for i = 0, args.argc do
    if c.strcmp(args.argv[i], "-i") == 0 then
      matrix_file_path = args.argv[i+1]
    elseif c.strcmp(args.argv[i], "-s") == 0 then
      separator_file = args.argv[i+1]
    elseif c.strcmp(args.argv[i], "-c") == 0 then
      clusters_file = args.argv[i+1]
    elseif c.strcmp(args.argv[i], "-m") == 0 then
      factor_file = args.argv[i+1]
    elseif c.strcmp(args.argv[i], "-o") == 0 then
      solution_file = args.argv[i+1]
    elseif c.strcmp(args.argv[i], "-b") == 0 then
      b_file = args.argv[i+1]
    elseif c.strcmp(args.argv[i], "-d") == 0 then
      debug_path = args.argv[i+1]
      debug = true
    end
  end

  var matrix_file = c.fopen(matrix_file_path, 'r')

  var banner = read_matrix_banner(matrix_file)
  c.printf("M: %d N: %d nz: %d typecode: %s\n", banner.M, banner.N, banner.NZ, banner.typecode)

  var separators = mnd.read_separators(separator_file, banner.M)
  var levels = separators[0][0]
  var num_separators = separators[0][1]

  var tree = mnd.build_separator_tree(separators)

  var clusters = mnd.read_clusters(clusters_file, num_separators+1, 10, 50)

  c.printf("levels: %d\n", levels)
  c.printf("separators: %d\n", num_separators)

  var mat = region(ispace(int2d, {x = banner.M, y = banner.N}), double)

  var matrix_entries = read_matrix(matrix_file, banner.NZ)
  c.fclose(matrix_file)

  var coloring = c.legion_domain_point_coloring_create()
  var prev_size = int2d{x = banner.M-1, y = banner.N-1}

  var separator_bounds = region(ispace(int1d, num_separators, 1), rect2d)

  for level = 0, levels do
    for sep_idx = 0, [int](math.pow(2, level)) do
      var sep = tree[level][sep_idx]
      var size = separators[sep][0]
      var bounds = rect2d { prev_size - {size-1, size-1}, prev_size }

      separator_bounds[sep] = bounds

      -- if debug then
      --   c.printf("level: %d sep: %d size: %d ", level, sep, size)
      --   c.printf("prev_size: %d %d bounds.lo: %d %d, bounds.hi: %d %d vol: %d\n",
      --            prev_size.x, prev_size.y,
      --            bounds.lo.x, bounds.lo.y,
      --            bounds.hi.x, bounds.hi.y,
      --            c.legion_domain_get_volume(c.legion_domain_from_rect_2d(bounds)))
      -- end

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

        -- if debug then
        --   c.printf("block: %d %d bounds.lo: %d %d bounds.hi: %d %d\n", sep, par_sep,
        --            child_bounds.lo.x, child_bounds.lo.y, child_bounds.hi.x, child_bounds.hi.y)
        -- end

        var color:int2d = {sep, par_sep}
        c.legion_domain_point_coloring_color_domain(coloring, color:to_domain_point(), c.legion_domain_from_rect_2d(child_bounds))

      end
    end
  end

  var colors = ispace(int2d, {num_separators, num_separators}, {1, 1})
  var mat_part = partition(disjoint, mat, coloring, colors)
  c.legion_domain_point_coloring_destroy(coloring)

  var interval = 0
  var max_int_size = clusters[num_separators][0][0]
  var filled_blocks = region(ispace(int3d, {num_separators, num_separators, max_int_size}, {1, 1, 0}), bool)
  fill(filled_blocks, false)

  for level = levels-1, -1, -1 do
    for sep_idx = 0, [int](math.pow(2, level)) do
      var sep = tree[level][sep_idx]
      var pivot = mat_part[{sep, sep}]
      var pivot_part = partition_separator(sep, sep, interval, clusters, pivot, debug)

      if interval == 0 then
        for color in pivot_part.colors do
          fill_block(pivot_part[color], color, separators, banner.NZ, matrix_entries, filled_blocks)
        end
      end

      dpotrf(pivot)

      if debug then
        var sizeA = pivot.bounds.hi - pivot.bounds.lo + {1, 1}

        c.printf("Level: %d POTRF A=(%d, %d)\nSize: %dx%d Lo: %d %d Hi: %d %d\n\n",
                 level, sep, sep, sizeA.x, sizeA.y,
                 pivot.bounds.lo.x, pivot.bounds.lo.y, pivot.bounds.hi.x, pivot.bounds.hi.y)

        write_blocks(mat, mat_part, level, int2d{sep, sep}, int2d{0, 0}, int2d{0, 0}, "POTRF", banner, debug_path)
      end

      -- we should make an empty partition and accumulate the partitions we make during the TRSM by taking the union
      -- so we can reuse them when we do the GEMM, but right now we just partition twice

      var par_idx = sep_idx
      for par_level = level-1, -1, -1 do

        par_idx = par_idx/2
        var par_sep = tree[par_level][par_idx]
        var off_diag_color = int2d{sep, par_sep}
        var off_diag = mat_part[off_diag_color]

        var off_diag_part = partition_separator(par_sep, sep, interval, clusters, off_diag, debug)

        if interval == 0 then
          for color in off_diag_part.colors do
            fill_block(off_diag_part[color], color, separators, banner.NZ, matrix_entries, filled_blocks)
          end
        end

        for color in off_diag_part.colors do
          var part = off_diag_part[color]
          -- c.printf("\t\tTRSM B=(%d, %d, %d) A=(%d, %d)\n", color.x, color.y, color.z, sep, sep)
          dtrsm(pivot, part)
        end

        if debug then
          var sizeA = pivot.bounds.hi - pivot.bounds.lo + {1, 1}
          var sizeB = off_diag.bounds.hi - off_diag.bounds.lo + {1, 1}

          c.printf("\tLevel: %d TRSM A=(%d, %d) B=(%d, %d)\n\tSizeA: %dx%d Lo: %d %d Hi: %d %d SizeB: %dx%d Lo: %d %d Hi: %d %d\n\n",
                   par_level, sep, sep, sep, par_sep,
                   sizeA.x, sizeA.y, pivot.bounds.lo.x, pivot.bounds.lo.y, pivot.bounds.hi.x, pivot.bounds.hi.y,
                   sizeB.x, sizeB.y, off_diag.bounds.lo.x, off_diag.bounds.lo.y, off_diag.bounds.hi.x, off_diag.bounds.hi.y)

          write_blocks(mat, mat_part, par_level, int2d{sep, sep}, int2d{sep, par_sep}, int2d{0, 0}, "TRSM", banner, debug_path)
        end
      end

      par_idx = sep_idx
      for par_level = level-1, -1, -1 do
        par_idx = par_idx/2
        var par_sep = tree[par_level][par_idx]

        var grandpar_idx = par_idx
        for grandpar_level = par_level, -1, -1 do
          var grandpar_sep = tree[grandpar_level][grandpar_idx]

          var A = mat_part[{sep, grandpar_sep}] -- ex: 16, 28
          var B = mat_part[{sep, par_sep}] -- ex: 16, 24
          var C = mat_part[{par_sep, grandpar_sep}] -- ex: 24, 28

          -- partition A (should be done in TRSM above) ex: 16, 28
          var A_part = partition_separator(grandpar_sep, sep, interval, clusters, A, debug)

          -- partition B (should be done in TRSM above) ex: 16, 24
          var B_part = partition_separator(par_sep, sep, interval, clusters, B, debug)

          -- partition C  ex: 24, 28
          var C_part = partition_separator(grandpar_sep, par_sep, interval, clusters, C, debug)

          if interval == 0 then
            for color in C_part.colors do
              if not filled_blocks[color] then
                fill_block(C_part[color], color, separators, banner.NZ, matrix_entries, filled_blocks)
              end
            end
          end

          var col_cluster_size = clusters[par_sep][interval][0]
          -- c.printf("\t\tA Vol: %d B Vol: %d C Vol: %d\n\n", A_colors.volume, B_colors.volume, C_colors.volume)

          if grandpar_sep == par_sep then
            for Acolor in A_part.colors do
              var row = Acolor.z
              var ABlock = A_part[Acolor]

              for Bcolor in A_part.colors do
                var col = Bcolor.z

                var Ccolor = int3d{grandpar_sep, par_sep, row*(col_cluster_size-1)+col}
                var CBlock = C_part[Ccolor]

                if col < row then
                  var BBlock = A_part[Bcolor]

                  -- c.printf("\t\tGEMM C=(%d, %d, %d) A=(%d, %d, %d) B=(%d, %d, %d)\n",
                  --          Ccolor.x, Ccolor.y, Ccolor.z,
                  --          Acolor.x, Acolor.y, Acolor.z,
                  --          Bcolor.x, Bcolor.y, Bcolor.z)

                  dgemm(ABlock, BBlock, CBlock)

                elseif col == row then

                  -- c.printf("\t\tSYRK C=(%d, %d, %d) A=(%d, %d, %d)\n",
                  --          Ccolor.x, Ccolor.y, Ccolor.z,
                  --          Acolor.x, Acolor.y, Acolor.z)
                  dsyrk(ABlock, CBlock)

                end
              end
            end
          else
            for Acolor in A_part.colors do
              var ABlock = A_part[Acolor]
              var row = Acolor.z

              for Bcolor in B_part.colors do
                var BBlock = B_part[Bcolor]
                var col = Bcolor.z

                var Ccolor = int3d{grandpar_sep, par_sep, row*(col_cluster_size-1)+col}
                var CBlock = C_part[Ccolor]

                -- c.printf("\t\tGEMM C=(%d, %d, %d) A=(%d, %d, %d) B=(%d, %d, %d)\n",
                --          Ccolor.x, Ccolor.y, Ccolor.z,
                --          Acolor.x, Acolor.y, Acolor.z,
                --          Bcolor.x, Bcolor.y, Bcolor.z)

                dgemm(ABlock, BBlock, CBlock)
              end
            end
          end

          if debug then
            var sizeA = A.bounds.hi - A.bounds.lo + {1, 1}
            var sizeB = B.bounds.hi - B.bounds.lo + {1, 1}
            var sizeC = C.bounds.hi - C.bounds.lo + {1, 1}

            c.printf("\tLevel: %d GEMM A=(%d, %d) B=(%d, %d) C=(%d, %d)\n\tSizeA: %dx%d Lo: %d %d Hi: %d %d SizeB: %dx%d Lo: %d %d Hi: %d %d SizeC: %dx%d Lo: %d %d Hi: %d %d\n\n",
                     grandpar_level,
                     sep, grandpar_sep,
                     sep, par_sep,
                     par_sep, grandpar_sep,
                     sizeA.x, sizeA.y, A.bounds.lo.x, A.bounds.lo.y, A.bounds.hi.x, A.bounds.hi.y,
                     sizeB.x, sizeB.y, B.bounds.lo.x, B.bounds.lo.y, B.bounds.hi.x, B.bounds.hi.y,
                     sizeC.x, sizeC.y, C.bounds.lo.x, C.bounds.lo.y, C.bounds.hi.x, C.bounds.hi.y)


            write_blocks(mat, mat_part, grandpar_level,
                         int2d{sep, grandpar_sep}, int2d{sep, par_sep}, int2d{par_sep, grandpar_sep}, "GEMM", banner, debug_path)
          end
          grandpar_idx = grandpar_idx/2
        end
      end
    end
    interval += 1
  end

  c.printf("Done factoring.\n")

  if c.strcmp(factor_file, '') ~= 0 then
    c.printf("saving factored matrix to: %s\n\n", factor_file)
    write_matrix(mat, mat_part, factor_file, banner)
  end

  if c.strcmp(b_file, '') == 0 then
    c.free(matrix_entries)

    for i = 0, num_separators+1 do
      c.free(separators[i])
    end
    c.free(separators)

    for i = 0, levels do
      c.free(tree[i])
    end

    c.free(tree)
    return
  end

  var Bentries = read_b(b_file, banner.N)
  var B = region(ispace(int1d, banner.N), double)
  var Bcoloring = c.legion_domain_point_coloring_create()
  var Bprev_size = 0
  var X = region(ispace(int1d, banner.N), double)

  for sep = 1, num_separators+1 do
    var size = separators[sep][0]
    var bounds = rect1d { Bprev_size, Bprev_size + size - 1}
    -- c.printf("Separator: %d Lo: %d Hi: %d Size: %d 1\n", sep, bounds.lo, bounds.hi, bounds.hi - bounds.lo + 1)
    var color:int1d = int1d{ sep }
    c.legion_domain_point_coloring_color_domain(Bcoloring, color:to_domain_point(), c.legion_domain_from_rect_1d(bounds))
    Bprev_size = Bprev_size + size
  end

  --c.printf("\n")

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

      -- c.printf("Level: %d TRSV A=(%d, %d) B=(%d)\nSizeA: %dx%d Lo: %d %d Hi: %d %d SizeB: %d Lo: %d 1 Hi: %d 1\n\n",
      --          level, sep, sep, sep,
      --          sizeA.x, sizeA.y, pivot.bounds.lo.x, pivot.bounds.lo.y, pivot.bounds.hi.x, pivot.bounds.hi.y,
      --          sizeB, bp.bounds.lo, bp.bounds.hi)

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

        -- c.printf("\tLevel: %d GEMV A=(%d, %d) X=(%d) Y=(%d)\n\tSizeA: %dx%d Lo: %d %d Hi: %d %d SizeX: %d Lo: %d 1 Hi: %d 1 SizeY: %d Lo: %d 1 Hi: %d 1\n\n",
        --          par_level, sep, par_sep, sep, par_sep,
        --          sizeA.x, sizeA.y, A.bounds.lo.x, A.bounds.lo.y, A.bounds.hi.x, A.bounds.hi.y,
        --          sizeX, X.bounds.lo, X.bounds.hi,
        --          sizeY, Y.bounds.lo, Y.bounds.hi)

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

      -- c.printf("Level: %d TRSV A=(%d, %d) B=(%d)\nSizeA: %dx%d Lo: %d %d Hi: %d %d SizeB: %d Lo: %d 1 Hi: %d 1\n\n",
      --          par_level, par_sep, par_sep, par_sep,
      --          sizeA.x, sizeA.y, pivot.bounds.lo.x, pivot.bounds.lo.y, pivot.bounds.hi.x, pivot.bounds.hi.y,
      --          sizeB, bp.bounds.lo, bp.bounds.hi)

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
            -- c.printf("\tLevel: %d GEMV A=(%d, %d) X=(%d) Y=(%d)\n\tSizeA: %dx%d Lo: %d %d Hi: %d %d SizeX: %d Lo: %d 1 Hi: %d 1 SizeY: %d Lo: %d 1 Hi: %d 1\n\n",
            --          level, sep, par_sep, par_sep, sep,
            --          sizeA.x, sizeA.y, A.bounds.lo.x, A.bounds.lo.y, A.bounds.hi.x, A.bounds.hi.y,
            --          sizeX, X.bounds.lo, X.bounds.hi,
            --          sizeY, Y.bounds.lo, Y.bounds.hi)

            dgemv(A, X, Y, cblas.CblasTrans)
          end
        end
      end
    end
  end

  c.printf("Done solve.\n")

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
