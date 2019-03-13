-- Copyright 2019 Stanford University
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

fspace Filled {
  nz:int,
  filled:int1d,
  sep:int2d
}

struct MMatBanner {
  M:int
  N:int
  NZ:int
  typecode:mmio.MM_typecode
}

fspace Cluster {
  bounds:rect2d,
  cluster:int3d,
  sep: int2d,
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

terra read_matrix(file:&c.FILE, nz:int, cols:uint64)
  for i = 0, nz do
    var entry:MatrixEntry
    c.fscanf(file, "%d %d %lg\n", &(entry.I), &(entry.J), &(entry.Val))
    entry.I = entry.I - 1
    entry.J = entry.J - 1
    var eidx:uint64 = entry.I*cols+entry.J
    mnd.add_entry(eidx, entry.Val)
  end
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

__demand(__leaf)
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

terra gen_filename(level:int, Ax:int, Ay:int, Bx:int, By:int, Cx:int, Cy:int, operation:rawstring, mm:bool, output_dir:regentlib.string)
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
                  level:int, A:int2d, B:int2d, C:int2d, operation:rawstring, banner:MMatBanner, output_dir:regentlib.string)
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

task partition_matrix(tree:&&int, separators:&&int, mat:region(ispace(int2d), double), pprev_size:int2d, debug:bool)

  var coloring = c.legion_domain_point_coloring_create()
  var levels = separators[0][0]
  var num_separators = separators[0][1]
  var prev_size = pprev_size
  var separator_bounds = region(ispace(int1d, num_separators, 1), rect2d)

  for level = 0, levels do
    for sep_idx = 0, [int](math.pow(2, level)) do
      var sep = tree[level][sep_idx]
      var size = separators[sep][0]
      var bounds = rect2d { prev_size - {size-1, size-1}, prev_size }

      separator_bounds[sep] = bounds

      if debug then
        c.printf("level: %d sep: %d size: %d ", level, sep, size)
        c.printf("prev_size: %d %d bounds.lo: %d %d, bounds.hi: %d %d vol: %d\n",
                 prev_size.x, prev_size.y,
                 bounds.lo.x, bounds.lo.y,
                 bounds.hi.x, bounds.hi.y,
                 c.legion_domain_get_volume(c.legion_domain_from_rect_2d(bounds)))
      end

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

        if debug then
          c.printf("block: %d %d bounds.lo: %d %d bounds.hi: %d %d\n", par_sep, sep,
                   child_bounds.lo.x, child_bounds.lo.y, child_bounds.hi.x, child_bounds.hi.y)
        end

        var color:int2d = {par_sep, sep}
        c.legion_domain_point_coloring_color_domain(coloring, color:to_domain_point(), c.legion_domain_from_rect_2d(child_bounds))
      end
    end
  end

  var colors = ispace(int2d, {num_separators, num_separators}, {1, 1})
  var mat_part = partition(disjoint, mat, coloring, colors)
  c.legion_domain_point_coloring_destroy(coloring)
  return mat_part
end

__demand(__leaf)
task partition_separator(cluster_bounds:region(ispace(int3d), Cluster), block_color:int2d, block_bounds:rect2d,
                         interval:int, clusters:&&&int, debug:bool)
where
  reads writes(cluster_bounds.bounds)
do
  for i in cluster_bounds do
    cluster_bounds[i].bounds = rect2d{lo=int2d{0, 0}, hi=int2d{-1, -1}}
  end

  var row_sep = block_color.x
  var row_cluster = clusters[row_sep]
  var row_cluster_size = row_cluster[interval][0]

  var col_sep = block_color.y
  var col_cluster = clusters[col_sep]
  var col_cluster_size = col_cluster[interval][0]
  var prev_lo = block_bounds.lo

  if debug then
    c.printf("\t\tPartitioning (%d, %d) Cluster: %d Rows: %d Cols: %d\n",
             row_sep, col_sep, interval, row_cluster_size-1, col_cluster_size-1)
  end

  for row = 1, row_cluster_size do

    var top = row_cluster[interval][row]
    var bottom = row_cluster[interval][row+1]

    for i = interval-1, -1, -1 do
      top = row_cluster[i][top+1]
      bottom = row_cluster[i][bottom+1]
    end

    for col = 1, col_cluster_size do

      var left = col_cluster[interval][col]
      var right = col_cluster[interval][col+1]

      for i = interval-1, -1, -1 do
        left = col_cluster[i][left+1]
        right = col_cluster[i][right+1]
      end

      var part_size = int2d{x = bottom - top - 1, y = right - left - 1}
      var color:int3d = {x = row_sep, y = col_sep, z = (row-1)*(col_cluster_size-1)+(col-1)}
      var bounds = rect2d { prev_lo, prev_lo + part_size }
      var size = bounds.hi - bounds.lo + {1, 1}

      cluster_bounds[color].bounds = bounds

      if debug then
        c.printf("\t\tcolor: %d %d %d bounds.lo: %d %d, bounds.hi: %d %d size: %d %d vol: %d\n",
                 color.x, color.y, color.z,
                 bounds.lo.x, bounds.lo.y,
                 bounds.hi.x, bounds.hi.y,
                 size.x, size.y,
                 c.legion_domain_get_volume(c.legion_domain_from_rect_2d(bounds)))
      end

      prev_lo = prev_lo + int2d{0, right-left}
    end

    prev_lo = int2d{prev_lo.x + bottom-top, block_bounds.lo.y}

    if debug then
      c.printf("\n")
    end
  end
end

__demand(__leaf)
task fill_block(block:region(ispace(int2d), double), color:int2d, separators:&&int, clusters:&&&int, cols:uint64,
                filled_blocks:region(ispace(int3d), Filled), debug:bool)
where
  reads writes(block, filled_blocks)
do
  var row_sep = color.x
  var row_dofs = separators[row_sep]
  var row_size = row_dofs[0]
  var row_cluster = clusters[row_sep][0]
  var row_cluster_size = row_cluster[0]

  var col_sep = color.y
  var col_dofs = separators[col_sep]
  var col_size = col_dofs[0]
  var col_cluster = clusters[col_sep][0]
  var col_cluster_size = col_cluster[0]

  -- c.printf("Filling: %d %d From: %d %d To: %d %d\n",
  --          color.x, color.y, block.bounds.lo.x, block.bounds.lo.y, block.bounds.hi.x, block.bounds.hi.y)

  var nz = 0
  var block_idx = block.bounds.lo

  for col = 1, col_cluster_size do
    var left = col_cluster[col]
    var right = col_cluster[col+1]
    var col_bound = right - left

    for row = 1, row_cluster_size do
      var top = row_cluster[row]
      var bottom = row_cluster[row+1]
      var row_bound = bottom - top

      var z = (row-1)*(col_cluster_size-1)+(col-1)

      var nnz = 0
      -- c.printf("Bounds: %d %d %d row: %d %d col: %d %d\n", row_sep, col_sep, z, top, bottom, left, right)
      var cidx = int2d{0, 0}
      for i = top, bottom do
        for j = left, right do
          var idxi = row_dofs[i+1]
          var idxj = col_dofs[j+1]

          if idxj > idxi then
            var t = idxi
            idxi = idxj
            idxj = t
          end

          var eidx:uint64 = idxi*cols+idxj
          var entry = mnd.find_entry(eidx)
          var idx = block_idx + cidx
          if row_sep == col_sep and idx.y <= idx.x then
            -- c.printf("Block: %d %d %d Filling Diagonal: %d %d I: %d J: %d key: %lu Entry: %0.2f\n",
            --          row_sep, col_sep, z, idx.x, idx.y, idxi, idxj, eidx, entry)
            if entry ~= 0 then
              block[idx] = entry
              nnz += 1
            end
          elseif row_sep ~= col_sep then
            -- c.printf("Block: %d %d %d Filling Off-Diagonal: %d %d I: %d J: %d key: %lu Entry: %0.2f\n",
            --          row_sep, col_sep, z, idx.x, idx.y, idxi, idxj, eidx, entry)
            if entry ~= 0 then
              block[idx] = entry
              nnz += 1
            end
          end
          cidx = cidx + {0, 1}
        end
        cidx = {cidx.x + 1, 0}
      end
      block_idx = block_idx + {row_bound, 0}

      var color = int3d{x = row_sep, y = col_sep, z = z}
      filled_blocks[color].nz = nnz

      if nnz > 0 then
        filled_blocks[color].filled = 0
      end

      if debug then
        if nnz > 0 then
          c.printf("Filled: %d %d %d with %d non-zeros\n", color.x, color.y, color.z, nnz)
        else
          c.printf("Block %d %d %d empty\n", color.x, color.y, color.z)
        end
      end

      nz += nnz
    end

    block_idx = {block.bounds.lo.x, block_idx.y+col_bound}
  end

  return nz
end

--__demand(__inline)
task merge_filled_blocks(allocated_blocks:ispace(int2d), filled_blocks:region(ispace(int3d), Filled),
                         num_separators:int, interval:int, clusters:&&&int)
where
  reads writes(filled_blocks)
do
  var blocks = region(filled_blocks.ispace, Filled)
  copy(filled_blocks, blocks)

  -- max_int_size = clusters[num_separators][interval][0]
  -- filled_blocks = region(ispace(int3d {num_separators, num_separators, max_int_size}, {1, 1, 0}), bool)
  fill(filled_blocks.nz, 0)
  fill(filled_blocks.filled, 1)

  c.printf("Merging blocks from interval: %d\n", interval-1)
  for block in allocated_blocks do
    var row_sep = block.x
    var col_sep = block.y

    if interval < clusters[0][0][row_sep] and interval < clusters[0][0][col_sep] then

      var row_cluster = clusters[row_sep][interval]
      var row_cluster_size = row_cluster[0]

      var col_cluster = clusters[col_sep][interval]
      var col_cluster_size = col_cluster[0]

      var prev_cols = clusters[col_sep][interval-1][0]-1

      for row = 1, row_cluster_size do

        var top = row_cluster[row]
        var bottom = row_cluster[row+1]

        for col = 1, col_cluster_size do

          var left = col_cluster[col]
          var right = col_cluster[col+1]
          var new_id = int3d{row_sep, col_sep, (row-1)*(col_cluster_size-1)+(col-1)}
          -- c.printf("New Id: %d %d %d -> ", new_id.x, new_id.y, new_id.z)

          for i = top, bottom do
            for j = left, right do
              var old_id = int3d{row_sep, col_sep, i*prev_cols+j}
              -- c.printf("%d ", old_id.z)
              filled_blocks[new_id].nz = filled_blocks[new_id].nz + blocks[old_id].nz
              if filled_blocks[new_id].nz > 0 then
                filled_blocks[new_id].filled = 0
              end
            end
          end
          -- c.printf("\n")
        end
      end
    end
  end

  -- for block in filled_blocks.ispace do
  --   c.printf("Merged: %d %d %d Interval: %d NZ: %d\n", block.x, block.y, block.z, interval, filled_blocks[block].nz)
  -- end
end

--__demand(__inline)
task find_color_space(color:int2d, interval:int, clusters:&&&int, filled_ispace:ispace(int3d))
  var rows = clusters[color.x][interval][0]-1
  var cols = clusters[color.y][interval][0]-1

  var colors = ispace(int3d, {1, 1, rows*cols}, {color.x, color.y, 0})
  return filled_ispace & colors
end

__demand(__leaf)
task find_index_space_3d(levels:int, blocks:region(ispace(int3d), int1d), tree:&&int, clusters:&&&int)
where
  reads writes (blocks)
do
    for lvl = 0, levels do
      for sep_idx = 0, [int](math.pow(2, lvl)) do
        var row_sep = tree[lvl][sep_idx]
        var row_cluster_size = clusters[row_sep][0][0]-1

        for i = 0, row_cluster_size do
          for j = 0, row_cluster_size do
            -- c.printf("Block: %d %d %d\n", row_sep, row_sep, i*row_cluster_size+j)
            blocks[{row_sep, row_sep, i*row_cluster_size+j}] = 1
          end
        end

        for clvl = lvl+1, levels do
          for csep_idx = [int](sep_idx*math.pow(2, clvl-lvl)), [int]((sep_idx+1)*math.pow(2, clvl-lvl)) do
            var col_sep = tree[clvl][csep_idx]
            var col_cluster_size = clusters[col_sep][0][0]-1

            for i = 0, row_cluster_size do
              for j = 0, col_cluster_size do
                -- c.printf("Block: %d %d %d\n", row_sep, col_sep, i*col_cluster_size+j)
                blocks[{row_sep, col_sep, i*col_cluster_size+j}] = 1
              end
            end
          end
        end
      end
    end
end

__demand(__leaf)
task find_index_space_2d(levels:int, blocks:region(ispace(int2d), int1d), tree:&&int)
where
  reads writes (blocks)
do
    for lvl = 0, levels do
      for sep_idx = 0, [int](math.pow(2, lvl)) do
        var row_sep = tree[lvl][sep_idx]

        -- c.printf("Block: %d %d %d\n", row_sep, row_sep, i*row_cluster_size+j)
        blocks[{row_sep, row_sep}] = 1

        for clvl = lvl+1, levels do
          for csep_idx = [int](sep_idx*math.pow(2, clvl-lvl)), [int]((sep_idx+1)*math.pow(2, clvl-lvl)) do
            var col_sep = tree[clvl][csep_idx]
            -- c.printf("Block: %d %d %d\n", row_sep, col_sep, i*col_cluster_size+j)
            blocks[{row_sep, col_sep}] = 1
          end
        end
      end
    end
end

__demand(__leaf)
task fill_b(separators:&&int, Bentries:&double, BP:region(ispace(int1d), double), Bcolor:int1d)
where
  reads writes(BP)
do
  var ib:int = [int](Bcolor)
  var sep = separators[ib]
  var sep_size = sep[0]

  for i = 0, sep_size do
    var idxi = sep[i+1]
    BP[BP.bounds.lo + i] = Bentries[idxi]
  end
end

__demand(__leaf)
task unpermute_solution(separators:&&int, X:region(ispace(int1d), double), B:region(ispace(int1d), double))
where
  reads(B), reads writes(X)
do
  var num_separators = separators[0][1]
  var j = 0
  for sep = 1, num_separators+1 do
    var size = separators[sep][0]
    for i = 1, size+1 do
      var idxi = separators[sep][i]
      X[idxi] = B[j]
      j += 1
    end
  end
end

__demand(__leaf)
task write_solution(solution_file:regentlib.string, X:region(ispace(int1d), double))
where
  reads(X)
do
  c.printf("Saving solution to: %s\n", solution_file)
  var solution = c.fopen(solution_file, 'w')

  for i in X.ispace do
    c.fprintf(solution, "%0.5g\n", X[i])
  end

  c.fclose(solution)
end

__demand(__leaf)
task update_filled_blocks(filled_blocks:region(ispace(int3d), Filled), Acolor:int3d, Bcolor:int3d, Ccolor:int3d)
where
  reads writes(filled_blocks)
do
  if filled_blocks[Ccolor].nz <= 0 then
    filled_blocks[Ccolor].nz = filled_blocks[Acolor].nz*filled_blocks[Bcolor].nz
    filled_blocks[Ccolor].filled = 0
  end
end

local function generate_partition(r_type)
  local r = regentlib.newsymbol(r_type, "r")
  local n = regentlib.newsymbol(int, "n")

  local task fill_sep([r], [n])
  where
    reads writes(r.sep)
  do
    for color in r.ispace do
      r[color].sep = int2d{color.x, color.y}
    end

    var part = partition(r.sep, ispace(int2d, {n, n}, {1, 1}))
    return part
  end

  return fill_sep
end

local partition_filled_blocks_by_sep = generate_partition(region(ispace(int3d), Filled))
local partition_cluster_bounds_by_sep = generate_partition(region(ispace(int3d), Cluster))


task partition_cluster_bounds_by_cluster(cluster_bounds:region(ispace(int3d), Cluster))
where
  reads writes(cluster_bounds.cluster)
do
  for i in cluster_bounds.ispace do
    cluster_bounds[i].cluster = i
  end

  var part = partition(cluster_bounds.cluster, cluster_bounds.ispace)
  return part
end


task partition_by_image_range(mat:region(ispace(int2d), double),
                              cluster_bounds:region(ispace(int3d), Cluster),
                              cluster_part:partition(disjoint, cluster_bounds, ispace(int3d)))
where
  reads(cluster_bounds.bounds)
do
  var fid = __fields(cluster_bounds)[0]

  var ip = c.legion_index_partition_create_by_image_range(__runtime(),
                                                          __context(),
                                                          __raw(mat.ispace),
                                                          __raw(cluster_part),
                                                          __raw(cluster_bounds),
                                                          fid,
                                                          __raw(cluster_bounds.ispace),
                                                          c.DISJOINT_KIND,
                                                          -1)

  var raw_part = c.legion_logical_partition_create(__runtime(), __context(), __raw(mat), ip)

  return __import_partition(disjoint, mat, cluster_bounds.ispace, raw_part)
end

task fused_dpotrf(rA:region(ispace(int2d), double), cluster_bounds:region(ispace(int3d), Cluster), filled:ispace(int3d))
where
  reads writes(rA), reads(cluster_bounds)
do
  for color in filled do
    var rectA = cluster_bounds[color]
    var size:int2d = rectA.bounds.hi - rectA.bounds.lo + {1, 1}
    dpotrf_terra(rectA.bounds, size.x, __physical(rA)[0], __fields(rA)[0])
  end
end

task fused_dtrsm(rA:region(ispace(int2d), double),
                 rB:region(ispace(int2d), double),
                 cluster_bounds_rA:region(ispace(int3d), Cluster),
                 cluster_bounds_rB:region(ispace(int3d), Cluster),
                 filled_rA:ispace(int3d),
                 filled_rB:ispace(int3d))
where
  reads(rA, cluster_bounds_rA, cluster_bounds_rB), reads writes(rB)
do
  for acolor in filled_rA do
    var rectA = cluster_bounds_rA[acolor].bounds
    for bcolor in filled_rB do
      var rectB = cluster_bounds_rB[bcolor].bounds
      var size:int2d = rectB.hi - rectB.lo + {1, 1}
      dtrsm_terra(rectA, rectB, size.x, size.y,
                  __physical(rA)[0], __fields(rA)[0],
                  __physical(rB)[0], __fields(rB)[0])
    end
  end
end


__demand(__inner)
task main()
  var args = c.legion_runtime_get_input_args()

  var matrix_file_path = ""
  var separator_file = ""
  var clusters_file = ""
  var b_file = ""
  var solution_file = ""
  var factor_file = ""
  var permuted_matrix_file = ""
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
    elseif c.strcmp(args.argv[i], "-p") == 0 then
      permuted_matrix_file = args.argv[i+1]
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

  var clusters = mnd.read_clusters(clusters_file, num_separators+1, num_separators, num_separators)
  -- var clusters = mnd.read_clusters2(clusters_file, 2048)

  c.printf("levels: %d\n", levels)
  c.printf("separators: %d\n", num_separators)

  var mat = region(ispace(int2d, {x = banner.M, y = banner.N}), double)
  fill(mat, 0)

  var cols:uint64 = [uint64](banner.N)
  read_matrix(matrix_file, banner.NZ, cols)
  c.fclose(matrix_file)

  var prev_size = int2d{x = banner.M-1, y = banner.N-1}
  var mat_part = partition_matrix(tree, separators, mat, prev_size, debug)

  var nz = 0
  var interval = 0

  -- find_max_int_sizes()
  var max_int_size = 0
  for i = 1, num_separators+1 do
      max_int_size = max(clusters[i][0][0]-1, max_int_size)
  end

  var blocks = region(ispace(int2d, {num_separators, num_separators}, {1, 1}), int1d)
  fill(blocks, 0)
  var allocated_blocks = find_index_space_2d(levels, blocks, tree)
  var allocated_blocks_part = partition(blocks, ispace(int1d, 2))
  var allocated_blocks_ispace = allocated_blocks_part[1].ispace

  -- var allocated_blocks_ispace = ispace(int3d, {num_separators, num_separators, max_int_size*max_int_size}, {1, 1, 0})
  var cluster_blocks = region(ispace(int3d, {num_separators, num_separators, max_int_size*max_int_size}, {1, 1, 0}), int1d)
  fill(cluster_blocks, 0)
  var allocated_cluster_blocks = find_index_space_3d(levels, cluster_blocks, tree, clusters)
  var allocated_cluster_blocks_part = partition(cluster_blocks, ispace(int1d, 2))
  var allocated_cluster_blocks_ispace = allocated_cluster_blocks_part[1].ispace

  var filled_blocks = region(allocated_cluster_blocks_ispace, Filled)
  fill(filled_blocks.nz, -1)
  fill(filled_blocks.filled, 1)
  fill(filled_blocks.sep, int2d{-1, -1})
  var filled_block_part = partition_filled_blocks_by_sep(filled_blocks, num_separators)

  var cluster_bounds = region(allocated_cluster_blocks_ispace, Cluster)
  fill(cluster_bounds.sep, int2d{-1, -1})
  fill(cluster_bounds.bounds, rect2d{lo=int2d{0, 0}, hi=int2d{-1, -1}})
  var cluster_bounds_part = partition_cluster_bounds_by_sep(cluster_bounds, num_separators)
  var cpart = partition_cluster_bounds_by_cluster(cluster_bounds)

  for level = levels-1, -1, -1 do

    for lvl = 0, level+1 do
      for sep_idx = 0, [int](math.pow(2, lvl)) do
        var row = tree[lvl][sep_idx]
        -- c.printf("Partitioning: %d %d\n", row, row)
        var block_color = int2d{row, row}
        var block_bounds = mat_part[block_color].bounds
        var cluster = cluster_bounds_part[block_color]
        partition_separator(cluster, block_color, block_bounds, interval, clusters, debug)

        for clvl = lvl+1, level+1 do
          for csep_idx = [int](sep_idx*math.pow(2, clvl-lvl)), [int]((sep_idx+1)*math.pow(2, clvl-lvl)) do
            var col = tree[clvl][csep_idx]
            -- c.printf("Partitioning: %d %d\n", row, col)
            var block_color = int2d{row, col}
            var block_bounds = mat_part[block_color].bounds
            var cluster = cluster_bounds_part[block_color]
            partition_separator(cluster, block_color, block_bounds, interval, clusters, debug)
          end
        end
      end
    end

    -- var cpart = partition(equal, cluster_bounds, cluster_bounds.ispace)
    -- var sep_part = image(mat, cpart, cluster_bounds.bounds)
    var sep_part = partition_by_image_range(mat, cluster_bounds, cpart)

    if interval == 0 then
      for color in mat_part.colors do
        var block = mat_part[color]
        if block.volume ~= 0 then
          -- c.printf("Filling: %d %d\n", color.x, color.y)
          var fpblock = filled_block_part[color]
          fill(block, 0)
          nz += fill_block(block, color, separators, clusters, cols, fpblock, debug)
          -- regentlib.assert(nz <= banner.NZ, "Mismatch in number of entries.")
        end
      end

      c.printf("Filled: %d Expected: %d\n", nz, banner.NZ)

      if c.strcmp(permuted_matrix_file, '') ~= 0 then
        c.printf("saving permuted matrix to: %s\n\n", permuted_matrix_file)
        write_matrix(mat, mat_part, permuted_matrix_file, banner)
      end

      regentlib.assert(nz == banner.NZ, "Mismatch in number of entries.")
    end

    var filled_part = partition(filled_blocks.filled, ispace(int1d, 2))
    var filled_ispace = filled_part[0].ispace

    for sep_idx = 0, [int](math.pow(2, level)) do
      var sep = tree[level][sep_idx]
      var pivot_color = int2d{sep, sep}
      var filled_pivot = find_color_space(pivot_color, interval, clusters, filled_ispace)

      fused_dpotrf(mat_part[pivot_color], cluster_bounds_part[pivot_color], filled_pivot)
      -- __demand(__parallel)
      -- for color in filled_pivot do
      --   dpotrf(sep_part[color])
      -- end

      if debug then
        var pivot = mat_part[pivot_color]
        var sizeA = pivot.bounds.hi - pivot.bounds.lo + {1, 1}

        c.printf("Level: %d POTRF A=(%d, %d)\nSize: %dx%d Lo: %d %d Hi: %d %d\n\n",
                 level, sep, sep, sizeA.x, sizeA.y,
                 pivot.bounds.lo.x, pivot.bounds.lo.y, pivot.bounds.hi.x, pivot.bounds.hi.y)

        write_blocks(mat, mat_part, level, pivot_color, int2d{0, 0}, int2d{0, 0}, "POTRF", banner, debug_path)
      end
    end

    for sep_idx = 0, [int](math.pow(2, level)) do
      var sep = tree[level][sep_idx]
      var pivot_color = int2d{sep, sep}
      var filled_pivot = find_color_space(pivot_color, interval, clusters, filled_ispace)

      var par_idx = sep_idx
      for par_level = level-1, -1, -1 do

        par_idx = par_idx/2
        var par_sep = tree[par_level][par_idx]
        var off_diag_color = int2d{par_sep, sep}
        var filled_off_diag = find_color_space(off_diag_color, interval, clusters, filled_ispace)

        fused_dtrsm(mat_part[pivot_color], mat_part[off_diag_color],
                    cluster_bounds_part[pivot_color], cluster_bounds_part[off_diag_color],
                    filled_pivot, filled_off_diag)
        -- for pcolor in filled_pivot do
        --   for ocolor in filled_off_diag do
        --     dtrsm(sep_part[pcolor], sep_part[ocolor])
        --   end
        -- end

        if debug then
          var pivot = mat_part[pivot_color]
          var sizeA = pivot.bounds.hi - pivot.bounds.lo + {1, 1}
          var off_diag = mat_part[off_diag_color]
          var sizeB = off_diag.bounds.hi - off_diag.bounds.lo + {1, 1}

          c.printf("\tLevel: %d TRSM A=(%d, %d) B=(%d, %d)\n\tSizeA: %dx%d Lo: %d %d Hi: %d %d SizeB: %dx%d Lo: %d %d Hi: %d %d\n\n",
                   par_level, sep, sep, par_sep, sep,
                   sizeA.x, sizeA.y, pivot.bounds.lo.x, pivot.bounds.lo.y, pivot.bounds.hi.x, pivot.bounds.hi.y,
                   sizeB.x, sizeB.y, off_diag.bounds.lo.x, off_diag.bounds.lo.y, off_diag.bounds.hi.x, off_diag.bounds.hi.y)

          write_blocks(mat, mat_part, par_level, pivot_color, off_diag_color, int2d{0, 0}, "TRSM", banner, debug_path)
        end
      end
    end

    for sep_idx = 0, [int](math.pow(2, level)) do
      var sep = tree[level][sep_idx]
      var par_idx = sep_idx

      for par_level = level-1, -1, -1 do
        par_idx = par_idx/2
        var par_sep = tree[par_level][par_idx]

        var grandpar_idx = par_idx
        for grandpar_level = par_level, -1, -1 do
          var grandpar_sep = tree[grandpar_level][grandpar_idx]

          var A_color = int2d{grandpar_sep, sep}
          var B_color = int2d{par_sep, sep}
          var C_color = int2d{grandpar_sep, par_sep}

          var col_cluster_size = clusters[par_sep][interval][0]

          var filled_A_colors = find_color_space(A_color, interval, clusters, filled_ispace)
          var filled_B_colors = find_color_space(B_color, interval, clusters, filled_ispace)

          if grandpar_sep == par_sep then
            for Acolor in filled_A_colors do
              var row = Acolor.z
              var ABlock = sep_part[Acolor]

              for Bcolor in filled_B_colors do
                var col = Bcolor.z

                var Ccolor = int3d{grandpar_sep, par_sep, row*(col_cluster_size-1)+col}
                var CBlock = sep_part[Ccolor]

                if col < row then
                  var BBlock = sep_part[Bcolor]

                  -- c.printf("\t\tGEMM C=(%d, %d, %d) A=(%d, %d, %d) B=(%d, %d, %d)\n",
                  --          Ccolor.x, Ccolor.y, Ccolor.z,
                  --          Acolor.x, Acolor.y, Acolor.z,
                  --          Bcolor.x, Bcolor.y, Bcolor.z)

                  dgemm(ABlock, BBlock, CBlock)
                  update_filled_blocks(filled_blocks, Acolor, Bcolor, Ccolor)

                elseif col == row then

                  -- c.printf("\t\tSYRK C=(%d, %d, %d) A=(%d, %d, %d)\n",
                  --          Ccolor.x, Ccolor.y, Ccolor.z,
                  --          Acolor.x, Acolor.y, Acolor.z)

                  dsyrk(ABlock, CBlock)
                  update_filled_blocks(filled_blocks, Acolor, Ccolor, Ccolor)
                end
              end
            end
          else
            for Acolor in filled_A_colors do
              var ABlock = sep_part[Acolor]
              var row = Acolor.z

              for Bcolor in filled_B_colors do
                var BBlock = sep_part[Bcolor]
                var col = Bcolor.z

                var Ccolor = int3d{grandpar_sep, par_sep, row*(col_cluster_size-1)+col}
                var CBlock = sep_part[Ccolor]

                -- c.printf("\t\tGEMM C=(%d, %d, %d) A=(%d, %d, %d) B=(%d, %d, %d)\n",
                --          Ccolor.x, Ccolor.y, Ccolor.z,
                --          Acolor.x, Acolor.y, Acolor.z,
                --          Bcolor.x, Bcolor.y, Bcolor.z)

                dgemm(ABlock, BBlock, CBlock)
                update_filled_blocks(filled_blocks, Acolor, Bcolor, Ccolor)

              end
            end
          end

          if debug then
            var A = mat_part[A_color]
            var B = mat_part[B_color]
            var C = mat_part[C_color]
            var sizeA = A.bounds.hi - A.bounds.lo + {1, 1}
            var sizeB = B.bounds.hi - B.bounds.lo + {1, 1}
            var sizeC = C.bounds.hi - C.bounds.lo + {1, 1}

            c.printf("\tLevel: %d GEMM A=(%d, %d) B=(%d, %d) C=(%d, %d)\n\tSizeA: %dx%d Lo: %d %d Hi: %d %d SizeB: %dx%d Lo: %d %d Hi: %d %d SizeC: %dx%d Lo: %d %d Hi: %d %d\n\n",
                     grandpar_level,
                     grandpar_sep, sep,
                     par_sep, sep,
                     grandpar_sep, par_sep,
                     sizeA.x, sizeA.y, A.bounds.lo.x, A.bounds.lo.y, A.bounds.hi.x, A.bounds.hi.y,
                     sizeB.x, sizeB.y, B.bounds.lo.x, B.bounds.lo.y, B.bounds.hi.x, B.bounds.hi.y,
                     sizeC.x, sizeC.y, C.bounds.lo.x, C.bounds.lo.y, C.bounds.hi.x, C.bounds.hi.y)

            write_blocks(mat, mat_part, grandpar_level, A_color, B_color, C_color, "GEMM", banner, debug_path)
          end
          grandpar_idx = grandpar_idx/2
        end
      end
    end
    interval += 1
    merge_filled_blocks(allocated_blocks_ispace, filled_blocks, num_separators, interval, clusters)
  end

  c.printf("Done factoring.\n")

  if c.strcmp(factor_file, '') ~= 0 then
    c.printf("saving factored matrix to: %s\n\n", factor_file)
    write_matrix(mat, mat_part, factor_file, banner)
  end

  if c.strcmp(b_file, '') == 0 then
    __fence(__execution, __block)

    for i = 0, num_separators+1 do
      c.free(separators[i])
    end
    c.free(separators)

    for i = 0, levels do
      c.free(tree[i])
    end

    c.free(tree)

    mnd.delete_entries()
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
    fill_b(separators, Bentries, part, Bcolor)
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

        var A = mat_part[{par_sep, sep}]
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

          var A = mat_part[{par_sep, sep}]
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

  unpermute_solution(separators, X, B)

  if c.strcmp(solution_file, '') ~= 0 then
    write_solution(solution_file, X)
  end

  __fence(__execution, __block)
  for i = 0, num_separators+1 do
    c.free(separators[i])
  end
  c.free(separators)

  for i = 0, levels do
    c.free(tree[i])
  end
  c.free(tree)

  c.free(Bentries)

  mnd.delete_entries()
end

regentlib.start(main)
