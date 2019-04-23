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

terralib.linklibrary("./libmmio.so")
local mmio = terralib.includec("mmio.h")
local mnd = terralib.includec("mnd.h")
local math = terralib.includec("math.h")

terralib.linklibrary("./libcholesky.so")
local runtime_dir = os.getenv('LG_RT_DIR') .. "/"
local ccholesky = terralib.includec("cholesky.h", {"-I", runtime_dir})
-- print(c.legion_domain_from_index_space)

local blas = require("blas")
terralib.linklibrary("libcblas.so")
local cblas = terralib.includec("cblas.h")

struct MMatBanner {
  M  : int
  N  : int
  NZ : int
  typecode : mmio.MM_typecode
}

fspace MatrixEntry {
  idx : int2d,
  val : double
}

fspace SepIndex {
  idx : int,
  sep : int1d
}

fspace ClusterIndex {
  idx      : int,
  interval : int1d,
  sep      : int1d
}

fspace TreeNode {
  node  : int,
  level : int1d
}

fspace Filled {
  nz       : int,
  filled   : int1d,
  sep      : int2d,
  interval : int1d,
  cluster  : int1d,
  bounds   : rect2d
}

fspace BlockBounds {
  bounds : rect2d,
  sep    : int2d
}

fspace ClusterBounds {
  bounds  : rect2d,
  sep     : int2d,
  cluster : int3d
}

fspace VecBounds {
  bounds : rect1d,
  sep    : int1d
}

terra read_matrix_banner(file : regentlib.string)
  var banner:MMatBanner
  var ret:int

  var fp = c.fopen(file, 'r')
  ret = mmio.mm_read_banner(fp, &(banner.typecode))

  if ret ~= 0 then
    c.printf("Unable to read banner.\n")
    return MMatBanner{0, 0, 0}
  end

  var M : int[1]
  var N : int[1]
  var nz : int[1]
  ret = mmio.mm_read_mtx_crd_size(fp, &(banner.M), &(banner.N), &(banner.NZ))

  if ret ~= 0 then
    c.printf("Unable to read matrix size.\n")
    return MMatBanner{0, 0, 0}
  end

  c.fclose(fp);
  return banner
end

__demand(__leaf)
task write_matrix(mat      : region(ispace(int2d), double),
                  mat_part : partition(disjoint, mat, ispace(int2d)),
                  file     : regentlib.string,
                  banner   : MMatBanner)
where
  reads(mat)
do
  c.printf("saving matrix to: %s\n\n", file)
  var matrix_file = c.fopen(file, 'w')
  var nnz = 0

  for color in mat_part.colors do
    var part = mat_part[color]
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

  mmio.mm_write_banner(matrix_file, banner.typecode)
  mmio.mm_write_mtx_crd_size(matrix_file, banner.M, banner.N, nnz)

  for color in mat_part.colors do
    var part = mat_part[color]
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

task partition_vec_by_image_range(vec        : region(ispace(int1d), double),
                                  vec_bounds : region(ispace(int1d), VecBounds),
                                  vec_part   : partition(disjoint, vec_bounds, ispace(int1d)))
where
  reads(vec_bounds.bounds)
do
  var fid = __fields(vec_bounds)[0]

  var ip = c.legion_index_partition_create_by_image_range(__runtime(),
                                                          __context(),
                                                          __raw(vec.ispace),
                                                          __raw(vec_part),
                                                          __raw(vec_bounds),
                                                          fid,
                                                          __raw(vec_bounds.ispace),
                                                          c.DISJOINT_KIND,
                                                          -1)

  var raw_part = c.legion_logical_partition_create(__runtime(), __context(), __raw(vec), ip)

  return __import_partition(disjoint, vec, vec_bounds.ispace, raw_part)
end

task partition_mat_by_image_range(mat          : region(ispace(int2d), double),
                                  block_bounds : region(ispace(int2d), BlockBounds),
                                  block_part   : partition(disjoint, block_bounds, ispace(int2d)))
where
  reads(block_bounds.bounds)
do
  var fid = __fields(block_bounds)[0]

  var ip = c.legion_index_partition_create_by_image_range(__runtime(),
                                                          __context(),
                                                          __raw(mat.ispace),
                                                          __raw(block_part),
                                                          __raw(block_bounds),
                                                          fid,
                                                          __raw(block_bounds.ispace),
                                                          c.DISJOINT_KIND,
                                                          -1)

  var raw_part = c.legion_logical_partition_create(__runtime(), __context(), __raw(mat), ip)

  return __import_partition(disjoint, mat, block_bounds.ispace, raw_part)
end

__demand(__leaf)
task partition_matrix(sepinfo           : mnd.SepInfo,
                      separators_region : region(ispace(int1d), SepIndex),
                      separators        : partition(disjoint, separators_region, ispace(int1d)),
                      tree_region       : region(ispace(int1d), TreeNode),
                      tree              : partition(disjoint, tree_region, ispace(int1d)),
                      banner            : MMatBanner,
                      mat               : region(ispace(int2d), double),
                      block_bounds      : region(ispace(int2d), BlockBounds),
                      debug             : bool)
where
  reads(separators_region, tree_region), reads writes(block_bounds)
do

  var levels = sepinfo.levels
  var num_separators = sepinfo.num_separators
  var prev_size = int2d{x = banner.M-1, y = banner.N-1}

  for lvl = 0, levels do
    var level = tree[lvl]
    for sep_idx in level.ispace do
      var sep = level[sep_idx].node
      var size = separators[sep].volume
      var bounds = rect2d { prev_size - {size-1, size-1}, prev_size }

      if debug then
        c.printf("level: %d sep: %d size: %d ", lvl, sep, size)
        c.printf("prev_size: %d %d bounds.lo: %d %d, bounds.hi: %d %d vol: %d\n",
                 prev_size.x, prev_size.y,
                 bounds.lo.x, bounds.lo.y,
                 bounds.hi.x, bounds.hi.y,
                 c.legion_domain_get_volume(c.legion_domain_from_rect_2d(bounds)))
      end

      var color:int2d = {x = sep, y = sep}
      block_bounds[color].sep = color
      block_bounds[color].bounds = bounds

      prev_size = prev_size - {size, size}

      var par_idx = [int](sep_idx)
      for par_lvl = lvl-1, -1, -1 do
        par_idx = [int](par_idx/2)
        var par_level = tree[par_lvl]
        var par_sep = par_level[par_idx].node
        var par_bounds = block_bounds[{par_sep, par_sep}].bounds

        var child_bounds = rect2d{ {x = par_bounds.lo.x, y = bounds.lo.y},
                                   {x = par_bounds.hi.x, y = bounds.hi.y } }

        if debug then
          c.printf("block: %d %d bounds.lo: %d %d bounds.hi: %d %d\n", par_sep, sep,
                   child_bounds.lo.x, child_bounds.lo.y, child_bounds.hi.x, child_bounds.hi.y)
        end

        var color:int2d = {par_sep, sep}
        block_bounds[color].sep = color
        block_bounds[color].bounds = child_bounds
      end
    end
  end
end

__demand(__inline)
task partition_separator(block_color     : int2d,
                         block_bounds    : rect2d,
                         cluster_bounds  : region(ispace(int3d), ClusterBounds),
                         interval        : int,
                         clusters_region : region(ispace(int1d), ClusterIndex),
                         clusters_sep    : partition(disjoint, clusters_region, ispace(int1d)),
                         clusters_int    : partition(disjoint, clusters_region, ispace(int1d)),
                         clusters        : cross_product(clusters_sep, clusters_int),
                         debug           : bool)
where
  reads(clusters_region), reads writes(cluster_bounds)
do
  for i in cluster_bounds do
    cluster_bounds[i].bounds = rect2d{lo=int2d{0, 0}, hi=int2d{-1, -1}}
    cluster_bounds[i].cluster = int3d{-1, -1, -1}
  end

  var row_sep = block_color.x
  var row_cluster = clusters[row_sep]
  var row_cluster_size = row_cluster[interval].volume-1
  var row_lo = row_cluster[interval].bounds.lo

  var col_sep = block_color.y
  var col_cluster = clusters[col_sep]
  var col_cluster_size = col_cluster[interval].volume-1
  var col_lo = col_cluster[interval].bounds.lo

  var prev_lo = block_bounds.lo

  if debug then
    c.printf("\t\tPartitioning (%d, %d) Cluster: %d Rows: %d Cols: %d\n",
             row_sep, col_sep, interval, row_cluster_size, col_cluster_size)
  end

  for row = 0, row_cluster_size do
    var rowcluster = row_cluster[interval]
    var top = rowcluster[row+row_lo].idx
    var bottom = rowcluster[row+row_lo+1].idx

    for i = interval-1, -1, -1 do
      var rowcluster = row_cluster[i]
      var row_lo = rowcluster.bounds.lo
      top = rowcluster[top+row_lo].idx
      bottom = rowcluster[bottom+row_lo].idx
    end

    for col = 0, col_cluster_size do
      var colcluster = col_cluster[interval]
      var left = colcluster[col+col_lo].idx
      var right = colcluster[col+col_lo+1].idx

      for i = interval-1, -1, -1 do
        var colcluster = col_cluster[i]
        var col_lo = colcluster.bounds.lo
        left = colcluster[left+col_lo].idx
        right = colcluster[right+col_lo].idx
      end

      var part_size = int2d{x = bottom - top - 1, y = right - left - 1}
      var color:int3d = {x = row_sep, y = col_sep, z = row*col_cluster_size+col}
      var bounds = rect2d { prev_lo, prev_lo + part_size }
      var size = bounds.hi - bounds.lo + {1, 1}
      cluster_bounds[color].bounds = bounds
      cluster_bounds[color].cluster = color

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

__demand(__inline)
task partition_separators(depth                  : int,
                          tree_region            : region(ispace(int1d), TreeNode),
                          tree                   : partition(disjoint, tree_region, ispace(int1d)),
                          interval               : int,
                          clusters_region        : region(ispace(int1d), ClusterIndex),
                          clusters_sep           : partition(disjoint, clusters_region, ispace(int1d)),
                          clusters_int           : partition(disjoint, clusters_region, ispace(int1d)),
                          clusters               : cross_product(clusters_sep, clusters_int),
                          block_bounds           : region(ispace(int2d), BlockBounds),
                          cluster_bounds_region  : region(ispace(int3d), ClusterBounds),
                          cluster_bounds         : partition(disjoint, cluster_bounds_region, ispace(int2d)),
                          debug                  : bool)
where
  reads(tree_region, clusters_region, block_bounds), reads writes(cluster_bounds_region)
do
  for lvl = 0, depth+1 do
    var level = tree[lvl]

    for sep_idx in level.ispace do
      var row = level[sep_idx].node
      var block_color = int2d{row, row}
      var block_bound = block_bounds[block_color].bounds
      var cluster_bound = cluster_bounds[block_color]
      -- c.printf("Partitioning: %d %d Bounds: %d %d %d %d\n", row, row, block.lo.x, block.lo.y, block.hi.x, block.hi.y)
      partition_separator(block_color, block_bound, cluster_bound,
                          interval, clusters_region, clusters_sep, clusters_int, clusters,
                          debug)

      for clvl = lvl+1, depth+1 do
        var clevel = tree[clvl]
        for csep_idx = [int](sep_idx*math.pow(2, clvl-lvl)), [int]((sep_idx+1)*math.pow(2, clvl-lvl)) do
          var col = clevel[csep_idx].node
          -- c.printf("Partitioning: %d %d\n", row, col)
          var block_color = int2d{row, col}
          var block_bound = block_bounds[block_color].bounds
          var cluster_bound = cluster_bounds[block_color]
          partition_separator(block_color, block_bound, cluster_bound,
                              interval, clusters_region, clusters_sep, clusters_int, clusters,
                              debug)
        end
      end
    end
  end
end

__demand(__inline)
task contains(point  : int2d,
              domain : c.legion_domain_t)
  var dp = c.legion_domain_point_from_point_2d(point)
  return c.legion_domain_contains(domain, dp)
end

__demand(__leaf)
task fill_block(color           : int2d,
                entries_domain  : c.legion_domain_t,
                sparse_entries  : region(ispace(int2d), double),
                block           : region(ispace(int2d), double),
                row_dofs        : region(ispace(int1d), SepIndex),
                col_dofs        : region(ispace(int1d), SepIndex),
                clusters_region : region(ispace(int1d), ClusterIndex),
                clusters_sep    : partition(disjoint, clusters_region, ispace(int1d)),
                clusters_int    : partition(disjoint, clusters_region, ispace(int1d)),
                clusters        : cross_product(clusters_sep, clusters_int),
                filled_blocks   : region(ispace(int3d), Filled),
                debug           : bool)
where
  reads(sparse_entries, row_dofs, col_dofs, clusters_region), reads writes(block, filled_blocks)
do
  var row_sep = color.x
  var row_cluster = clusters[row_sep][0]
  var row_cluster_size = row_cluster.volume-1
  var row_lo = row_cluster.bounds.lo

  var col_sep = color.y
  var col_cluster = clusters[col_sep][0]
  var col_cluster_size = col_cluster.volume-1
  var col_lo = col_cluster.bounds.lo
  -- c.printf("Filling: %d %d From: %d %d To: %d %d\n",
  --          color.x, color.y, block.bounds.lo.x, block.bounds.lo.y, block.bounds.hi.x, block.bounds.hi.y)

  var nz = 0
  var block_idx = block.bounds.lo

  for col = 0, col_cluster_size do
    var left = col_cluster[col+col_lo].idx
    var right = col_cluster[col+col_lo+1].idx
    var col_bound = right - left

    for row = 0, row_cluster_size do
      var top = row_cluster[row+row_lo].idx
      var bottom = row_cluster[row+row_lo+1].idx
      var row_bound = bottom - top

      var z = row*col_cluster_size+col

      var nnz = 0
      -- c.printf("Bounds: %d %d %d row: %d %d col: %d %d\n", row_sep, col_sep, z, top, bottom, left, right)

      var cidx = int2d{0, 0}
      for i = top, bottom do
        for j = left, right do
          var idxi = row_dofs[block.bounds.lo.x+i].idx
          var idxj = col_dofs[block.bounds.lo.y+j].idx

          if idxj > idxi then
            var t = idxi
            idxi = idxj
            idxj = t
          end

          var eidx = int2d{idxi, idxj}
          var idx = block_idx + cidx
          if row_sep == col_sep and idx.y <= idx.x then
            -- c.printf("Block: %d %d %d Filling Diagonal: %d %d I: %d J: %d key: %lu Entry: %0.2f\n",
            --          row_sep, col_sep, z, idx.x, idx.y, idxi, idxj, eidx, entry)
            if contains(eidx, entries_domain) then
              block[idx] = sparse_entries[eidx]
              nnz += 1
            end
          elseif row_sep ~= col_sep then
            -- c.printf("Block: %d %d %d Filling Off-Diagonal: %d %d I: %d J: %d key: %lu Entry: %0.2f\n",
            --          row_sep, col_sep, z, idx.x, idx.y, idxi, idxj, eidx, entry)
            if contains(eidx, entries_domain) then
              block[idx] = sparse_entries[eidx]
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

__demand(__inline)
task merge_filled_clusters(allocated_blocks : ispace(int2d),
                           filled_clusters  : region(ispace(int3d), Filled),
                           interval         : int,
                           clusters_region  : region(ispace(int1d), ClusterIndex),
                           clusters_sep     : partition(disjoint, clusters_region, ispace(int1d)),
                           clusters_int     : partition(disjoint, clusters_region, ispace(int1d)),
                           clusters         : cross_product(clusters_sep, clusters_int))
where
  reads(clusters_region), reads writes(filled_clusters)
do
  var blocks = region(filled_clusters.ispace, Filled)
  copy(filled_clusters, blocks)

  fill(filled_clusters.nz, 0)
  fill(filled_clusters.filled, 1)
  fill(filled_clusters.sep, int2d{-1, -1})

  c.printf("Merging blocks from interval: %d\n", interval-1)
  for block in allocated_blocks do
    var row_sep = block.x
    var col_sep = block.y

    var row_cluster = clusters[row_sep][interval]
    var col_cluster = clusters[col_sep][interval]

    if row_cluster.volume ~= 0 and col_cluster.volume ~= 0 then
      var row_cluster_size = row_cluster.volume-1
      var row_lo = row_cluster.bounds.lo

      var col_cluster_size = col_cluster.volume-1
      var col_lo = col_cluster.bounds.lo

      var prev_cols = clusters[col_sep][interval-1].volume-1

      for row = 0, row_cluster_size do

        var top = row_cluster[row+row_lo].idx
        var bottom = row_cluster[row+row_lo+1].idx

        for col = 0, col_cluster_size do

          var left = col_cluster[col+col_lo].idx
          var right = col_cluster[col+col_lo+1].idx
          var new_id = int3d{row_sep, col_sep, row*col_cluster_size+col}
          -- c.printf("New Id: %d %d %d -> ", new_id.x, new_id.y, new_id.z)

          for i = top, bottom do
            for j = left, right do
              var old_id = int3d{row_sep, col_sep, i*prev_cols+j}
              -- c.printf("%d ", old_id.z)
              filled_clusters[new_id].nz = filled_clusters[new_id].nz + blocks[old_id].nz
              if filled_clusters[new_id].nz > 0 then
                filled_clusters[new_id].filled = 0
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

__demand(__leaf)
task find_index_space_3d(sepinfo         : mnd.SepInfo,
                         tree_region     : region(ispace(int1d), TreeNode),
                         tree            : partition(disjoint, tree_region, ispace(int1d)),
                         blocks          : region(ispace(int3d), int1d),
                         clusters_region : region(ispace(int1d), ClusterIndex),
                         clusters_sep    : partition(disjoint, clusters_region, ispace(int1d)),
                         clusters_int    : partition(disjoint, clusters_region, ispace(int1d)),
                         clusters        : cross_product(clusters_sep, clusters_int))
where
  reads(tree_region, clusters_region), reads writes (blocks)
do
  for lvl = 0, sepinfo.levels do
    var level = tree[lvl]
    for sep_idx in level.ispace do
      var row_sep = level[sep_idx].node
      var row_cluster_size = clusters[row_sep][0].volume-1

      for i = 0, row_cluster_size do
        for j = 0, row_cluster_size do
          -- c.printf("Block: %d %d %d\n", row_sep, row_sep, i*row_cluster_size+j)
          blocks[{row_sep, row_sep, i*row_cluster_size+j}] = 1
        end
      end

      for clvl = lvl+1, sepinfo.levels do
        var clevel = tree[clvl]
        for csep_idx = [int](sep_idx*math.pow(2, clvl-lvl)), [int]((sep_idx+1)*math.pow(2, clvl-lvl)) do
          var col_sep = clevel[csep_idx].node
          var col_cluster_size = clusters[col_sep][0].volume-1

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
task find_index_space_2d(sepinfo     : mnd.SepInfo,
                         tree_region : region(ispace(int1d), TreeNode),
                         tree        : partition(disjoint, tree_region, ispace(int1d)),
                         blocks      : region(ispace(int2d), int1d))
where
  reads(tree_region), reads writes (blocks)
do
  for lvl = 0, sepinfo.levels do
    var level = tree[lvl]
    for sep_idx in level.ispace do
      var row_sep = level[sep_idx].node

      -- c.printf("Sep: %d %d\n", row_sep, row_sep)
      blocks[{row_sep, row_sep}] = 1

      for clvl = lvl+1, sepinfo.levels do
        var clevel = tree[clvl]
        for csep_idx = [int](sep_idx*math.pow(2, clvl-lvl)), [int]((sep_idx+1)*math.pow(2, clvl-lvl)) do
          var col_sep = clevel[csep_idx].node

          -- c.printf("\tSep: %d %d\n", row_sep, col_sep)
          blocks[{row_sep, col_sep}] = 1
        end
      end
    end
  end
end

__demand(__leaf)
task fill_b(separator : region(ispace(int1d), SepIndex),
            Bentries  : region(ispace(int1d), double),
            B         : region(ispace(int1d), double))
where
  reads(separator, Bentries),
  writes(B)
do
  var j = 0
  for i in separator.ispace do
    var idxi = separator[i].idx
    B[B.bounds.lo + j] = Bentries[idxi]
    j += 1
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
local partition_cluster_bounds_by_sep = generate_partition(region(ispace(int3d), ClusterBounds))

task partition_cluster_bounds_by_cluster(cluster_bounds:region(ispace(int3d), ClusterBounds))
where
  reads writes(cluster_bounds.cluster)
do
  for i in cluster_bounds.ispace do
    cluster_bounds[i].cluster = i
  end

  var part = partition(cluster_bounds.cluster, cluster_bounds.ispace)
  return part
end

__demand(__leaf)
task fused_dpotrf(rA             : region(ispace(int2d), double),
                  filled_rA      : region(ispace(ptr), Filled))
where
  reads writes(rA), reads(filled_rA)
do
  for i in filled_rA.ispace do
    var a = filled_rA[i]
    var color = int3d{a.sep.x, a.sep.y, a.cluster}
    var rectA = a.bounds
    var size:int2d = rectA.hi - rectA.lo + {1, 1}
    -- c.printf("POTRF A=(%d %d %d) Size: %d %d\n", color.x, color.y, color.z, size.x, size.y)
    dpotrf_terra(rectA, size.x, __physical(rA)[0], __fields(rA)[0])
  end
end

__demand(__leaf)
task fused_dtrsm(rA                : region(ispace(int2d), double),
                 rB                : region(ispace(int2d), double),
                 filled_rA         : region(ispace(ptr), Filled),
                 filled_rB         : region(ispace(ptr), Filled))
where
  reads(rA, filled_rA, filled_rB), reads writes(rB)
do
  for i in filled_rA.ispace do
    var a = filled_rA[i]
    var Acolor = int3d{a.sep.x, a.sep.y, a.cluster}
    var rectA = a.bounds
    var sizeA = rectA.hi - rectA.lo + {1, 1}
    for j in filled_rB.ispace do
      var b = filled_rB[j]
      var Bcolor = int3d{b.sep.x, b.sep.y, b.cluster}
      var rectB = b.bounds
      var size:int2d = rectB.hi - rectB.lo + {1, 1}
      -- c.printf("TRSM A=(%d %d %d) Size: %d %d B=(%d %d %d) Size: %d %d\n",
      --          Acolor.x, Acolor.y, Acolor.z, sizeA.x, sizeA.y,
      --          Bcolor.x, Bcolor.y, Bcolor.z, size.x, size.y)
      dtrsm_terra(rectA, rectB, size.x, size.y,
                  __physical(rA)[0], __fields(rA)[0],
                  __physical(rB)[0], __fields(rB)[0])
    end
  end
end

__demand(__leaf)
task fused_dsyrk(rA                : region(ispace(int2d), double),
                 rB                : region(ispace(int2d), double),
                 rC                : region(ispace(int2d), double),
                 filled_rA         : region(ispace(ptr), Filled),
                 filled_rB         : region(ispace(ptr), Filled),
                 filled_rC         : region(ispace(ptr), Filled),
                 col_cluster_size  : int)
where
  reads(rA, rB, filled_rA, filled_rB, filled_rC),
  reads writes(rC)
do
  for i in filled_rA.ispace do
    var a = filled_rA[i]
    var Acolor = int3d{a.sep.x, a.sep.y, a.cluster}
    var row = Acolor.z
    var rectA = a.bounds
    var sizeA:int2d = rectA.hi - rectA.lo + {1, 1}

    for j in filled_rB.ispace do
      var b = filled_rB[j]
      var Bcolor = int3d{b.sep.x, b.sep.y, b.cluster}
      var col = Bcolor.z

      var Ccolor = int3d{Acolor.x, Bcolor.x, row*col_cluster_size+col}
      var rectC = rect2d{lo=int2d{0, 0}, hi={int2d{-1, -1}}}

      for k in filled_rC.ispace do
        var p = filled_rC[k]
        var clust = int3d{p.sep.x, p.sep.y, p.cluster}
        if clust == Ccolor then
          rectC = p.bounds
          break
        end
      end

      var vol = c.legion_domain_get_volume(c.legion_domain_from_rect_2d(rectC))
      regentlib.assert(vol ~= 0, "Empty GEMM cluster.")

      var sizeC:int2d = rectC.hi - rectC.lo + {1, 1}

      if col < row then
        var rectB = b.bounds

        var m = sizeC.x
        var n = sizeC.y
        var k = sizeA.y

        dgemm_terra(rectA, rectB, rectC, m, n, k,
                    __physical(rA)[0], __fields(rA)[0],
                    __physical(rB)[0], __fields(rB)[0],
                    __physical(rC)[0], __fields(rC)[0])

      elseif col == row then
        var n = sizeC.x
        var k = sizeA.y

        dsyrk_terra(rectA, rectC, n, k,
                    __physical(rA)[0], __fields(rA)[0],
                    __physical(rC)[0], __fields(rC)[0])
      end
    end
  end
end

__demand(__leaf)
task fused_dgemm(rA                : region(ispace(int2d), double),
                 rB                : region(ispace(int2d), double),
                 rC                : region(ispace(int2d), double),
                 filled_rA         : region(ispace(ptr), Filled),
                 filled_rB         : region(ispace(ptr), Filled),
                 filled_rC         : region(ispace(ptr), Filled),
                 col_cluster_size  : int)
where
  reads(rA, rB, filled_rA, filled_rB, filled_rC),
  reads writes(rC)
do
  for i in filled_rA.ispace do
    var a = filled_rA[i]
    var Acolor = int3d{a.sep.x, a.sep.y, a.cluster}
    var row = Acolor.z
    var rectA = a.bounds
    var sizeA:int2d = rectA.hi - rectA.lo + {1, 1}

    for j in filled_rB do
      var b = filled_rB[j]
      var Bcolor = int3d{b.sep.x, b.sep.y, b.cluster}
      var col = Bcolor.z

      var Ccolor = int3d{Acolor.x, Bcolor.x, row*col_cluster_size+col}
      var rectC = rect2d{lo=int2d{0, 0}, hi={int2d{-1, -1}}}

      for k in filled_rC.ispace do
        var p = filled_rC[k]
        var clust = int3d{p.sep.x, p.sep.y, p.cluster}
        if clust == Ccolor then
          rectC = p.bounds
          break
        end
      end

      var vol = c.legion_domain_get_volume(c.legion_domain_from_rect_2d(rectC))
      regentlib.assert(vol ~= 0, "Empty GEMM cluster.")

      var sizeC:int2d = rectC.hi - rectC.lo + {1, 1}

      var rectB = b.bounds

      var m = sizeC.x
      var n = sizeC.y
      var k = sizeA.y

      dgemm_terra(rectA, rectB, rectC, m, n, k,
                  __physical(rA)[0], __fields(rA)[0],
                  __physical(rB)[0], __fields(rB)[0],
                  __physical(rC)[0], __fields(rC)[0])
    end
  end
end

__demand(__inline)
task inline_fused_dpotrf(rA             : region(ispace(int2d), double),
                         cluster_bounds : region(ispace(int3d), ClusterBounds),
                         filled         : region(ispace(ptr), Filled))
where
  reads writes(rA), reads(filled, cluster_bounds)
do
  for i in filled.ispace do
    var p = filled[i]
    var color = int3d{p.sep.x, p.sep.y, p.cluster}
    var rectA = cluster_bounds[color]
    var size:int2d = rectA.bounds.hi - rectA.bounds.lo + {1, 1}
    dpotrf_terra(rectA.bounds, size.x, __physical(rA)[0], __fields(rA)[0])
  end
end

__demand(__inline)
task inline_fused_dtrsm(rA                : region(ispace(int2d), double),
                        rB                : region(ispace(int2d), double),
                        cluster_bounds_rA : region(ispace(int3d), ClusterBounds),
                        cluster_bounds_rB : region(ispace(int3d), ClusterBounds),
                        filled_rA         : region(ispace(ptr), Filled),
                        filled_rB         : region(ispace(ptr), Filled))
where
  reads(rA, cluster_bounds_rA, cluster_bounds_rB, filled_rA, filled_rB), reads writes(rB)
do
  for i in filled_rA.ispace do
    var a = filled_rA[i]
    var Acolor = int3d{a.sep.x, a.sep.y, a.cluster}
    var rectA = cluster_bounds_rA[Acolor].bounds
    for j in filled_rB.ispace do
      var b = filled_rB[j]
      var Bcolor = int3d{b.sep.x, b.sep.y, b.cluster}
      var rectB = cluster_bounds_rB[Bcolor].bounds
      var size:int2d = rectB.hi - rectB.lo + {1, 1}
      dtrsm_terra(rectA, rectB, size.x, size.y,
                  __physical(rA)[0], __fields(rA)[0],
                  __physical(rB)[0], __fields(rB)[0])
    end
  end
end

__demand(__inline)
task inline_fused_dsyrk(rA                : region(ispace(int2d), double),
                        rB                : region(ispace(int2d), double),
                        rC                : region(ispace(int2d), double),
                        cluster_bounds_rA : region(ispace(int3d), ClusterBounds),
                        cluster_bounds_rB : region(ispace(int3d), ClusterBounds),
                        cluster_bounds_rC : region(ispace(int3d), ClusterBounds),
                        filled_rA         : region(ispace(ptr), Filled),
                        filled_rB         : region(ispace(ptr), Filled),
                        col_cluster_size  : int)
where
  reads(rA, rB, cluster_bounds_rA, cluster_bounds_rB, cluster_bounds_rC, filled_rA, filled_rB),
  reads writes(rC)
do
  for i in filled_rA.ispace do
    var a = filled_rA[i]
    var Acolor = int3d{a.sep.x, a.sep.y, a.cluster}
    var row = Acolor.z
    var rectA = cluster_bounds_rA[Acolor].bounds
    var sizeA:int2d = rectA.hi - rectA.lo + {1, 1}

    for j in filled_rB.ispace do
      var b = filled_rB[j]
      var Bcolor = int3d{b.sep.x, b.sep.y, b.cluster}
      var col = Bcolor.z

      var Ccolor = int3d{Acolor.x, Bcolor.x, row*col_cluster_size+col}
      var rectC = cluster_bounds_rC[Ccolor].bounds
      var sizeC:int2d = rectC.hi - rectC.lo + {1, 1}

      if col < row then
        var rectB = cluster_bounds_rB[Bcolor].bounds

        var m = sizeC.x
        var n = sizeC.y
        var k = sizeA.y

        dgemm_terra(rectA, rectB, rectC, m, n, k,
                    __physical(rA)[0], __fields(rA)[0],
                    __physical(rB)[0], __fields(rB)[0],
                    __physical(rC)[0], __fields(rC)[0])

      elseif col == row then
        var n = sizeC.x
        var k = sizeA.y

        dsyrk_terra(rectA, rectC, n, k,
                    __physical(rA)[0], __fields(rA)[0],
                    __physical(rC)[0], __fields(rC)[0])
      end
    end
  end
end

__demand(__inline)
task inline_fused_dgemm(rA                : region(ispace(int2d), double),
                        rB                : region(ispace(int2d), double),
                        rC                : region(ispace(int2d), double),
                        cluster_bounds_rA : region(ispace(int3d), ClusterBounds),
                        cluster_bounds_rB : region(ispace(int3d), ClusterBounds),
                        cluster_bounds_rC : region(ispace(int3d), ClusterBounds),
                        filled_rA         : region(ispace(ptr), Filled),
                        filled_rB         : region(ispace(ptr), Filled),
                        col_cluster_size  : int)
where
  reads(rA, rB, cluster_bounds_rA, cluster_bounds_rB, cluster_bounds_rC, filled_rA, filled_rB),
  reads writes(rC)
do
  for i in filled_rA.ispace do
    var a = filled_rA[i]
    var Acolor = int3d{a.sep.x, a.sep.y, a.cluster}
    var row = Acolor.z
    var rectA = cluster_bounds_rA[Acolor].bounds
    var sizeA:int2d = rectA.hi - rectA.lo + {1, 1}

    for j in filled_rB do
      var b = filled_rB[j]
      var Bcolor = int3d{b.sep.x, b.sep.y, b.cluster}
      var col = Bcolor.z

      var Ccolor = int3d{Acolor.x, Bcolor.x, row*col_cluster_size+col}
      var rectC = cluster_bounds_rC[Ccolor].bounds
      var sizeC:int2d = rectC.hi - rectC.lo + {1, 1}

      var rectB = cluster_bounds_rB[Bcolor].bounds

      var m = sizeC.x
      var n = sizeC.y
      var k = sizeA.y

      dgemm_terra(rectA, rectB, rectC, m, n, k,
                  __physical(rA)[0], __fields(rA)[0],
                  __physical(rB)[0], __fields(rB)[0],
                  __physical(rC)[0], __fields(rC)[0])
    end
  end
end

__demand(__leaf)
task build_separator_tree(tree_region:region(ispace(int1d), TreeNode), sepinfo:mnd.SepInfo)
where
  reads writes(tree_region)
do
  var num_separators = sepinfo.num_separators
  var i = int1d{1}
  for level = 0, sepinfo.levels do
    for elem = 0, [int](math.pow(2, level)) do
      tree_region[i].node = num_separators
      tree_region[i].level = int1d{level}
      num_separators -= 1
      i += 1
    end
  end
end

task read_matrix(matrix_file    : regentlib.string,
                 NZ             : int,
                 entries_region : region(ispace(int2d), MatrixEntry))
where
  writes(entries_region)
do
  mnd.read_matrix(matrix_file, NZ,
                  __runtime(), __context(),
                  __raw(entries_region.ispace), __physical(entries_region), __fields(entries_region))
end

__demand(__leaf)
task copy_sparse_entries(entries        : region(ispace(int2d), MatrixEntry),
                         sparse_entries : region(ispace(int2d), double))
where
  reads(entries.val),
  writes(sparse_entries)
do
  var idx = int2d{0, 0}
  for i in sparse_entries.ispace do
    sparse_entries[i] = entries[idx].val
    idx += int2d{1 ,0}
  end
end

task read_separators(separator_file    : regentlib.string,
                     M                 : int,
                     separators_region : region(ispace(int1d), SepIndex))
where
  writes(separators_region)
do
  return mnd.read_separators(separator_file, M,
                             __runtime(), __context(),
                             __raw(separators_region.ispace), __physical(separators_region), __fields(separators_region))
end

task read_clusters(clusters_file   : regentlib.string,
                   M               : int,
                   clusters_region : region(ispace(int1d), ClusterIndex))
where
  writes(clusters_region)
do
  return mnd.read_clusters(clusters_file, M,
                           __runtime(), __context(),
                           __raw(clusters_region.ispace), __physical(clusters_region), __fields(clusters_region))
end

task read_vector(vector_file   : regentlib.string,
                 N             : int,
                 vector_region : region(ispace(int1d), double))
where
  writes(vector_region)
do
  mnd.read_vector(vector_file, N,
                  __runtime(), __context(),
                  __raw(vector_region.ispace), __physical(vector_region), __fields(vector_region))
end

task compute_filled_clusters(levels                  : int,
                             tree_region             : region(ispace(int1d), TreeNode),
                             tree                    : partition(disjoint, tree_region, ispace(int1d)),
                             clusters_region         : region(ispace(int1d), ClusterIndex),
                             clusters_sep            : partition(disjoint, clusters_region, ispace(int1d)),
                             clusters_int            : partition(disjoint, clusters_region, ispace(int1d)),
                             clusters                : cross_product(clusters_sep, clusters_int),
                             block_bounds            : region(ispace(int2d), BlockBounds),
                             cluster_bounds_region   : region(ispace(int3d), ClusterBounds),
                             cluster_bounds          : partition(disjoint, cluster_bounds_region, ispace(int2d)),
                             allocated_blocks_ispace : ispace(int2d),
                             filled_clusters_region  : region(ispace(int3d), Filled),
                             filled_clusters_region2 : region(ispace(ptr), Filled),
                             debug                   : bool)
where
  reads(tree_region, clusters_region, block_bounds),
  reads writes(filled_clusters_region, filled_clusters_region2, cluster_bounds_region)
do
  var interval = 0
  var i = 0

  for lvl = levels-1, -1, -1 do
    partition_separators(lvl, tree_region, tree,
                         interval, clusters_region, clusters_sep, clusters_int, clusters,
                         block_bounds, cluster_bounds_region, cluster_bounds,
                         debug)


    var level = tree[lvl]

    for sep_idx in level.ispace do
      var sep = level[sep_idx].node
      var par_idx = [int](sep_idx)

      for par_lvl = lvl-1, -1, -1 do
        par_idx = [int](par_idx/2)
        var par_level = tree[par_lvl]
        var par_sep = par_level[par_idx].node

        var grandpar_idx = [int](par_idx)
        for grandpar_lvl = par_lvl, -1, -1 do
          var grandpar_level = tree[grandpar_lvl]
          var grandpar_sep = grandpar_level[grandpar_idx].node

          var col_cluster_size = clusters[par_sep][interval].volume-1
          var A_clusters = (clusters[grandpar_sep][interval].volume-1)*(clusters[sep][interval].volume-1)
          var B_clusters = (clusters[par_sep][interval].volume-1)*(clusters[sep][interval].volume-1)

          for i = 0, A_clusters do
            var A_color = int3d{grandpar_sep, sep, i}

            if filled_clusters_region[A_color].nz > 0 then
              for j = 0, B_clusters do
                var B_color = int3d{par_sep, sep, j}

                if filled_clusters_region[B_color].nz > 0 then
                  var C_color = int3d{grandpar_sep, par_sep, i*col_cluster_size+j}

                  if filled_clusters_region[C_color].nz <= 0 then
                    -- c.printf("Level: %d Interval: %d Block: %d %d %d Filled.\n",
                    --          lvl, interval,
                    --          C_color.x, C_color.y, C_color.z)
                    filled_clusters_region[C_color].nz = filled_clusters_region[A_color].nz * filled_clusters_region[B_color].nz
                    filled_clusters_region[C_color].filled = 0
                  end
                end
              end
            end
          end

          grandpar_idx = [int](grandpar_idx/2)
        end
      end
    end

    for j in filled_clusters_region.ispace do
      var sep = int2d{j.x, j.y}
      filled_clusters_region2[i].nz = filled_clusters_region[j].nz
      filled_clusters_region2[i].filled = filled_clusters_region[j].filled
      filled_clusters_region2[i].sep = sep
      filled_clusters_region2[i].interval = interval
      filled_clusters_region2[i].cluster = j.z
      var clust = cluster_bounds[sep]
      filled_clusters_region2[i].bounds = clust[j].bounds
      -- c.printf("Level: %d Interval: %d Cluster: %d %d %d Filled: %d\n",
      --          lvl, interval, j.x, j.y, j.z, filled_clusters_region[j].filled)
      i += 1
    end

    interval += 1
    if interval < levels then
      merge_filled_clusters(allocated_blocks_ispace, filled_clusters_region, interval,
                            clusters_region, clusters_sep, clusters_int, clusters)
    end
  end
end

__demand(__inline)
task factor(lvl                    : int,
            tree_region            : region(ispace(int1d), TreeNode),
            tree                   : partition(disjoint, tree_region, ispace(int1d)),
            banner                 : MMatBanner,
            mat                    : region(ispace(int2d), double),
            mat_part               : partition(disjoint, mat, ispace(int2d)),
            filled_clusters_region : region(ispace(ptr), Filled),
            filled_clusters_sep    : partition(disjoint, filled_clusters_region, ispace(int2d)),
            filled_clusters_int    : partition(disjoint, filled_clusters_region, ispace(int1d)),
            filled_clusters        : cross_product(filled_clusters_sep, filled_clusters_int),
            cluster_bounds_region  : region(ispace(int3d), ClusterBounds),
            cluster_bounds         : partition(disjoint, cluster_bounds_region, ispace(int2d)),
            interval               : int,
            clusters_region        : region(ispace(int1d), ClusterIndex),
            clusters_sep           : partition(disjoint, clusters_region, ispace(int1d)),
            clusters_int           : partition(disjoint, clusters_region, ispace(int1d)),
            clusters               : cross_product(clusters_sep, clusters_int),
            debug_path             : regentlib.string,
            debug                  : bool)
where
  reads(tree_region, cluster_bounds_region, clusters_region, filled_clusters_region), reads writes(mat)
do
  var level = tree[lvl]

  for sep_idx in level.ispace do
    var sep = level[sep_idx].node
    var pivot_color = int2d{sep, sep}
    var filled_pivot = filled_clusters[pivot_color][interval]

    inline_fused_dpotrf(mat_part[pivot_color], cluster_bounds[pivot_color], filled_pivot)

    if debug then
      var pivot = mat_part[pivot_color]
      var sizeA = pivot.bounds.hi - pivot.bounds.lo + {1, 1}

      c.printf("Level: %d POTRF A=(%d, %d)\nSize: %dx%d Lo: %d %d Hi: %d %d\n\n",
               lvl, sep, sep, sizeA.x, sizeA.y,
               pivot.bounds.lo.x, pivot.bounds.lo.y, pivot.bounds.hi.x, pivot.bounds.hi.y)

      write_blocks(mat, mat_part, lvl, pivot_color, int2d{0, 0}, int2d{0, 0}, "POTRF", banner, debug_path)
    end
  end

  for sep_idx in level.ispace do
    var sep = level[sep_idx].node
    var pivot_color = int2d{sep, sep}
    var filled_pivot = filled_clusters[pivot_color][interval]

    var par_idx = [int](sep_idx)
    for par_lvl = lvl-1, -1, -1 do

      par_idx = [int](par_idx/2)
      var par_level = tree[par_lvl]
      var par_sep = par_level[par_idx].node
      var off_diag_color = int2d{par_sep, sep}
      var filled_off_diag = filled_clusters[off_diag_color][interval]

      inline_fused_dtrsm(mat_part[pivot_color], mat_part[off_diag_color],
                         cluster_bounds[pivot_color], cluster_bounds[off_diag_color],
                         filled_pivot, filled_off_diag)

      if debug then
        var pivot = mat_part[pivot_color]
        var sizeA = pivot.bounds.hi - pivot.bounds.lo + {1, 1}
        var off_diag = mat_part[off_diag_color]
        var sizeB = off_diag.bounds.hi - off_diag.bounds.lo + {1, 1}

        c.printf("\tLevel: %d TRSM A=(%d, %d) B=(%d, %d)\n\tSizeA: %dx%d Lo: %d %d Hi: %d %d SizeB: %dx%d Lo: %d %d Hi: %d %d\n\n",
                 par_lvl, sep, sep, par_sep, sep,
                 sizeA.x, sizeA.y, pivot.bounds.lo.x, pivot.bounds.lo.y, pivot.bounds.hi.x, pivot.bounds.hi.y,
                 sizeB.x, sizeB.y, off_diag.bounds.lo.x, off_diag.bounds.lo.y, off_diag.bounds.hi.x, off_diag.bounds.hi.y)

        write_blocks(mat, mat_part, par_lvl, pivot_color, off_diag_color, int2d{0, 0}, "TRSM", banner, debug_path)
      end
    end
  end

  for sep_idx in level.ispace do
    var sep = level[sep_idx].node
    var par_idx = [int](sep_idx)

    for par_lvl = lvl-1, -1, -1 do
      par_idx = [int](par_idx/2)
      var par_level = tree[par_lvl]
      var par_sep = par_level[par_idx].node

      var grandpar_idx = [int](par_idx)
      for grandpar_lvl = par_lvl, -1, -1 do
        var grandpar_level = tree[grandpar_lvl]
        var grandpar_sep = grandpar_level[grandpar_idx].node

        var A_color = int2d{grandpar_sep, sep}
        var filled_A = filled_clusters[A_color][interval]
        var B_color = int2d{par_sep, sep}
        var filled_B = filled_clusters[B_color][interval]
        var C_color = int2d{grandpar_sep, par_sep}

        var col_cluster_size = clusters[par_sep][interval].volume-1

        if grandpar_sep == par_sep then
          inline_fused_dsyrk(mat_part[A_color], mat_part[B_color], mat_part[C_color],
                             cluster_bounds[A_color], cluster_bounds[B_color], cluster_bounds[C_color],
                             filled_A, filled_B, col_cluster_size)
        else
          inline_fused_dgemm(mat_part[A_color], mat_part[B_color], mat_part[C_color],
                             cluster_bounds[A_color], cluster_bounds[B_color], cluster_bounds[C_color],
                             filled_A, filled_B, col_cluster_size)
        end

        if debug then
          var A = mat_part[A_color]
          var B = mat_part[B_color]
          var C = mat_part[C_color]
          var sizeA = A.bounds.hi - A.bounds.lo + {1, 1}
          var sizeB = B.bounds.hi - B.bounds.lo + {1, 1}
          var sizeC = C.bounds.hi - C.bounds.lo + {1, 1}

          c.printf("\tLevel: %d GEMM A=(%d, %d) B=(%d, %d) C=(%d, %d)\n\tSizeA: %dx%d Lo: %d %d Hi: %d %d SizeB: %dx%d Lo: %d %d Hi: %d %d SizeC: %dx%d Lo: %d %d Hi: %d %d\n\n",
                   grandpar_lvl,
                   grandpar_sep, sep,
                   par_sep, sep,
                   grandpar_sep, par_sep,
                   sizeA.x, sizeA.y, A.bounds.lo.x, A.bounds.lo.y, A.bounds.hi.x, A.bounds.hi.y,
                   sizeB.x, sizeB.y, B.bounds.lo.x, B.bounds.lo.y, B.bounds.hi.x, B.bounds.hi.y,
                   sizeC.x, sizeC.y, C.bounds.lo.x, C.bounds.lo.y, C.bounds.hi.x, C.bounds.hi.y)

          write_blocks(mat, mat_part, grandpar_lvl, A_color, B_color, C_color, "GEMM", banner, debug_path)
        end
        grandpar_idx = [int](grandpar_idx/2)
      end
    end
  end
end

-- __demand(__inner)
task main()
  cblas.openblas_set_num_threads(1)

  var args = c.legion_runtime_get_input_args()

  var matrix_file = ""
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
      matrix_file = args.argv[i+1]
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

  var banner = read_matrix_banner(matrix_file)
  c.printf("M: %d N: %d nz: %d typecode: %s\n", banner.M, banner.N, banner.NZ, banner.typecode)

  var separators_region = region(ispace(int1d, banner.M), SepIndex)
  var sepinfo = read_separators(separator_file, banner.M, separators_region)
  var levels = sepinfo.levels
  var num_separators = sepinfo.num_separators
  var separators = partition(separators_region.sep, ispace(int1d, num_separators, 1))

  var tree_region = region(ispace(int1d, num_separators, 1), TreeNode)
  fill(tree_region.node, -1)
  fill(tree_region.level, -1)

  build_separator_tree(tree_region, sepinfo)
  var tree = partition(tree_region.level, ispace(int1d, sepinfo.levels))

  var clusters_region = region(ispace(int1d, num_separators*num_separators, 1), ClusterIndex)
  var max_int_size = read_clusters(clusters_file, banner.M, clusters_region)
  var clusters_sep = partition(clusters_region.sep, ispace(int1d, num_separators, 1))
  var clusters_int = partition(clusters_region.interval, ispace(int1d, levels))
  var clusters = cross_product(clusters_sep, clusters_int)

  c.printf("levels: %d\n", levels)
  c.printf("separators: %d\n", num_separators)
  c.printf("Max Interval Size: %d\n", max_int_size)

  var mat = region(ispace(int2d, {x = banner.M, y = banner.N}), double)
  fill(mat, 0)

  var nonzero_entries = region(ispace(int2d, {banner.NZ, 1}), MatrixEntry)
  fill(nonzero_entries.idx, int2d{-1, -1})
  fill(nonzero_entries.val, 0)
  read_matrix(matrix_file, banner.NZ, nonzero_entries)
  var nz_part = partition(equal, nonzero_entries, ispace(int2d, {1, 1}))
  var sparse_part = image(mat, nz_part, nonzero_entries.idx)

  var sparse_entry_ispace = sparse_part[{0, 0}].ispace
  var sparse_entries = region(sparse_entry_ispace, double)
  fill(sparse_entries, 0)
  copy_sparse_entries(nonzero_entries, sparse_entries)
  var entries_domain = c.legion_domain_from_index_space(__runtime(), __raw(sparse_entry_ispace))

  var blocks = region(ispace(int2d, {num_separators, num_separators}, {1, 1}), int1d)
  fill(blocks, 0)
  find_index_space_2d(sepinfo, tree_region, tree, blocks)
  var allocated_blocks = partition(blocks, ispace(int1d, 2))
  var allocated_blocks_ispace = allocated_blocks[1].ispace

  var block_bounds = region(allocated_blocks_ispace, BlockBounds)
  fill(block_bounds.sep, int2d{-1, -1})
  fill(block_bounds.bounds, rect2d{lo=int2d{0, 0}, hi=int2d{-1, -1}})

  partition_matrix(sepinfo, separators_region, separators,
                   tree_region, tree,
                   banner, mat, block_bounds, debug)

  var block_bounds_part = partition(block_bounds.sep, allocated_blocks_ispace)
  var mat_part = partition_mat_by_image_range(mat, block_bounds, block_bounds_part)

  var nz = 0
  var interval = 0

  var cluster_blocks = region(ispace(int3d, {num_separators, num_separators, max_int_size*max_int_size}, {1, 1, 0}), int1d)
  fill(cluster_blocks, 0)
  find_index_space_3d(sepinfo, tree_region, tree, cluster_blocks,
                      clusters_region, clusters_sep, clusters_int, clusters)
  var allocated_clusters = partition(cluster_blocks, ispace(int1d, 2))
  var allocated_clusters_ispace = allocated_clusters[1].ispace

  c.printf("Blocks ispace: %d\n", allocated_blocks_ispace.volume)
  c.printf("Clusters ispace: %d\n", allocated_clusters_ispace.volume)

  var filled_clusters_region = region(allocated_clusters_ispace, Filled)
  fill(filled_clusters_region.nz, -1)
  fill(filled_clusters_region.filled, 1)
  fill(filled_clusters_region.sep, int2d{-1, -1})
  fill(filled_clusters_region.interval, -1)
  fill(filled_clusters_region.cluster, -1)
  fill(filled_clusters_region.bounds, rect2d{lo=int2d{0, 0}, hi=int2d{-1, -1}})

  var filled_clusters_by_sep = partition_filled_blocks_by_sep(filled_clusters_region, num_separators)

  var cluster_bounds_region = region(allocated_clusters_ispace, ClusterBounds)
  fill(cluster_bounds_region.sep, int2d{-1, -1})
  fill(cluster_bounds_region.cluster, int3d{-1, -1, -1})
  fill(cluster_bounds_region.bounds, rect2d{lo=int2d{0, 0}, hi=int2d{-1, -1}})

  var cluster_bounds = partition_cluster_bounds_by_sep(cluster_bounds_region, num_separators)

  for color in mat_part.colors do
    var block = mat_part[color]
    -- c.printf("Filling: %d %d\n", color.x, color.y)
    var fpblock = filled_clusters_by_sep[color]
    fill(block, 0)
    var row_dofs = separators[color.x]
    var col_dofs = separators[color.y]
    fill_block(color, entries_domain, sparse_entries, block, row_dofs, col_dofs, clusters_region, clusters_sep, clusters_int, clusters, fpblock, debug)
    -- nz += fill_block(color, sparse_entries, block, row_dofs, col_dofs, clusters_region, clusters_sep, clusters_int, clusters, fpblock, debug)
    -- regentlib.assert(nz <= banner.NZ, "Mismatch in number of entries.")
  end

  -- c.printf("Filled: %d Expected: %d\n", nz, banner.NZ)

  if c.strcmp(permuted_matrix_file, '') ~= 0 then
    write_matrix(mat, mat_part, permuted_matrix_file, banner)
  end

  -- regentlib.assert(nz == banner.NZ, "Mismatch in number of entries.")

  var filled_clusters_region2 = region(ispace(ptr, allocated_clusters_ispace.volume*levels), Filled)
  fill(filled_clusters_region2.nz, -1)
  fill(filled_clusters_region2.filled, 1)
  fill(filled_clusters_region2.sep, int2d{-1, -1})
  fill(filled_clusters_region2.interval, -1)
  fill(filled_clusters_region2.cluster, -1)
  fill(filled_clusters_region2.bounds, rect2d{lo=int2d{0, 0}, hi=int2d{-1, -1}})

  compute_filled_clusters(levels, tree_region, tree,
                          clusters_region, clusters_sep, clusters_int, clusters,
                          block_bounds, cluster_bounds_region, cluster_bounds,
                          allocated_blocks_ispace, filled_clusters_region, filled_clusters_region2, debug)

  var filled_part = partition(filled_clusters_region2.filled, ispace(int1d, 2))
  var filled_clusters_filled = filled_part[0]
  var filled_clusters_sep = partition(filled_clusters_filled.sep, allocated_blocks_ispace)
  var filled_clusters_intervals = partition(filled_clusters_filled.interval, ispace(int1d, levels))
  var filled_clusters = cross_product(filled_clusters_sep, filled_clusters_intervals)

  for lvl = levels-1, -1, -1 do

    c.printf("Factoring Level: %d\n", lvl)

    if lvl < 0 then
      factor(lvl, tree_region, tree,
             banner, mat, mat_part,
             filled_clusters_filled, filled_clusters_sep, filled_clusters_intervals, filled_clusters,
             cluster_bounds_region, cluster_bounds,
             interval, clusters_region, clusters_sep, clusters_int, clusters,
             debug_path, debug)
    else
      var level = tree[lvl]

      __demand(__parallel)
      for sep_idx in level.ispace do
        var sep = level[sep_idx].node
        var pivot_color = int2d{sep, sep}
        var filled_pivot = filled_clusters[pivot_color][interval]
        fused_dpotrf(mat_part[pivot_color], filled_pivot)

        if debug then
          var pivot = mat_part[pivot_color]
          var sizeA = pivot.bounds.hi - pivot.bounds.lo + {1, 1}

          c.printf("Level: %d POTRF A=(%d, %d)\nSize: %dx%d Lo: %d %d Hi: %d %d\n\n",
                   lvl, sep, sep, sizeA.x, sizeA.y,
                   pivot.bounds.lo.x, pivot.bounds.lo.y, pivot.bounds.hi.x, pivot.bounds.hi.y)

          write_blocks(mat, mat_part, lvl, pivot_color, int2d{0, 0}, int2d{0, 0}, "POTRF", banner, debug_path)
        end
      end

      __demand(__parallel)
      for sep_idx in level.ispace do
        var sep = level[sep_idx].node
        var pivot_color = int2d{sep, sep}
        var filled_pivot = filled_clusters[pivot_color][interval]

        var par_idx = [int](sep_idx)
        for par_lvl = lvl-1, -1, -1 do

          par_idx = [int](par_idx/2)
          var par_level = tree[par_lvl]
          var par_sep = par_level[par_idx].node
          var off_diag_color = int2d{par_sep, sep}
          var filled_off_diag = filled_clusters[off_diag_color][interval]

          fused_dtrsm(mat_part[pivot_color], mat_part[off_diag_color],
                      filled_pivot, filled_off_diag)

          if debug then
            var pivot = mat_part[pivot_color]
            var sizeA = pivot.bounds.hi - pivot.bounds.lo + {1, 1}
            var off_diag = mat_part[off_diag_color]
            var sizeB = off_diag.bounds.hi - off_diag.bounds.lo + {1, 1}

            c.printf("\tLevel: %d TRSM A=(%d, %d) B=(%d, %d)\n\tSizeA: %dx%d Lo: %d %d Hi: %d %d SizeB: %dx%d Lo: %d %d Hi: %d %d\n\n",
                     par_lvl, sep, sep, par_sep, sep,
                     sizeA.x, sizeA.y, pivot.bounds.lo.x, pivot.bounds.lo.y, pivot.bounds.hi.x, pivot.bounds.hi.y,
                     sizeB.x, sizeB.y, off_diag.bounds.lo.x, off_diag.bounds.lo.y, off_diag.bounds.hi.x, off_diag.bounds.hi.y)

            write_blocks(mat, mat_part, par_lvl, pivot_color, off_diag_color, int2d{0, 0}, "TRSM", banner, debug_path)
          end
        end
      end

      __demand(__parallel)
      for sep_idx in level.ispace do
        var sep = level[sep_idx].node
        var par_idx = [int](sep_idx)

        for par_lvl = lvl-1, -1, -1 do
          par_idx = [int](par_idx/2)
          var par_level = tree[par_lvl]
          var par_sep = par_level[par_idx].node

          var grandpar_idx = [int](par_idx)
          for grandpar_lvl = par_lvl, -1, -1 do
            var grandpar_level = tree[grandpar_lvl]
            var grandpar_sep = grandpar_level[grandpar_idx].node

            var A_color = int2d{grandpar_sep, sep}
            var filled_A = filled_clusters[A_color][interval]
            var B_color = int2d{par_sep, sep}
            var filled_B = filled_clusters[B_color][interval]
            var C_color = int2d{grandpar_sep, par_sep}
            var filled_C = filled_clusters[C_color][interval]

            var col_cluster_size = clusters[par_sep][interval].volume-1


            if grandpar_sep == par_sep then
              fused_dsyrk(mat_part[A_color], mat_part[B_color], mat_part[C_color],
                          filled_A, filled_B, filled_C, col_cluster_size)
            else
              fused_dgemm(mat_part[A_color], mat_part[B_color], mat_part[C_color],
                          filled_A, filled_B, filled_C, col_cluster_size)
            end

            if debug then
              var A = mat_part[A_color]
              var B = mat_part[B_color]
              var C = mat_part[C_color]
              var sizeA = A.bounds.hi - A.bounds.lo + {1, 1}
              var sizeB = B.bounds.hi - B.bounds.lo + {1, 1}
              var sizeC = C.bounds.hi - C.bounds.lo + {1, 1}

              c.printf("\tLevel: %d GEMM A=(%d, %d) B=(%d, %d) C=(%d, %d)\n\tSizeA: %dx%d Lo: %d %d Hi: %d %d SizeB: %dx%d Lo: %d %d Hi: %d %d SizeC: %dx%d Lo: %d %d Hi: %d %d\n\n",
                       grandpar_lvl,
                       grandpar_sep, sep,
                       par_sep, sep,
                       grandpar_sep, par_sep,
                       sizeA.x, sizeA.y, A.bounds.lo.x, A.bounds.lo.y, A.bounds.hi.x, A.bounds.hi.y,
                       sizeB.x, sizeB.y, B.bounds.lo.x, B.bounds.lo.y, B.bounds.hi.x, B.bounds.hi.y,
                       sizeC.x, sizeC.y, C.bounds.lo.x, C.bounds.lo.y, C.bounds.hi.x, C.bounds.hi.y)

              write_blocks(mat, mat_part, grandpar_lvl, A_color, B_color, C_color, "GEMM", banner, debug_path)
            end
            grandpar_idx = [int](grandpar_idx/2)
          end
        end
      end
    end

    interval += 1
  end

  c.printf("Done factoring.\n")

  if c.strcmp(factor_file, '') ~= 0 then
    write_matrix(mat, mat_part, factor_file, banner)
  end

  var Bentries = region(ispace(int1d, banner.N), double)
  read_vector(b_file, banner.N, Bentries)
  var Bprev_size = 0
  var B = region(ispace(int1d, banner.N), double)
  var X = region(ispace(int1d, banner.N), double)

  var vec_bounds = region(ispace(int1d, num_separators, 1), VecBounds)
  fill(vec_bounds.bounds, rect1d{lo=int1d{0}, hi=int1d{-1}})

  for sep = 1, num_separators+1 do
    var size = separators[sep].volume
    vec_bounds[sep].bounds = rect1d { int1d{Bprev_size}, int1d{Bprev_size + size - 1} }
    vec_bounds[sep].sep = int1d{ sep }
    -- c.printf("Separator: %d Lo: %d Hi: %d Size: %d 1\n", sep, bounds.lo, bounds.hi, bounds.hi - bounds.lo + 1)
    Bprev_size = Bprev_size + size
  end

  var vec_bounds_part = partition(vec_bounds.sep, ispace(int1d, num_separators, 1))
  var Bpart = partition_vec_by_image_range(B, vec_bounds, vec_bounds_part)

  for color in Bpart.colors do
    var part = Bpart[color]
    var sep = separators[color]
    fill_b(sep, Bentries, part)
  end

  c.printf("Forward Substitution\n")
  for lvl = levels-1, -1, -1 do
    var level = tree[lvl]

    for sep_idx in level.ispace do
      var sep = level[sep_idx].node
      var pivot = mat_part[{sep, sep}]
      var bp = Bpart[sep]

      var sizeA = pivot.bounds.hi - pivot.bounds.lo + {1, 1}
      var sizeB = bp.bounds.hi - bp.bounds.lo + 1

      -- c.printf("Level: %d TRSV A=(%d, %d) B=(%d)\nSizeA: %dx%d Lo: %d %d Hi: %d %d SizeB: %d Lo: %d 1 Hi: %d 1\n\n",
      --          level, sep, sep, sep,
      --          sizeA.x, sizeA.y, pivot.bounds.lo.x, pivot.bounds.lo.y, pivot.bounds.hi.x, pivot.bounds.hi.y,
      --          sizeB, bp.bounds.lo, bp.bounds.hi)

      dtrsv(pivot, bp, cblas.CblasLower, cblas.CblasNoTrans)

      var par_idx = [int](sep_idx)
      for par_lvl = lvl-1, -1, -1 do
        par_idx = [int](par_idx/2)
        var par_level = tree[par_lvl]
        var par_sep = par_level[par_idx].node

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
  for par_lvl = 0, levels do
    var par_level = tree[par_lvl]
    for par_idx in par_level.ispace do
      var par_sep = par_level[par_idx].node
      var pivot = mat_part[{par_sep, par_sep}]
      var bp = Bpart[par_sep]

      var sizeA = pivot.bounds.hi - pivot.bounds.lo + {1, 1}
      var sizeB = bp.bounds.hi - bp.bounds.lo + {1, 1}

      -- c.printf("Level: %d TRSV A=(%d, %d) B=(%d)\nSizeA: %dx%d Lo: %d %d Hi: %d %d SizeB: %d Lo: %d 1 Hi: %d 1\n\n",
      --          par_level, par_sep, par_sep, par_sep,
      --          sizeA.x, sizeA.y, pivot.bounds.lo.x, pivot.bounds.lo.y, pivot.bounds.hi.x, pivot.bounds.hi.y,
      --          sizeB, bp.bounds.lo, bp.bounds.hi)

      dtrsv(pivot, bp, cblas.CblasLower, cblas.CblasTrans)

      for lvl = par_lvl+1, levels do
        var level = tree[lvl]
        for sep_idx = [int](par_idx*math.pow(2, lvl-par_lvl)), [int]((par_idx+1)*math.pow(2, lvl-par_lvl)) do
          var sep = level[sep_idx].node
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

  var j = 0
  for sep = 1, num_separators+1 do
    var separator = separators[sep]
    for i in separator.ispace do
      var idxi = separator[i].idx
      X[idxi] = B[j]
      j += 1
    end
  end

  if c.strcmp(solution_file, '') ~= 0 then
    write_solution(solution_file, X)
  end
end

regentlib.start(main, ccholesky.register_mappers)
