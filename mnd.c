// Copyright 2019 Stanford University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mnd.h"
#include <stdio.h>
#include <string.h>
#include <math.h>


SepInfo read_separators(char *file,
                        int dim,
                        legion_runtime_t runtime,
                        legion_context_t context,
                        legion_index_space_t is,
                        legion_physical_region_t pr[],
                        legion_field_id_t fld[])
{
  char *line = NULL;
  ssize_t read = 0;
  int i = 0;
  FILE *fp = fopen(file, "r");
  legion_accessor_array_1d_t i_accessor = legion_physical_region_get_field_accessor_array_1d(pr[0], fld[0]);
  legion_accessor_array_1d_t sep_accessor = legion_physical_region_get_field_accessor_array_1d(pr[1], fld[1]);
  legion_index_iterator_t it = legion_index_iterator_create(runtime, context, is);
  SepInfo info;

  while ((read = getline(&line, &dim, fp)) != -1) {
    if (i == 0)
    {
      info.levels = atoi(&(line[0]));
      info.num_separators = atoi(&(line[2]));
      ++i;

      continue;
    }

    char *rows = strtok(line, ";");
    int separator = atoi(&rows[0])+1;
    rows = strtok(NULL, ",");
    while (rows != NULL)
    {
      int row = atoi(&rows[0]);
      legion_ptr_t point = legion_index_iterator_next(it);
      legion_accessor_array_1d_write(sep_accessor, point, &separator, sizeof(int));
      legion_accessor_array_1d_write(i_accessor, point, &row, sizeof(int));
      rows = strtok(NULL, ",");
    }
    ++i;
  }

  fclose(fp);
  return info;
}

int read_clusters(char *file,
                  int dim,
                  legion_runtime_t runtime,
                  legion_context_t context,
                  legion_index_space_t is,
                  legion_physical_region_t pr[],
                  legion_field_id_t fld[])
{
  // clusters[separator][interval][dof] = dof of permuted matrix

  char *line = NULL;
  ssize_t read = 0;
  int i = 0;
  FILE *fp = fopen(file, "r");
  int max_int_size = -1;
  legion_accessor_array_1d_t idx_accessor = legion_physical_region_get_field_accessor_array_1d(pr[0], fld[0]);
  legion_accessor_array_1d_t int_accessor = legion_physical_region_get_field_accessor_array_1d(pr[1], fld[1]);
  legion_accessor_array_1d_t sep_accessor = legion_physical_region_get_field_accessor_array_1d(pr[2], fld[2]);
  legion_index_iterator_t it = legion_index_iterator_create(runtime, context, is);

  while ((read = getline(&line, &dim, fp)) != -1) {
    if (i == 0)
    {
      i++;
      continue;
    }

    char *rows = strtok(line, "; ");
    int separator = atoi(&rows[0])+1;
    int interval = 0;
    int dofs = 0;
    rows = strtok(NULL, ",; ");
    while(rows != NULL)
    {
      int row = atoi(&rows[0]);
      dofs++;
      rows = strtok(NULL, ",; ");

      if(rows == NULL)
      {
        /* printf("Sep Done.\n"); */
        if(dofs > max_int_size)
        {
          max_int_size = dofs;
        }
      }
      else if (strcmp("0", rows) == 0)
      {
        /* printf("Separator: %d Interval: %d Row: %d\n", separator, interval, row); */
        /* printf("Int done.\n"); */

        legion_ptr_t point = legion_index_iterator_next(it);
        legion_accessor_array_1d_write(idx_accessor, point, &row, sizeof(int));
        legion_accessor_array_1d_write(int_accessor, point, &interval, sizeof(int));
        legion_accessor_array_1d_write(sep_accessor, point, &separator, sizeof(int));

        if(dofs > max_int_size)
        {
          max_int_size = dofs;
        }
        interval++;
        dofs = 0;
      }
      else
      {
        /* printf("Separator: %d Interval: %d Row: %d\n", separator, interval, row); */

        legion_ptr_t point = legion_index_iterator_next(it);
        legion_accessor_array_1d_write(idx_accessor, point, &row, sizeof(int));
        legion_accessor_array_1d_write(int_accessor, point, &interval, sizeof(int));
        legion_accessor_array_1d_write(sep_accessor, point, &separator, sizeof(int));
      }
    }
    i++;
  }

  fclose(fp);

  return max_int_size;
}

void read_matrix(char* file,
                 uint64_t cols,
                 int nz,
                 legion_runtime_t runtime,
                 legion_context_t context,
                 legion_index_space_t is,
                 legion_physical_region_t pr[],
                 legion_field_id_t fld[])
{
  FILE *fp = fopen(file, "r");
  char buff[1024];
  fgets(buff, 1024, fp);
  fgets(buff, 1024, fp);

  legion_accessor_array_1d_t idx_accessor = legion_physical_region_get_field_accessor_array_1d(pr[0], fld[0]);
  legion_accessor_array_1d_t val_accessor = legion_physical_region_get_field_accessor_array_1d(pr[1], fld[1]);
  legion_accessor_array_1d_t col_accessor = legion_physical_region_get_field_accessor_array_1d(pr[2], fld[2]);

  for(int n = 0; n < nz; n++)
  {
    uint64_t i = 0;
    uint64_t j = 0;
    double val = 0.0;
    fscanf(fp, "%lu %lu %lg\n", &i, &j, &val);
    i -= 1;
    j -= 1;
    uint64_t iidx = i*cols+j;
    legion_point_1d_t p = { iidx % nz };
    legion_point_2d_t idx = {i, j};

    double pval = 0.0;
    legion_accessor_array_1d_read_point(val_accessor, p, (void *)&pval, sizeof(double));
    legion_point_1d_t p2 = p;
    int collision = 0;
    while (pval != 0)
    {
      collision = 1;
      p2.x[0] = (p2.x[0] + 1) % nz;
      legion_accessor_array_1d_read_point(val_accessor, p2, (void *)&pval, sizeof(double));
    }

    if (collision)
    {
      legion_point_1d_t col = {-1};
      legion_accessor_array_1d_read_point(col_accessor, p, &col, sizeof(legion_point_1d_t));
      while (*col.x != -1)
      {
        p = col;
        legion_accessor_array_1d_read_point(col_accessor, p, &col, sizeof(legion_point_1d_t));
      }
      legion_accessor_array_1d_write_point(col_accessor, p, &p2, sizeof(legion_point_1d_t));
      p = p2;
    }

    legion_accessor_array_1d_write_point(idx_accessor, p, &idx, sizeof(legion_point_2d_t));
    legion_accessor_array_1d_write_point(val_accessor, p, &val, sizeof(double));
  }

  fclose(fp);
}

void read_vector(char* file,
                 int n,
                 legion_runtime_t runtime,
                 legion_context_t context,
                 legion_index_space_t is,
                 legion_physical_region_t pr[],
                 legion_field_id_t fld[])
{
  FILE *fp = fopen(file, "r");

  for(int i = 0; i < 3; i++)
  {
    char buff[1024];
    fgets(buff, 1024, fp);
  }

  legion_accessor_array_1d_t val_accessor = legion_physical_region_get_field_accessor_array_1d(pr[0], fld[0]);
  legion_index_iterator_t it = legion_index_iterator_create(runtime, context, is);

  for(int i = 0; i < n; i++)
  {
    legion_ptr_t point = legion_index_iterator_next(it);
    double val = 0.0;
    fscanf(fp, "%lg\n", &val);
    legion_accessor_array_1d_write(val_accessor, point, &val, sizeof(double));
  }

  fclose(fp);
}
