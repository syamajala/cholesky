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
      info.num_separators = atoi(&(line[1]));
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

int*** read_clusters(char *file, int num_separators, int max_intervals, int max_interval_size)
{
  // clusters[separator][interval][dof] = dof of permuted matrix

  char *line = NULL;
  ssize_t read = 0;
  int i = 0;
  FILE *fp = fopen(file, "r");

  int ***clusters = (int ***)malloc(num_separators*sizeof(int**));
  for(int i = 0; i < num_separators; i++)
  {
    clusters[i] = (int **)malloc(max_intervals*sizeof(int *));
    for(int j = 0; j < max_intervals; j++)
    {
      clusters[i][j] = (int *)malloc(max_interval_size*sizeof(int));
    }
  }

  while ((read = getline(&line, &max_interval_size, fp)) != -1) {
    if (i == 0)
    {
      i++;
      continue;
    }

    int interval_sizes[max_interval_size];
    char *rows = strtok(line, ";");
    int separator = atoi(&rows[0])+1;
    int interval = 0;
    int dofs = 1;
    rows = strtok(NULL, ";,");
    while(rows != NULL)
    {
      int row = atoi(&rows[0]);
      clusters[separator][interval][dofs] = row;
      dofs++;
      rows = strtok(NULL, ";,");
      if(rows == NULL)
      {
        dofs -= 2;
        interval_sizes[interval] = dofs;
        interval++;
        clusters[0][0][separator] = interval;

        for(int j = 0; j < interval; j++)
        {
          int interval_size = interval_sizes[j];
          clusters[separator][j][0] = interval_size;
        }
      }
      else if (strcmp("0", rows) == 0)
      {
        interval_sizes[interval] = dofs-1;
        interval++;
        dofs = 1;
      }
    }
    i++;
  }

  fclose(fp);
  if (line)
    free(line);

  return clusters;
}

int*** read_clusters2(char *file, size_t len)
{
  // clusters[separator][interval][dof] = dof of permuted matrix

  char *line = NULL;
  ssize_t read = 0;
  int i = 0;
  FILE *fp = fopen(file, "r");
  int ***clusters = NULL;

  while ((read = getline(&line, &len, fp)) != -1) {
    if (i == 0)
    {
      int num_separators = atoi(&(line[1]));

      clusters = (int ***)malloc((num_separators+1) * sizeof(int **));
      clusters[0] = (int **)malloc(sizeof(int*));
      clusters[0][0] = (int *)malloc((num_separators+1) * sizeof(int)); // num of intervals/separator

      i++;
      continue;
    }

    int temp_row[len][len];
    int interval_sizes[len];
    char *rows = strtok(line, ";");
    int separator = atoi(&rows[0])+1;
    int interval = 0;
    int dofs = 0;
    rows = strtok(NULL, ";,");
    while(rows != NULL)
    {
      int row = atoi(&rows[0]);
      temp_row[interval][dofs] = row;
      dofs++;
      rows = strtok(NULL, ";,");
      if(rows == NULL)
      {
        dofs--;
        interval_sizes[interval] = dofs;
        interval++;
        clusters[0][0][separator] = interval;
        clusters[separator] = (int **)malloc(interval*sizeof(int*));
        for(int i = 0; i < interval; i++)
        {
          int interval_size = interval_sizes[i];
          clusters[separator][i] = (int *)malloc(interval_size * sizeof(int));
          memcpy(&(clusters[separator][i][1]), temp_row[i], interval_size*sizeof(int));
          clusters[separator][i][0] = interval_size;
        }
      }
      else if (strcmp("0", rows) == 0)
      {
        interval_sizes[interval] = dofs;
        interval++;
        dofs = 0;
      }
    }
    i++;
  }

  fclose(fp);
  if (line)
    free(line);

  return clusters;
}

void print_clusters(int ***clusters, int num_separators)
{
  int *intervals = clusters[0][0];
  for(int separator = 1; separator <= num_separators; separator++)
  {
    int num_intervals = intervals[separator];
    printf("separator %d intervals %d\n", separator, num_intervals);

    for(int interval = 0; interval < num_intervals; interval++)
    {
      int interval_size = clusters[separator][interval][0];
      printf("interval %d: ", interval);
      for (int dof = 1; dof <= interval_size; dof++)
      {
        printf("%d ", clusters[separator][interval][dof]);
      }
      printf("\n");
    }
    printf("\n");
  }
}

void add_entry(uint64_t idx, double val)
{
  Entry *e = malloc(sizeof(Entry));
  e->idx = idx;
  e->val = val;

  HASH_ADD(hh, entries, idx, sizeof(uint64_t), e);
}

double find_entry(uint64_t idx)
{
  Entry *e;
  HASH_FIND(hh, entries, &idx, sizeof(uint64_t), e);
  if(e)
    return e->val;

  return 0;
}

void delete_entries() {
  Entry *ce, *tmp;
  HASH_ITER(hh, entries, ce, tmp) {
    HASH_DEL(entries, ce);
    free(ce);
  }
}

/* int main() { */
/*   int ROWS = 400; */
/*   int **separators = read_separators("lapl_20_2_ord_5.txt", ROWS); */
/*   print_separators(separators); */
/*   int ***clusters = read_clusters("lapl_20_2_clust_5.txt", ROWS); */
/*   print_clusters(clusters, separators[0][1]); */
/* } */
