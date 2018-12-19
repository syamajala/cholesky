// Copyright 2018 Stanford University
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


int** build_separator_tree(int **separators)
{
  int levels = separators[0][0];
  int num_separators = separators[0][1];

  int **tree = (int **)malloc(levels*sizeof(int*));
  for(int level = 0; level < levels; level++)
  {
    int elems = pow(2, level);
    tree[level] = (int *)malloc(elems*sizeof(int));

    for(int n = 0; n < elems; n++)
    {
      tree[level][n] = num_separators;
      num_separators--;
    }
  }
  return tree;
}

int** read_separators(char *file, size_t len)
{
  char *line = NULL;
  ssize_t read = 0;
  int i = 0;
  FILE *fp = fopen(file, "r");
  int **separators = NULL;

  while ((read = getline(&line, &len, fp)) != -1) {
    if (i == 0)
    {
      int levels = atoi(&(line[0]));
      int num_separators = atoi(&(line[1]));

      separators = (int **)malloc((num_separators+1) * sizeof(int *));
      separators[0] = (int *)malloc(3 * sizeof(int));

      separators[0][0] = levels;
      separators[0][1] = num_separators;
      separators[0][2] = 0;
      ++i;

      continue;
    }

    int temp_row[len];
    char *rows = strtok(line, ";");
    int separator = atoi(&rows[0])+1;
    int num_rows = 0;
    rows = strtok(NULL, ",");

    while (rows != NULL)
    {
      int row = atoi(&rows[0]);
      temp_row[num_rows] = row;
      num_rows++;
      rows = strtok(NULL, ",");
    }
    separators[separator] = (int *)malloc((num_rows+1) * sizeof(int));
    memcpy(&(separators[separator][1]), temp_row, num_rows*sizeof(int));
    separators[separator][0] = num_rows;
    ++i;
  }

  fclose(fp);
  if (line)
    free(line);

  return separators;
}

void print_separators(int **separators)
{
  printf("levels: %d\n", separators[0][0]);
  printf("separators: %d\n", separators[0][1]);
  int num_separators = separators[0][1];

  for(int separator = 1; separator <= num_separators; separator++)
  {
    int separator_size = separators[separator][0];
    printf("separator %d size %d: ", separator, separator_size);
    for(int row = 1; row <= separator_size; row++)
    {
      printf("%d ", separators[separator][row]);
    }
    printf("\n");
  }
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

void add_entry(int idx, double val)
{
  Entry *e = malloc(sizeof(Entry));
  e->idx = idx;
  e->val = val;

  HASH_ADD_INT( entries, idx, e);
}

double find_entry(int idx)
{
  Entry *e;
  HASH_FIND_INT(entries, &idx, e);
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
