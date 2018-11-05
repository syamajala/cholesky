#include "mnd.h"
#include <stdio.h>
#include <string.h>
#include <math.h>


int*** read_clusters(char *file, size_t len)
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
      int levels = atoi(&(line[0]));
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
      int row = atoi(&rows[0])+1;
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
      temp_row[num_rows] = row+1;
      num_rows++;
      rows = strtok(NULL, ",");
    }
    separators[separator] = (int *)malloc((num_rows+2) * sizeof(int));
    memcpy(&(separators[separator][1]), temp_row, num_rows*sizeof(int));
    separators[separator][0] = num_rows;
    separators[separator][num_rows] = 0;
    ++i;
  }

  fclose(fp);
  if (line)
    free(line);

  return separators;
}

int* row_to_separator(int** separators, int num_rows)
{
  int *rows = (int *)malloc(num_rows * sizeof(int));
  int num_separators = separators[0][1];
  for(int separator = 1; separator <= num_separators; separator++)
  {
    int *row = &(separators[separator][1]);
    while(*row)
    {
      rows[(*row)-1] = separator-1;
      row++;
    }
  }
  return rows;
}

void print_separator(int **separators)
{
  printf("levels: %d\n", separators[0][0]);
  printf("separators: %d\n", separators[0][1]);
  int num_separators = separators[0][1];

  for(int separator = 1; separator <= num_separators; separator++)
  {
    printf("separator %d:", separator-1);
    int *row = separators[separator];
    while(*row)
    {
      printf("%d ", (*row)-1);
      row++;
    }
    printf("\n");
  }
}

/* int main() { */
/*   char *file = "lapl_20_2_ord_5.txt"; */
/*   int ROWS = 400; */
/*   int **separators = read_separators(file, ROWS); */
/*   int rows[ROWS]; */
/*   row_to_separator(separators, rows); */
/* } */
