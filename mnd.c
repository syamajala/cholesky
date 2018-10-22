#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int **read_separators(size_t len)
{
  char *file = "lapl_20_2_ord_5.txt";
  char *line = NULL;
  ssize_t read = 0;
  int i = 0;
  FILE *fp = fopen(file, "r");
  int **separators = NULL;

  while ((read = getline(&line, &len, fp)) != -1) {
    if (i == 0)
    {
      int num_separators = atoi(&(line[1]));
      separators = (int **)malloc(num_separators * sizeof(int *));
      ++i;
      continue;
    }

    int temp_row[len];
    char *rows = strtok(line, ";");
    int separator = atoi(&rows[0]);
    int num_rows = 0;
    rows = strtok(NULL, ",");
    while (rows != NULL)
    {
      int row = atoi(&rows[0]);
      temp_row[num_rows] = row;
      num_rows++;
      rows = strtok(NULL, ",");
    }
    separators[separator] = (int *)malloc(num_rows * sizeof(int));
    memcpy(separators[separator], temp_row, num_rows);

    ++i;
  }

  fclose(fp);
  if (line)
    free(line);

  return separators;
}

int main() {
  int **separators = read_separators(400);
}
