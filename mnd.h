#ifndef MND_H
#define MND_H

#include <stdlib.h>

int** read_separators(char *file, size_t len);
int* row_to_separator(int **separators, int num_rows);
void print_separator(int **separators);

#endif
