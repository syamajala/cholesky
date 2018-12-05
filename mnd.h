#ifndef MND_H
#define MND_H

#include <stdlib.h>

int*** read_clusters(char *file, int num_separators, int max_intervals, int max_interval_size);
void print_clusters(int ***clusters, int num_separators);
int** build_separator_tree(int **separators);
int** read_separators(char *file, size_t len);
void print_separators(int **separators);

#endif
