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


#ifndef MND_H
#define MND_H

#include <stdlib.h>
#include "uthash.h"

typedef struct Entry {
  uint64_t idx;
  double val;
  UT_hash_handle hh;
} Entry;

Entry *entries = NULL;

int*** read_clusters(char *file, int num_separators, int max_intervals, int max_interval_size);
int*** read_clusters2(char *file, size_t len);
void print_clusters(int ***clusters, int num_separators);
int** build_separator_tree(int **separators);
int** read_separators(char *file, size_t len);
void print_separators(int **separators);
void add_entry(uint64_t idx, double val);
double find_entry(uint64_t idx);
void delete_entries();

#endif
