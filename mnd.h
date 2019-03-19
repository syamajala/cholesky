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


#ifndef MND_H
#define MND_H

#include <stdlib.h>
#include "uthash.h"
#include "legion/legion_c.h"

typedef struct Entry {
  uint64_t idx;
  double val;
  UT_hash_handle hh;
} Entry;

Entry *entries = NULL;

typedef struct SepInfo {
  int levels;
  int num_separators;
} SepInfo;

SepInfo read_separators(char *file,
                     int dim,
                     legion_runtime_t runtime,
                     legion_context_t context,
                     legion_index_space_t is,
                     legion_physical_region_t pr[],
                     legion_field_id_t fld[]);

int*** read_clusters(char *file, int num_separators, int max_intervals, int max_interval_size);
int*** read_clusters2(char *file, size_t len);
void print_clusters(int ***clusters, int num_separators);
void add_entry(uint64_t idx, double val);
double find_entry(uint64_t idx);
void delete_entries();

#endif
