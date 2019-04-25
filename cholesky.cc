/* Copyright 2019 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cholesky.h"
#include "mappers/default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

///
/// Mapper
///

static LegionRuntime::Logger::Category log_cholesky("cholesky");

class CholeskyMapper : public DefaultMapper
{
public:
  CholeskyMapper(MapperRuntime *rt,
                 Machine machine,
                 Processor local,
                 const char *mapper_name);

  void select_partition_projection(const MapperContext ctx,
                                   const Partition& partition,
                                   const SelectPartitionProjectionInput& input,
                                   SelectPartitionProjectionOutput& output);

  LogicalRegion default_policy_select_instance_region(MapperContext ctx,
                                                      Memory target_memory,
                                                      const RegionRequirement &req,
                                                      const LayoutConstraintSet &layout_constraints,
                                                      bool force_new_instances,
                                                      bool meets_constraints);
};

CholeskyMapper::CholeskyMapper(MapperRuntime *rt, Machine machine, Processor local,
                               const char *mapper_name)
  : DefaultMapper(rt, machine, local, mapper_name)
{
}

void CholeskyMapper::select_partition_projection(const MapperContext ctx,
                                                 const Partition& partition,
                                                 const SelectPartitionProjectionInput& input,
                                                 SelectPartitionProjectionOutput& output)
{
  log_cholesky.spew("Default select_partition_projection in %s",
                    get_mapper_name());
  output.chosen_partition = LogicalPartition::NO_PART;
}

LogicalRegion CholeskyMapper::default_policy_select_instance_region(MapperContext ctx,
                                                                    Memory target_memory,
                                                                    const RegionRequirement &req,
                                                                    const LayoutConstraintSet &layout_constraints,
                                                                    bool force_new_instances,
                                                                    bool meets_constraints)
{
  return req.region;
}


static void create_mappers(Machine machine, HighLevelRuntime *runtime, const std::set<Processor> &local_procs)
{

  for (std::set<Processor>::const_iterator it = local_procs.begin();
       it != local_procs.end(); it++)
  {
    CholeskyMapper* mapper = new CholeskyMapper(runtime->get_mapper_runtime(),
                                                machine, *it, "cholesky_mapper");
    runtime->replace_default_mapper(mapper, *it);
  }
}

void register_mappers()
{
  HighLevelRuntime::add_registration_callback(create_mappers);
}
