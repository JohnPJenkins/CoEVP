/*
Copyright (c) 2016, UChicago Argonne, LLC
All Rights Reserved
SDS TOOLS (ANL-SF-16-009)

OPEN SOURCE LICENSE

Under the terms of Contract No. DE-AC02-06CH11357 with UChicago Argonne,
LLC, the U.S. Government retains certain rights in this software.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the names of UChicago Argonne, LLC or the Department of
Energy nor the names of its contributors may be used to endorse or
promote products derived from this software without specific prior
written permission.

******************************************************************************
DISCLAIMER

THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND.

NEITHER THE UNTED STATES GOVERNMENT, NOR THE UNITED STATES DEPARTMENT
OF ENERGY, NOR UCHICAGO ARGONNE, LLC, NOR ANY OF THEIR EMPLOYEES,
MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY
OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY
INFORMATION, DATA, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS
THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.

******************************************************************************
*/

#ifndef included_ApproxNearestNeighborsFLANNDB_h
#define included_ApproxNearestNeighborsFLANNDB_h
#include "ApproxNearestNeighborsDB.h"

#ifdef FLANN
#include <flann/flann.hpp>

#include <chrono>
#include <vector>
#include <unordered_map>
#include <memory>

class ApproxNearestNeighborsFLANNDB : public ApproxNearestNeighborsDB
{
  public:

  int dim;
  int n_trees, n_checks_default;

  flann::Index<flann::L2<double>> flann_index;
  bool is_empty;
  bool is_timing;
  std::vector<std::chrono::duration<double>> ins_times;
  std::vector<std::chrono::duration<double>> knn_times;

  struct pvpair { std::unique_ptr<double[]> pv; size_t num_values; };

  std::vector<pvpair> ann_pvs;

  ApproxNearestNeighborsFLANNDB(
      int dim,
      int n_trees,
      int n_checks_default,
      bool enable_timing = false) :
    dim(dim),
    n_trees(n_trees),
    n_checks_default(n_checks_default),
    flann_index(flann::KDTreeIndexParams(n_trees)),
    is_empty(true),
    is_timing(enable_timing) { }

  ~ApproxNearestNeighborsFLANNDB();

  void insert(
      double const *point,
      double const *value,
      size_t num_values) override;

  int knn(
      double const *x,
      int k,
      std::vector<size_t> &ids,
      std::vector<double> &dists,
      std::vector<std::vector<double>> &points,
      std::vector<std::vector<double>> &values) override;

  // non-(re)allocating version - fails if not enough space for values
  knnRet knn(
      double const * x,
      int k,
      size_t *ids,
      double *dists,
      double *point_buf,
      double *value_buf,
      size_t value_size_avail,
      size_t *value_offsets) override;

  void dump(const std::string &filename) override;

  private:

  int knn_helper(
      double const *x,
      int k,
      int n_checks,
      std::vector<size_t> &ids,
      std::vector<double> &dists,
      std::vector<std::vector<double>> &points,
      std::vector<std::vector<double>> &values);
};
#endif
#endif
