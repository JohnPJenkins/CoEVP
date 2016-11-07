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

#ifndef included_ApproxNearestNeighborsDB_h
#define included_ApproxNearestNeighborsDB_h

#include <vector>
#include <string>

class ApproxNearestNeighborsDB
{
  public:

  virtual void insert(
      double const * point,
      double const * value,
      size_t num_values) = 0;

  struct knnRet { int k; bool overflow; };

  virtual int knn(
      double const * x,
      int k,
      std::vector<size_t> &ids,
      std::vector<double> &dists,
      std::vector<std::vector<double>> &points,
      std::vector<std::vector<double>> &values) = 0;

  // non-(re)allocating version - fails if not enough space for values
  virtual knnRet knn(
      double const * x,
      int k,
      size_t *ids,
      double *dists,
      double *point_buf,
      double *value_buf,
      size_t value_size_avail,
      size_t *value_offsets) = 0; // should be of size k+1

  // optional
  virtual void dump(const std::string &filename) { }

  virtual ~ApproxNearestNeighborsDB() {};
};

#endif
