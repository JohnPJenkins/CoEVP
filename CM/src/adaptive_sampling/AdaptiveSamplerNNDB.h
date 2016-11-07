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

#ifndef ADAPTIVESAMPLERNNDB_INCLUDED
#define ADAPTIVESAMPLERNNDB_INCLUDED

#include <KrigingDataBaseNNDB.h>
#include <ApproxNearestNeighborsDB.h>
#include <ModelDatabase.h>

#include "FineScale.h"

using namespace krigalg;
using namespace krigcpl;

class AdaptiveSamplerNNDB
{
  public:

  AdaptiveSamplerNNDB(
      int pointDimension,
      int valueDimension,
      const std::vector<double> &pointScaling,
      const std::vector<double> &valueScaling,
      int maxKrigingModelSize,
      double theta,
      double meanErrorFactor,
      double tolerance,
      double maxQueryPointModelDistance,
      ApproxNearestNeighborsDB* ann);

  ~AdaptiveSamplerNNDB();

  void sample(
      std::vector<double> &value,
      const std::vector<double> &point,
      const FineScale &fineScaleModel,
      double &error_estimate);

  void printStatistics(std::ostream& outputStream);

  void printNewInterpolationStatistics(std::ostream& outputStream);

  // shim for KrigingDataBaseNNDB calls + my stats
  int getNumberStatistics() const;
  void getStatistics(double *stats, int size) const;
  std::vector<std::string> getStatisticsNames() const;

  int getNumSamples() const {return m_num_samples;}

  int getNumSuccessfulInterpolations() const {return m_num_successful_interpolations;}

  double getAveragePointNorm() const {return m_point_norm_sum / m_num_samples;}

  double getAverageValueNorm() const {return m_value_norm_sum / m_num_samples;}

  double getPointNormMax() const {return m_point_norm_max;}

  double getValueNormMax() const {return m_value_norm_max;}

  void setVerbose(const bool verbose) const {m_verbose = verbose;}

  void verifyInterpolationAccuracy(
      const std::vector<double>& point,
      const std::vector<double>& value,
      const FineScale&           fineScaleModel ) const;

  void printInterpolationFailure(
      const std::vector<double>& point,
      const std::vector<double>& value,
      const std::vector<double>& exact_value ) const;

  double pointL2Norm( const double* point ) const;

  double pointMaxNorm( const double* point ) const;

  double pointL2Norm( const std::vector<double>& point ) const;

  double pointMaxNorm( const std::vector<double>& point ) const;

  double valueL2Norm( const double* value ) const;

  double valueMaxNorm( const double* value ) const;

  double valueL2Norm( const std::vector<double>& value ) const;

  double valueMaxNorm( const std::vector<double>& value ) const;

  KrigingDataBaseNNDB* m_interp;

  ApproxNearestNeighborsDB* m_ann;

  std::vector<double> m_pointScaling;
  std::vector<double> m_valueScaling;

  int m_pointDimension;
  int m_valueDimension;
  int m_valueAllocated;
  int m_maxKrigingModelSize;
  int m_maxNumberSearchModels;
  double m_theta;
  double m_meanErrorFactor;
  double m_tolerance;
  double m_maxQueryPointModelDistance;

  // cache of previous stats for printNewInterpolationStatistics
  double *m_prev_stats;
  int m_num_samples;
  int m_num_successful_interpolations;
  int m_num_fine_scale_evaluations;

  double m_point_norm_sum;
  double m_value_norm_sum;

  double m_point_norm_max;
  double m_value_norm_max;

  mutable bool m_verbose;
};

#endif
