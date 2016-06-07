// DO-NOT-DELETE revisionify.begin() 
/*
 * Copyright (c) 2016 UChicago Argonne, LLC
 * See COPYRIGHT-ANL in top-level directory.
 *
 * Copyright (c) 2014 Lawrence Livermore National Security, LLC.
 * See below notice.
 *
 */
/*

                            Copyright (c) 2014.
               Lawrence Livermore National Security, LLC.
         Produced at the Lawrence Livermore National Laboratory
                             LLNL-CODE-656392.
                           All rights reserved.

This file is part of CoEVP, Version 1.0. Please also read this link -- http://www.opensource.org/licenses/index.php

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the disclaimer below.

   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the disclaimer (as noted below)
     in the documentation and/or other materials provided with the
     distribution.

   * Neither the name of the LLNS/LLNL nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Additional BSD Notice

1. This notice is required to be provided under our contract with the U.S.
   Department of Energy (DOE). This work was produced at Lawrence Livermore
   National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.

2. Neither the United States Government nor Lawrence Livermore National
   Security, LLC nor any of their employees, makes any warranty, express
   or implied, or assumes any liability or responsibility for the accuracy,
   completeness, or usefulness of any information, apparatus, product, or
   process disclosed, or represents that its use would not infringe
   privately-owned rights.

3. Also, reference herein to any specific commercial products, process, or
   services by trade name, trademark, manufacturer or otherwise does not
   necessarily constitute or imply its endorsement, recommendation, or
   favoring by the United States Government or Lawrence Livermore National
   Security, LLC. The views and opinions of authors expressed herein do not
   necessarily state or reflect those of the United States Government or
   Lawrence Livermore National Security, LLC, and shall not be used for
   advertising or product endorsement purposes.

*/
// DO-NOT-DELETE revisionify.end() 
//
// File:        KrigingDataBaseNNDB.cc
// 
// Revision:    $Revision$
// Modified:    $Date$
// Description: Interpolation database using kriging interpolation.
//

#include <fstream>

#include "KrigingDataBaseNNDB.h"
#include <kriging/SecondMoment.h>
#include <base/ResponsePoint.h>

#define PRINT_STATS 1

#define STRING_DIGITS 16
#define MURMUR_SEED 42

using namespace krigalg;

namespace krigcpl {

namespace {

//
// local functions
//

//
// compute center of mass for a kriging model
//
Point
getModelCenterMass(const InterpolationModel & krigingModel)
{
  // get all points in the model
  const std::vector<Point> & points = krigingModel.getPoints();

  assert(!points.empty());

  // instantiate center od mass
  Point centerMass(points.front().size(), 0.0);

  // iterate over all points
  std::vector<Point>::const_iterator pointsIter;
  std::vector<Point>::const_iterator pointsEnd = points.end();

  for (pointsIter = points.begin(); pointsIter != pointsEnd; ++pointsIter) {
    // get Point handle
    const Point & point = *pointsIter;
    // accumulate
    centerMass += static_cast<Vector>(point);
  }

  // scale
  mtl::scale(centerMass, 1.0/points.size());

  return centerMass;
}

} // namespace

// Object class member definitions start here

//
// construction/destruction
//

KrigingDataBaseNNDB::KrigingDataBaseNNDB(
    int pointDimension,
    int valueDimension,
    const krigalg::InterpolationModelFactoryPointer  & modelFactory,
    ApproxNearestNeighborsDB& ann,
    int maxKrigingModelSize,
    double meanErrorFactor,
    double tolerance,
    double maxQueryPointModelDistance) :
  _modelFactory(modelFactory),
  _ann(ann),
  _pointDimension(pointDimension),
  _valueDimension(valueDimension),
  _maxKrigingModelSize(maxKrigingModelSize),
  _meanErrorFactor(meanErrorFactor),
  _tolerance(tolerance),
  _maxQueryPointModelDistance(maxQueryPointModelDistance),
  _numPointValuePairs(0),
  _num_err_calls(0),
  _num_err_toosmalls(0) {  }

KrigingDataBaseNNDB::~KrigingDataBaseNNDB() { }

bool
KrigingDataBaseNNDB::interpolate(
    double            * value,
    const double      * point,
    double            & error_estimate)
{
  return interpolate(value, nullptr, point, error_estimate);
}

bool KrigingDataBaseNNDB::interpolate(
    double * value,
    double * gradient,
    const double * point,
    double & error_estimate)
{
  // TODO: insert model cache lookup here

  //
  // Find closest set of points comprising a kriging model.
  //
  InterpolationModelPtr modelptr = findBuildCoKrigingModel(point);

  if (modelptr == nullptr || !modelptr->isValid()) return false;

  const Point queryPoint(_pointDimension, point);

  return checkErrorAndInterpolate(
      queryPoint, *modelptr, value, gradient, error_estimate);
}

void
KrigingDataBaseNNDB::insert(
    const double * point,
    const double * value,
    const double * gradient)
{
  ++_numPointValuePairs;

  //
  // copy value and gradient data into form that can be ingested directly on
  // creation
  //

  const std::vector<double> pointvec(point, point+_pointDimension);

  // TODO: currently need a built model to determine whether gradient is being
  // used, could probably just cache the result of one. Assuming that a model
  // either does or does not (and a couple of models confirm this),
  // so shouldn't be any runtime issues
  const bool hasGradient = _modelFactory->build()->hasGradient();
  std::vector<double> valvec(
      _valueDimension * (hasGradient ? (_pointDimension + 1) : 1));

  if (hasGradient) {
    // NOTE: logic taken from copyValueData - puts values and point gradients
    // "in-line" with each other (value,gradient_0,...gradient_pdim-1). Note
    // that gradients must be transposed from their input format
    for (int i = 0; i < _valueDimension; i++) {
      int off = i * (_pointDimension+1);
      valvec[off] = value[i];
      for (int j = 0; j < _pointDimension; j++)
        valvec[off + j + 1] = gradient[j*_valueDimension + i];
    }
  }
  else {
    std::copy(value, value+_valueDimension, valvec.begin());
  }

  // add point to the NN-DB
  _ann.insert(pointvec, valvec);
}

//
// get the number of statistcs 
//

int
KrigingDataBaseNNDB::getNumberStatistics() const { return 1; }

//
// Write performance stats
//

void
KrigingDataBaseNNDB::getStatistics(double *stats, int size) const
{
  if (size <= 0) return;
  if (size > 4) size = 4;
  switch(size) {
    case 4: stats[3] = _num_err_toosmalls;
    case 3: stats[2] = _num_err_calls;
    case 2: stats[1] = _num_interpolations;
    case 1: stats[0] = _numPointValuePairs;
  }
}

//
// Provide short descriptions of statistic data
//

std::vector<std::string>
KrigingDataBaseNNDB::getStatisticsNames() const
{
  std::vector<std::string> names;

  names.emplace_back("Number of point/value pairs");
  names.emplace_back("Number of interpolation calls (across all value dims)");
  names.emplace_back("Number of error calls (across all value dims)");
  names.emplace_back("Number of error calls with too small models (across all value dims) (note: 2 interpolations for each of these)");

  return names;
}

InterpolationModelPtr KrigingDataBaseNNDB::findBuildCoKrigingModel(
    const double *point)
{
  // knn inputs
  std::vector<int> ids(_maxKrigingModelSize);
  std::vector<double> dists(_maxKrigingModelSize);
  std::vector<std::vector<double>> points(_maxKrigingModelSize);
  std::vector<std::vector<double>> values(_maxKrigingModelSize);
  const std::vector<double> x(point, point+_pointDimension);

  // do the knn
  int num_points = _ann.knn(x, _maxKrigingModelSize, ids, dists, points, values);
  if (num_points == 0) return nullptr;

  // TODO: replace point-by-point with variant that builds from
  // all points at once
  auto modelptr = _modelFactory->build();
  for (int i = 0; i < num_points; i++) {
    // sanity check the value sizes
    if (modelptr->hasGradient()) {
      assert(values[i].size() ==
          static_cast<size_t>(_valueDimension*(_pointDimension+1)));
    }
    else {
      assert(values[i].size() == static_cast<size_t>(_valueDimension));
    }

    const Point pt(_pointDimension, points[i].data());
    std::vector<Value> ptValue;

    // point is stored in the ANN in the already transformed form, so can
    // memcpy directly

    for (int j = 0; j < _valueDimension; j++) {
      if (modelptr->hasGradient())
        ptValue.emplace_back(
            _pointDimension+1, values[i].data()+j*(_pointDimension+1));
      else
        ptValue.emplace_back(1, values[i].data() + j);
    }

    modelptr->addPoint(pt, ptValue);
  }

  // perform distance check
  // TODO: can do this check without constructing the model
  const Point modelCenter = getModelCenterMass(*modelptr);
  const Vector pointRelativePosition =
    Point(_pointDimension, point) - modelCenter;
  const double distanceSqr =
    krigalg::dot(pointRelativePosition, pointRelativePosition);

  // distance check failure == build failure
  if (distanceSqr > _maxQueryPointModelDistance*_maxQueryPointModelDistance)
    return nullptr;
  else
    return modelptr;
}

bool KrigingDataBaseNNDB::checkErrorAndInterpolate(
    const Point &queryPoint,
    const InterpolationModel &model,
    double *value,
    double *gradient,
    double &errorEstimate)
{
  const bool hasGradient = model.hasGradient();
  assert(!(hasGradient && gradient == nullptr));

  const double toleranceSqr = _tolerance*_tolerance;

  // NOTE: doing max calc on the squared error estimate - take square root at
  // the end
  errorEstimate = 0.;

  for (int i = 0; i < _valueDimension; i++) {
    const double errEst = compKrigingError(queryPoint, model, i);

    errorEstimate = std::max(errorEstimate, errEst);

    if (fabs(errEst) > toleranceSqr) {
      errorEstimate = sqrt(errorEstimate);
      return false;
    }

    //
    // put the value of the function into value (valueEstimate
    // contains the value of the function follwed by the gradient;
    // here we are interested only in the value of the function so
    // the gradient is simply discarded).
    //
    _num_interpolations++;
    const Value valueEstimate = model.interpolate(i, queryPoint);
    value[i] = valueEstimate[0];
    if (hasGradient) {
      std::copy(valueEstimate.begin()+1, valueEstimate.end(), gradient);
    }
  }

  errorEstimate = sqrt(errorEstimate);
  return true;
}

//
// compute kriging error at a point; the issue here is that if the
// kriging model contains a single point then the error will
// naturally be computed as zero; if this is the case we will try to
// estimate the error wrt a constant function
//
double
KrigingDataBaseNNDB::compKrigingError(
    const Point              & queryPoint,
    const InterpolationModel & model,
    int                        valueId)
{
  _num_err_calls++;
  const int numberPoints = model.getNumberPoints();

  assert(numberPoints >= 1);

  // compute min number of points to attempt meaningful
  // interpolation
  const int minNumberPoints = model.hasGradient() ? 1 :
    2*(model.getPointDimension() + 1) - 1;

  // compute the error if the kriging model contains a single point;
  // otherwise, simply return the kriging prediction
  if (numberPoints <= minNumberPoints ) {
    _num_err_toosmalls++;

    // get the kriging estimate at query point
    _num_interpolations++;
    const Value queryValue = model.interpolate(valueId, queryPoint);

    // get all points in the model; there should really only be
    // ONE if we have gradient information
    const std::vector<Point> & points = model.getPoints();
    assert(model.hasGradient() ? (points.size() == 1) : true );

    // get the kriging estimate at the origin of the kriging model
    _num_interpolations++;
    const Value originValue = model.interpolate(valueId, points.front());

    // compute the error as the difference between the value at the
    // queryPoint and origin point
    return (queryValue[0] - originValue[0]) * (queryValue[0] - originValue[0]);
  }
  else
    return _meanErrorFactor*_meanErrorFactor*
      model.getMeanSquaredError(valueId, queryPoint)[0];
}

} // namespace krigcpl
