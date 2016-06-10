//
// File:        KrigingDataBaseNNDB.h
//
// Revision:    $Revision$
// Modified:    $Date$
// Description: Kriging interpolation using a keyed database.

#ifndef included_krigcpl_KrigingDataBaseNNDB_h
#define included_krigcpl_KrigingDataBaseNNDB_h

#include <unordered_map>

#include "InterpolationModel.h"
#include "InterpolationModelFactory.h"
#include "ApproxNearestNeighborsDB.h"

namespace krigcpl {
  /*!
   * @brief Interpolation database implementation using on-the-fly Kriging
   * model construction
   */
class KrigingDataBaseNNDB
{
  public:
  /*!
   * construction.
   * 
   * @param pointdimension the dimension of the point space.
   * @param valuedimension the dimension of the value space.
   * @param modelfactory   handle to a factory for creating interpolation models
   * @param ann            handle to an approximate nearest neighbor database
   * @param maxkrigingmodelsize maximum number of point/value pairs in
   *                            a single kriging model.
   * @param meanerrorfactor the value of the coefficient multiplying
   *                        the mean square error.
   * @param tolerance Requested tolerance.
   * @param maxQueryPointModelDistance The maximum distance between the 
   *                                   query point and the model for which
   *                                   interpolation is still attempted. 
   */
  KrigingDataBaseNNDB(
      int pointDimension,
      int valueDimension,
      const krigalg::InterpolationModelFactoryPointer  & modelFactory,
      ApproxNearestNeighborsDB& ann,
      int maxKrigingModelSize,
      double meanErrorFactor,
      double tolerance,
      double maxQueryPointModelDistance);

  KrigingDataBaseNNDB(const KrigingDataBaseNNDB &) = delete;
  KrigingDataBaseNNDB(KrigingDataBaseNNDB &&) = delete;
  const KrigingDataBaseNNDB & operator=(const KrigingDataBaseNNDB&) = delete;

  /*!
   * Destruction.
   */
  ~KrigingDataBaseNNDB();

  /*!
   * Compute interpolated value at a point.
   *
   * @param value Pointer for storing the value. Size of at least
   *              _valueDimension assumed.
   * @param point Pointer for accesing the point. Needs to have the size
   *              of at least _pointDimension.
   * @param error_estimate Error estimate
   *
   * @return true if the interpolation successful; false otherwise. 
   */
  bool interpolate(
      double * value,
      const double * point,
      double & error_estimate);

  /*!
   * Compute interpolated value at a point.
   *
   * @param value Pointer for storing the value. Size of at least
   *              _valueDimension assumed.
   * @param gradient Pointer for storing gradient of the value wrt.
   *                 point evaluated at the point.
   * @param point Pointer for accesing the point. Needs to have the size
   *              of at least _pointDimension.
   * @param error_estimate Error estimate
   *
   * @return true if the interpolation successful; false otherwise. 
   */
  bool interpolate(
      double * value,
      double * gradient,
      const double * point,
      double & error_estimate);

  /*!
   * Insert the point-value pair into the database.
   *
   * @param point  Pointer to point data. Needs to have the size of
   *               at least _pointDimension.
   * @param value  Pointer to value data. Needs to have the size of 
   *               at least _valueDimension
   * @param gradient Pointer to gradient of the value wrt. point.
   */
  void insert(
      const double * point,
      const double * value,
      const double * gradient);

  /*!
   * Get the number of performance statistic data collected
   *
   * @return Number of data collected.
   */
  int getNumberStatistics() const;

  /*!
   * Provide performance statistic data collected so far
   *
   * @param stats A handle to an array.
   * @param size  Size of the stats array. 
   */
  void getStatistics(double * stats, int size) const;

  /*! 
   * Provide string descriptions of statistics data.
   *
   * @return An STL-vector of strings.
   */
  std::vector<std::string> getStatisticsNames() const;

  void printDBStats(std::ostream & outputStream) const;

  private:

  krigalg::InterpolationModelFactoryPointer _modelFactory;

  // model cache
  krigalg::InterpolationModelPtr _modelcache;
  krigalg::Point _modelcache_center;

  ApproxNearestNeighborsDB&     _ann;

  const int    _pointDimension;
  const int    _valueDimension;
  const int    _maxKrigingModelSize;
  const double _meanErrorFactor;
  const double _tolerance;
  const double _maxQueryPointModelDistance;

  //
  // statistics
  //
  uint64_t _numPointValuePairs;
  uint64_t _num_interpolations;
  uint64_t _num_err_calls;
  uint64_t _num_err_toosmalls;
  uint64_t _num_knn_singleton_models;
  uint64_t _num_knn_nonsingleton_models;
  uint64_t _num_valid_knn_models;
  uint64_t _num_invalid_knn_models;
  uint64_t _num_modelcache_hits;
  uint64_t _num_modelcache_misses;

  //
  // helper functions
  //

  /*!
   * create a Kriging model by consulting the ANN
   *
   * @param point the input point, assumed to be of size _pointDimension
   */
  krigalg::InterpolationModelPtr findBuildCoKrigingModel(const double *point);

  /*!
   * interpolation helper. Doesn't strictly need to be a member, but we also
   * want to do some stat tracking.
   * @param queryPoint the input query point
   * @param modelptr the model to interpolate with
   * @param value the output value
   * @param gradient the output gradient (pass nullptr if inapplicable)
   * @param errorEstimate the output error estimation
   * @return true on successful interpolation, false otherwise
   */
  bool checkErrorAndInterpolate(
      const krigalg::Point &queryPoint,
      double *value,
      double *gradient,
      double &errorEstimate);

  /*!
   * error calculation helper. Doesn't strictly need to be a member, but we
   * also want to do some stat tracking.
   * @param queryPoint the query point
   * @param model the model to interpolate with
   * @param valueId the value dimension to interpolate
   * @param meanErrorFactor the multiplier for the MSE (which gets squared)
   * @return the squared error
   */
  double compKrigingError(
      const krigalg::Point &queryPoint,
      const krigalg::InterpolationModel &model,
      int valueId);
};

}

#endif // included_krigcpl_KrigingDataBaseNNDB_h
