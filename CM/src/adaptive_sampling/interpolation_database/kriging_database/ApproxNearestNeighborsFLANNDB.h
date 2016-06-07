#ifndef included_ApproxNearestNeighborsFLANNDB_h
#define included_ApproxNearestNeighborsFLANNDB_h
#include "ApproxNearestNeighborsDB.h"

#ifdef FLANN
#include <flann/flann.hpp>

#include <vector>
#include <unordered_map>

class ApproxNearestNeighborsFLANNDB : public ApproxNearestNeighborsDB
{
  public:

  int dim;
  int n_trees, n_checks_default;

  flann::Index<flann::L2<double>> flann_index;
  bool is_empty;
  std::vector<double *> ann_points;
  std::vector<std::vector<double>> ann_values;
  uint64_t ninsert;

  ApproxNearestNeighborsFLANNDB(int dim, int n_trees, int n_checks_default) :
    dim(dim),
    n_trees(n_trees),
    n_checks_default(n_checks_default),
    flann_index(flann::KDTreeIndexParams(n_trees)),
    is_empty(true),
    ninsert(0) { }

  void insert(
      std::vector<double> const &point,
      std::vector<double> const &value) override;

  int knn(
      std::vector<double> const &x,
      int k,
      std::vector<int> &ids,
      std::vector<double> &dists,
      std::vector<std::vector<double>> &points,
      std::vector<std::vector<double>> &values) override;

  private:

  int knn_helper(
      std::vector<double> const& x,
      int k,
      int n_checks,
      std::vector<int> &ids,
      std::vector<double> &dists,
      std::vector<std::vector<double>> &points,
      std::vector<std::vector<double>> &values);
};
#endif
#endif
