#ifndef included_ApproxNearestNeighborsFLANNDB_h
#define included_ApproxNearestNeighborsFLANNDB_h
#include "ApproxNearestNeighborsDB.h"

#ifdef FLANN
#include <flann/flann.hpp>

#include <chrono>
#include <vector>
#include <unordered_map>

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
