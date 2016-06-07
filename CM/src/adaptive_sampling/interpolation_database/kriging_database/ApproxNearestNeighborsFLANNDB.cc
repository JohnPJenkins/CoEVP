#include <algorithm>
#include "ApproxNearestNeighborsFLANNDB.h"

#ifdef FLANN

void ApproxNearestNeighborsFLANNDB::insert(
    std::vector<double> const &point,
    std::vector<double> const &value)
{
  assert(point.size() == static_cast<size_t>(dim));

  ninsert++;

  double * pdata = new double[dim];
  std::copy(point.begin(), point.end(), pdata);

  ann_points.push_back(pdata);
  ann_values.emplace_back(value);

  flann::Matrix<double> pts(pdata, 1, dim);
  if (is_empty) {
    flann_index.buildIndex(pts);
    is_empty = false;
  }
  else {
    flann_index.addPoints(pts,4.0);
  }
}

int
ApproxNearestNeighborsFLANNDB::knn_helper(
    std::vector<double> const& x,
    int k,
    int n_checks,
    std::vector<int> &ids,
    std::vector<double> &dists,
    std::vector<std::vector<double>> &points,
    std::vector<std::vector<double>> &values)
{
  if (is_empty) {
    ids.resize(0);
    dists.resize(0);
    points.resize(0);
    values.resize(0);
    return 0;
  }
  else {
    flann::Matrix<double> query(const_cast<double*>(x.data()), 1, dim);

    std::vector<std::vector<int>> indices;
    std::vector<std::vector<double>> mdists;

    flann_index.knnSearch(query, indices, mdists, k, flann::SearchParams(n_checks));

    int num_neighbors_found = indices[0].size();

    ids.resize(num_neighbors_found);
    dists.resize(num_neighbors_found);
    points.resize(num_neighbors_found);
    values.resize(num_neighbors_found);
    for (int i = 0; i < num_neighbors_found; i++) {
      int id = indices[0][i];
      double *dat = flann_index.getPoint(static_cast<size_t>(id));
      ids[i]  = id;
      points[i].resize(dim);
      std::copy(points[i].begin(), points[i].end(), dat);
      values[i] = ann_values[id];
      dists[i] = mdists[0][i];
    }
    return num_neighbors_found;
  }
}

int
ApproxNearestNeighborsFLANNDB::knn(
    std::vector<double> const& x,
    int k,
    std::vector<int> &ids,
    std::vector<double> &dists,
    std::vector<std::vector<double>> &points,
    std::vector<std::vector<double>> &values)
{
  return knn_helper(x, k, n_checks_default, ids, dists, points, values);
}

#endif
