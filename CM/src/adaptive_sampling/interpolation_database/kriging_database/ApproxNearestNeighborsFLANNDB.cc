/*
 * Copyright (c) 2016 UChicago Argonne, LLC
 *
 * See COPYRIGHT-ANL in top-level directory.
 */

#include <fstream>
#include <iostream>
#include <algorithm>
#include "ApproxNearestNeighborsFLANNDB.h"

#ifdef FLANN

namespace {
  auto now = &std::chrono::steady_clock::now;
  using time_point = std::chrono::steady_clock::time_point;
  using dur = std::chrono::duration<double>;
} /* namespace */

ApproxNearestNeighborsFLANNDB::~ApproxNearestNeighborsFLANNDB()
{
  if (is_timing) {
    std::cout <<
      "## FLANNDB INSERT TIMES (total: " << ins_times.size() << ")" << std::endl;
    for (size_t i = 0; i < ins_times.size(); i++) {
      std::cout << ins_times[i].count() << std::endl;
    }
    std::cout <<
      "## FLANNDB KNN TIMES (total: " << knn_times.size() << ")" << std::endl;
    for (size_t i = 0; i < knn_times.size(); i++) {
      std::cout << knn_times[i].count() << std::endl;
    }
    std::cout << "## DONE" << std::endl;
  }
}

void ApproxNearestNeighborsFLANNDB::insert(
    double const * point,
    double const * value,
    size_t num_values)
{
  double *pvdata = new double[dim+num_values];
  std::copy(point, point+dim, pvdata);
  std::copy(value, value+num_values, pvdata+dim);

  // this invocation moves the temporaty unique ptr
  ann_pvs.emplace_back(pvpair{std::unique_ptr<double[]>(pvdata), num_values});

  flann::Matrix<double> pts(pvdata, 1, dim);
  if (is_empty) {
    flann_index.buildIndex(pts);
    is_empty = false;
  }
  else {
    flann_index.addPoints(pts,4.0);
  }

}

void ApproxNearestNeighborsFLANNDB::insert(
    std::vector<double> const &point,
    std::vector<double> const &value)
{
  time_point start;
  if (is_timing) start = now();

  assert(point.size() == static_cast<size_t>(dim));

  insert(point.data(), value.data(), value.size());

  if (is_timing) ins_times.push_back(now()-start);
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
      const double *dat = flann_index.getPoint(static_cast<size_t>(id));
      const size_t num_values = ann_pvs[id].num_values;
      ids[i]  = id;
      points[i].resize(dim);
      std::copy(dat, dat+dim, points[i].begin());
      values[i].resize(num_values);
      std::copy(dat+dim, dat+dim+num_values, values[i].begin());
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
  int ret;
  time_point start;
  if (is_timing) start = now();
  ret = knn_helper(x, k, n_checks_default, ids, dists, points, values);
  if (is_timing) knn_times.push_back(now()-start);
  return ret;
}

void
ApproxNearestNeighborsFLANNDB::dump(
    const std::string &filename)
{
  std::ofstream of(filename,
      std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
  assert(of.is_open() && of.good());
  // print header, of the format:
  // <64-bit point dimension> <64-bit value dimension>
  // NOTE: assuming all values have the same dimension
  uint64_t header[2];
  header[0] = dim; header[1] = ann_pvs[0].num_values;
  of.write(reinterpret_cast<const char*>(header), sizeof header);
  // print point/values
  for (auto pv_iter = ann_pvs.cbegin(); pv_iter != ann_pvs.cend(); ++pv_iter) {
    std::cout << "point: ";
    const auto pp = pv_iter->pv.get();
    for (size_t i = 0; i < (size_t)dim; i++) {
      const double p = pp[i];
      std::cout << p << " ";
      of.write(reinterpret_cast<const char*>(&p), sizeof p);
      assert(of.good());
    }
    std::cout << std::endl;
    for (size_t i = (size_t)dim; i < dim+pv_iter->num_values; i++) {
      const double v = pp[i];
      of.write(reinterpret_cast<const char*>(&v), sizeof v);
      assert(of.good());
    }
  }
}

#endif
