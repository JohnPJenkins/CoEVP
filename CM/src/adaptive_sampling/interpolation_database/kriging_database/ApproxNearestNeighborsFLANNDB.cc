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
  time_point start;
  if (is_timing) start = now();

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

  if (is_timing) ins_times.push_back(now()-start);
}

int
ApproxNearestNeighborsFLANNDB::knn_helper(
    double const *x,
    int k,
    int n_checks,
    std::vector<size_t> &ids,
    std::vector<double> &dists,
    std::vector<std::vector<double>> &points,
    std::vector<std::vector<double>> &values)
{
  if (is_empty) {
    ids.resize(0);
    dists.resize(0);
    return 0;
  }
  else {
    flann::Matrix<double> query(const_cast<double*>(x), 1, dim);

    std::vector<std::vector<size_t>> indices;
    std::vector<std::vector<double>> mdists;
    // move ids and dists into the vector-of-vectors to prevent alloc in FLANN
    // (assuming caller has already allocated sufficient)
    indices.emplace_back(std::move(ids));
    mdists.emplace_back(std::move(dists));

    flann_index.knnSearch(query, indices, mdists, k, flann::SearchParams(n_checks));

    int num_neighbors_found = indices[0].size();

    points.resize(num_neighbors_found);
    values.resize(num_neighbors_found);
    // move ids and dists back
    ids = std::move(indices[0]);
    dists = std::move(mdists[0]);
    for (int i = 0; i < num_neighbors_found; i++) {
      const int id = ids[i];
      const double *dat = flann_index.getPoint(static_cast<size_t>(id));
      const size_t num_values = ann_pvs[id].num_values;
      points[i].resize(dim);
      std::copy(dat, dat+dim, points[i].begin());
      values[i].resize(num_values);
      std::copy(dat+dim, dat+dim+num_values, values[i].begin());
    }
    return num_neighbors_found;
  }
}

int
ApproxNearestNeighborsFLANNDB::knn(
    double const *x,
    int k,
    std::vector<size_t> &ids,
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

ApproxNearestNeighborsDB::knnRet
ApproxNearestNeighborsFLANNDB::knn(
    double const * x,
    int k,
    size_t *ids,
    double *dists,
    double *point_buf,
    double *value_buf,
    size_t value_size_avail,
    size_t *value_offsets)
{
  ApproxNearestNeighborsDB::knnRet ret;
  time_point start;
  if (is_timing) start = now();

  if (is_empty) { ret.k = 0; ret.overflow = false; }
  else {
    flann::Matrix<double> query(const_cast<double*>(x), 1, dim);
    flann::Matrix<size_t> ids_m(ids, 1, k);
    flann::Matrix<double> dists_m(dists, 1, k);

    ret.k = flann_index.knnSearch(query, ids_m, dists_m, k,
        flann::SearchParams(n_checks_default));
    ret.overflow = false;

    double *vbuf = value_buf;
    value_offsets[0] = 0;
    size_t total_values = 0;
    for (int i = 0; i < ret.k; i++) {
      const size_t id = ids[i];
      const size_t vsize = ann_pvs[id].num_values;
      total_values += vsize;
      if (total_values <= value_size_avail) {
        const double *pdat = flann_index.getPoint(static_cast<size_t>(id));
        const double *vdat = pdat+dim;
        std::copy(pdat, vdat, point_buf+i*dim);
        std::copy(vdat, vdat+vsize, vbuf);
        vbuf += vsize;
        value_offsets[i+1] = value_offsets[i] + vsize;
      }
      else {
        ret.overflow = true;
      }
    }
    if (ret.overflow) value_offsets[0] = total_values;
  }

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
