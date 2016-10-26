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
