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

  virtual void insert(
      std::vector<double> const &point,
      std::vector<double> const &value) = 0;

  virtual int knn(
      std::vector<double> const &x,
      int k,
      std::vector<size_t> &ids,
      std::vector<double> &dists,
      std::vector<std::vector<double>> &points,
      std::vector<std::vector<double>> &values) = 0;

  // optional
  virtual void dump(const std::string &filename) { }

  virtual ~ApproxNearestNeighborsDB() {};
};

#endif
