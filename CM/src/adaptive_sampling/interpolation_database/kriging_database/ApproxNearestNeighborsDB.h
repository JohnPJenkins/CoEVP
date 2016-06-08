#ifndef included_ApproxNearestNeighborsDB_h
#define included_ApproxNearestNeighborsDB_h

#include <vector>

class ApproxNearestNeighborsDB
{
  public:

  virtual void insert(
      std::vector<double> const &point,
      std::vector<double> const &value) = 0;

  virtual int knn(
      std::vector<double> const &x,
      int k,
      std::vector<int> &ids,
      std::vector<double> &dists,
      std::vector<std::vector<double>> &points,
      std::vector<std::vector<double>> &values) = 0;

};

#endif