#ifndef included_SingletonDB_Redis_h
#define included_SingletonDB_Redis_h

#ifdef REDIS
#include "SingletonDB_Backend.h"

#include <vector>
#include <hiredis.h>

class SingletonDB_Redis : public SingletonDB_Backend{
 public:
 
  SingletonDB_Redis(int nArgs = 0, ...);
  ~SingletonDB_Redis();

  virtual void  push(const uint128_t &key, const std::vector<double>& buf, const unsigned long key_length);
  virtual void  erase(const uint128_t &key);
  virtual std::vector<double> pull(const uint128_t &key);
  virtual std::vector<double> pull_key(const uint128_t &key);

private:
  redisContext*   redis;
  FILE * redisServerHandle;
  FILE * nutcrackerServerHandle;
  char hostBuffer[256];

  redisReply *pull_data(const uint128_t &key);
};

#endif //REDIS

#endif // included_SingletonDB_Redis_h
