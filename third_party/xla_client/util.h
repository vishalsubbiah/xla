#ifndef TENSORFLOW_COMPILER_XLA_RPC_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_RPC_UTIL_H_

#include <memory>
#include <vector>

#include "tensorflow/core/lib/gtl/array_slice.h"

namespace xla {
namespace util {

template <typename T>
std::vector<const T*> GetConstSharedPointers(
    tensorflow::gtl::ArraySlice<const std::shared_ptr<T>> shared_pointers) {
  std::vector<const T*> pointers;
  pointers.reserve(shared_pointers.size());
  for (auto& shared_pointer : shared_pointers) {
    pointers.push_back(shared_pointer.get());
  }
  return pointers;
}

template <typename T>
std::vector<T*> GetSharedPointers(
    tensorflow::gtl::ArraySlice<const std::shared_ptr<T>> shared_pointers) {
  std::vector<T*> pointers;
  pointers.reserve(shared_pointers.size());
  for (auto& shared_pointer : shared_pointers) {
    pointers.push_back(shared_pointer.get());
  }
  return pointers;
}

}  // namespace util
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RPC_UTIL_H_
