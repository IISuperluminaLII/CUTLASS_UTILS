/***************************************************************************************************
 * Copyright (c) 2025 NVIDIA CORPORATION.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Runtime probe that mirrors the SM120 backward kernel's TMEM->register staging path and records the
 * linear TMEM indices each compute-lane touches.  Duplicate indices in the output highlight the
 * race-condition that `cute::copy` flags during compilation.
 **************************************************************************************************/

#include <cuda_runtime.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cute/atom/copy_traits.hpp"
#include "cute/algorithm/tuple_algorithms.hpp"

#include "sm120/prefill/dense/collective/fmha_fusion.hpp"
#include "sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "sm120/prefill/dense/kernel/sm120_fmha_bwd_kernel_tma_warpspecialized.hpp"

namespace sm120_index_probe {

using KernelTraits = flash::Sm120WorkstationConfig;
using ProblemShape = cute::tuple<int, int, int, int, cute::tuple<int, int>>;
using Element = cutlass::bfloat16_t;
using ElementAcc = float;
using TileShape = typename KernelTraits::TileShapeFmhaBwd;
using Mask = cutlass::fmha::collective::ResidualMask;

using Kernel = cutlass::fmha::kernel::Sm120FmhaBwdKernelTmaWarpSpecialized<
    KernelTraits, ProblemShape, Element, ElementAcc, TileShape, Mask>;

constexpr int kComputeThreads =
    Kernel::kNumReduceWarps * cutlass::NumThreadsPerWarp;
constexpr int kMaxElementsPerThread = 1024;  // generous upper bound

#define CUDA_CHECK(cmd)                                                             \
  do {                                                                              \
    cudaError_t status = (cmd);                                                     \
    if (status != cudaSuccess) {                                                    \
      std::cerr << "CUDA error: " << cudaGetErrorString(status) << " (" << __LINE__ \
                << ")\n";                                                           \
      std::exit(EXIT_FAILURE);                                                      \
    }                                                                               \
  } while (0)

__global__ void probe_dq_indices(int* src, int* dst, int stride, int* counts,
                                 int* coord0, int* coord1, int* coord2) {
  using namespace cute;

  auto frag_builder =
      partition_fragment_C(typename Kernel::TiledMmaDSK{},
                           select<0, 1>(typename Kernel::TileShapeDSK{}));
  auto tDQtDQ = frag_builder(make_coord(_, _), _0{}, _0{});
  tDQtDQ.data() = Kernel::TmemAllocation::kDQ;

  auto cDQ = make_identity_tensor(take<0, 2>(typename Kernel::TileShapeDSK{}));

  int t = threadIdx.x;
  int lane = t % cutlass::NumThreadsPerWarp;
  if (t >= kComputeThreads) {
    return;
  }

  using CopyAtom = cute::Copy_Atom<cute::UniversalCopy<uint128_t>, ElementAcc>;
  auto copy_op = make_cotiled_copy(CopyAtom{}, tDQtDQ.layout(), tDQtDQ.layout());
  auto thread_copy = copy_op.get_slice(t);

  auto tTR_tDQ = thread_copy.partition_S(tDQtDQ);
  auto tTR_cDQ = thread_copy.partition_D(cDQ);
  auto tTR_rDQ = make_tensor<ElementAcc>(shape(tTR_cDQ));
  constexpr int kRank = decltype(rank(tTR_cDQ))::value;
  static_assert(kRank >= 1, "DQ tensor rank must be >= 1 for this probe.");

  int elements = static_cast<int>(size(tTR_tDQ));
  counts[t] = elements;

  int warp = t / cutlass::NumThreadsPerWarp;
  int per_warp = elements / Kernel::kNumReduceWarps;
  for (int i = warp * per_warp + lane;
       i < (warp + 1) * per_warp && i < stride;
       i += cutlass::NumThreadsPerWarp) {
    auto src_coord = tTR_tDQ.get_1d_coord(i);
    auto dst_coord = tTR_rDQ.get_1d_coord(i);

    int src_offset = static_cast<int>(tTR_tDQ.layout()(src_coord));
    int dst_offset = static_cast<int>(tTR_rDQ.layout()(dst_coord));

    int coord_vals[3] = {-1, -1, -1};
    int coord_idx = 0;
    cute::for_each(dst_coord, [&](auto const& val) {
      if (coord_idx < 3) {
        coord_vals[coord_idx] = static_cast<int>(val);
      }
      ++coord_idx;
    });
    coord0[t * stride + i] = coord_vals[0];
    coord1[t * stride + i] = coord_vals[1];
    coord2[t * stride + i] = coord_vals[2];

    src[t * stride + i] = src_offset;
    dst[t * stride + i] = dst_offset;
  }
}

struct CollisionReport {
  std::map<int, std::vector<std::string>> src_hits;
  std::map<int, std::vector<std::string>> dst_hits;
  int total_elements = 0;
  bool truncated = false;
};

CollisionReport analyse(const std::vector<int>& src,
                        const std::vector<int>& dst,
                        const std::vector<int>& counts,
                        const std::vector<int>& coord0,
                        const std::vector<int>& coord1,
                        const std::vector<int>& coord2,
                        int stride) {
  CollisionReport report{};

  for (int thread = 0; thread < kComputeThreads; ++thread) {
    int elements = counts[thread];
    if (elements < 0) {
      continue;
    }
    if (elements > stride) {
      report.truncated = true;
      elements = stride;
    }
    report.total_elements += elements;

    int warp = thread / cutlass::NumThreadsPerWarp;
    int lane = thread % cutlass::NumThreadsPerWarp;

    for (int i = 0; i < elements; ++i) {
      int src_offset = src[thread * stride + i];
      int dst_offset = dst[thread * stride + i];
      std::string info = "warp=" + std::to_string(warp) + "/lane=" +
                         std::to_string(lane) + "/elem=" + std::to_string(i) +
                         "/dst=" + std::to_string(dst_offset) + "/coord=" +
                         "(" + std::to_string(coord0[thread * stride + i]) +
                         "," + std::to_string(coord1[thread * stride + i]) +
                         "," + std::to_string(coord2[thread * stride + i]) +
                         ")";

      report.src_hits[src_offset].push_back(info);
      report.dst_hits[dst_offset].push_back(info);
    }
  }

  return report;
}

template <class Map>
bool dump_collisions(const char* label, const Map& hits, int max_examples = 8) {
  bool ok = true;
  for (auto const& kv : hits) {
    if (kv.second.size() < 2) {
      continue;
    }
    if (ok) {
      std::cout << "---- " << label << " collisions ----\n";
    }
    ok = false;
    std::cout << "index " << kv.first << " hit " << kv.second.size()
              << " times\n";
    int printed = 0;
    for (auto const& info : kv.second) {
      if (printed++ >= max_examples) {
        std::cout << "  ... +" << (kv.second.size() - max_examples)
                  << " more\n";
        break;
      }
      std::cout << "  " << info << "\n";
    }
  }
  return ok;
}

}  // namespace sm120_index_probe

int main() {
  using namespace sm120_index_probe;

  std::vector<int> host_src(kComputeThreads * kMaxElementsPerThread, -1);
  std::vector<int> host_dst(kComputeThreads * kMaxElementsPerThread, -1);
  std::vector<int> host_counts(kComputeThreads, -1);
  std::vector<int> host_coord0(kComputeThreads * kMaxElementsPerThread, -1);
  std::vector<int> host_coord1(kComputeThreads * kMaxElementsPerThread, -1);
  std::vector<int> host_coord2(kComputeThreads * kMaxElementsPerThread, -1);

  int* dev_src = nullptr;
  int* dev_dst = nullptr;
  int* dev_counts = nullptr;
  int* dev_coord0 = nullptr;
  int* dev_coord1 = nullptr;
  int* dev_coord2 = nullptr;

  CUDA_CHECK(cudaMalloc(&dev_src,
                        host_src.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dev_dst,
                        host_dst.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dev_counts,
                        host_counts.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dev_coord0,
                        host_coord0.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dev_coord1,
                        host_coord1.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dev_coord2,
                        host_coord2.size() * sizeof(int)));

  CUDA_CHECK(cudaMemset(dev_src, 0xFF, host_src.size() * sizeof(int)));
  CUDA_CHECK(cudaMemset(dev_dst, 0xFF, host_dst.size() * sizeof(int)));
  CUDA_CHECK(cudaMemset(dev_counts, 0xFF, host_counts.size() * sizeof(int)));
  CUDA_CHECK(cudaMemset(dev_coord0, 0xFF, host_coord0.size() * sizeof(int)));
  CUDA_CHECK(cudaMemset(dev_coord1, 0xFF, host_coord1.size() * sizeof(int)));
  CUDA_CHECK(cudaMemset(dev_coord2, 0xFF, host_coord2.size() * sizeof(int)));

  probe_dq_indices<<<1, kComputeThreads>>>(dev_src, dev_dst, kMaxElementsPerThread,
                               dev_counts, dev_coord0, dev_coord1, dev_coord2);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(host_src.data(), dev_src,
                        host_src.size() * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_dst.data(), dev_dst,
                        host_dst.size() * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_counts.data(), dev_counts,
                        host_counts.size() * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_coord0.data(), dev_coord0,
                        host_coord0.size() * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_coord1.data(), dev_coord1,
                        host_coord1.size() * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_coord2.data(), dev_coord2,
                        host_coord2.size() * sizeof(int),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(dev_src));
  CUDA_CHECK(cudaFree(dev_dst));
  CUDA_CHECK(cudaFree(dev_counts));
  CUDA_CHECK(cudaFree(dev_coord0));
  CUDA_CHECK(cudaFree(dev_coord1));
  CUDA_CHECK(cudaFree(dev_coord2));

  auto report = analyse(host_src, host_dst, host_counts,
                        host_coord0, host_coord1, host_coord2,
                        kMaxElementsPerThread);

  if (report.truncated) {
    std::cout << "Warning: some per-thread element lists were truncated. "
                 "Increase kMaxElementsPerThread for full coverage.\n";
  }

  std::cout << "Total recorded elements: " << report.total_elements
            << ", unique TMEM slots: " << report.src_hits.size()
            << ", unique register slots: " << report.dst_hits.size()
            << "\n";

  bool src_ok = dump_collisions("TMEM", report.src_hits);
  bool dst_ok = dump_collisions("register", report.dst_hits);

  if (src_ok && dst_ok) {
    std::cout << "No index collisions detected.\n";
    return 0;
  }

  std::cout << "Index collisions detected. Inspect the collision log for the "
               "responsible warp/lane pairs.\n";
  return 1;
}
template <class Coord>
std::string coord_to_string(Coord const& coord) {
  std::ostringstream oss;
  oss << "(";
  bool first = true;
  cute::for_each(coord, [&](auto const& value) {
    if (!first) {
      oss << ",";
    }
    first = false;
    oss << static_cast<int>(value);
  });
  oss << ")";
  return oss.str();
}
