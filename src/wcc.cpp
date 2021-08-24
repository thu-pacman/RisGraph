/* Copyright 2020 Guanyu Feng, Tsinghua University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdio>
#include <cstdint>
#include <cassert>
#include <string>
#include <vector>
#include <utility>
#include <chrono>
#include "core/type.hpp"
#include "core/graph.hpp"
#include "core/atomic.hpp"
#include "core/io.hpp"

int main(int argc, char** argv)
{
    assert(argc > 1);
    std::pair<uint64_t, uint64_t> *raw_edges;
    uint64_t raw_edges_len;
    std::tie(raw_edges, raw_edges_len) = mmap_binary(argv[1]);
    uint64_t num_vertices = 0;
    {
        auto start = std::chrono::system_clock::now();
        #pragma omp parallel for
        for(uint64_t i=0;i<raw_edges_len;i++)
        {
            const auto &e = raw_edges[i];
            write_max(&num_vertices, e.first+1);
            write_max(&num_vertices, e.second+1);
        }
        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "read: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
        fprintf(stderr, "|E|=%lu\n", raw_edges_len);
    }
    Graph<void> graph(num_vertices, raw_edges_len, true, true);
    {
        auto start = std::chrono::system_clock::now();
        #pragma omp parallel for
        for(uint64_t i=0;i<raw_edges_len;i++)
        {
            const auto &e = raw_edges[i];
            graph.add_edge({e.first, e.second}, false);
        }
        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "add: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }

    auto labels = graph.alloc_vertex_tree_array<uint64_t>();
    auto continue_reduce_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t>
    {
        return std::make_pair(local_result>0, total_result+local_result);
    };
    auto continue_reduce_print_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t>
    {
        fprintf(stderr, "active(%lu) >= %lu\n", depth, local_result);
        return std::make_pair(local_result>0, total_result+local_result);
    };
    auto update_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, decltype(graph)::adjedge_type adjedge) -> std::pair<bool, uint64_t>
    {
        return std::make_pair(src_data < dst_data, src_data);
    };
    auto active_result_func = [](uint64_t old_result, uint64_t src, uint64_t dst, uint64_t src_data, uint64_t old_dst_data, uint64_t new_dst_data) -> uint64_t
    {
        return old_result+1;
    };
    auto equal_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, decltype(graph)::adjedge_type adjedge) -> bool
    {
        return src_data == dst_data;
    };
    auto init_label_func = [](uint64_t vid) -> std::pair<uint64_t, bool>
    {
        return {vid, true};
    };

    {
        auto start = std::chrono::system_clock::now();

        graph.build_tree<uint64_t, uint64_t>(
            init_label_func, 
            continue_reduce_print_func,
            update_func,
            active_result_func,
            labels
        );

        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "exec: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }

    {
        auto num_wcces = graph.stream_vertices<uint64_t>(
            [&](uint64_t vid)
            {
                return labels[vid].data == vid;
            },
            graph.get_dense_active_all()
        );
        fprintf(stderr, "number of communities: %lu\n", num_wcces);
    }

    for(uint64_t i=0;i<num_vertices;i++)
    {
        printf("%lu %lu\n", i, labels[i].data);
    }
    return 0;
}
