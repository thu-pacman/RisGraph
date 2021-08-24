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
#include <thread>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <immintrin.h>
#include <omp.h>
#include <libaio.h>
#include "core/type.hpp"
#include "core/graph.hpp"
#include "core/io.hpp"

#if !defined(BFS) && !defined(WCC) && !defined(SSSP) && !defined(SSWP)
    #error Not supported algorithm.
#endif
#if defined(BFS) || defined(SSSP) || defined(SSWP)
    constexpr bool directed = true;
#elif defined(WCC)
    constexpr bool directed = false;
#endif

int main(int argc, char** argv)
{
    if(argc <= 3)
    {
        fprintf(stderr, "usage: %s graph root percent\n", argv[0]);
        exit(1);
    }
    std::pair<uint64_t, uint64_t> *raw_edges, *add_raw_edges, *del_raw_edges;
    uint64_t raw_edges_len, add_raw_edges_len, del_raw_edges_len;
    std::tie(raw_edges, raw_edges_len) = mmap_binary(argv[1]);
    uint64_t root = std::stoull(argv[2]);
    double imported_rate = std::stod(argv[3]);
    uint64_t imported_edges = raw_edges_len*imported_rate;

    if(argc > 5)
    {
        std::tie(add_raw_edges, add_raw_edges_len) = mmap_binary(argv[4]);
        std::tie(del_raw_edges, del_raw_edges_len) = mmap_binary(argv[5]);
    }
    else
    {
        add_raw_edges = raw_edges + imported_edges;
        add_raw_edges_len = raw_edges_len*0.1;
        del_raw_edges = raw_edges;
        del_raw_edges_len = raw_edges_len*0.1;
    }

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
        #pragma omp parallel for
        for(uint64_t i=0;i<add_raw_edges_len;i++)
        {
            const auto &e = add_raw_edges[i];
            write_max(&num_vertices, e.first+1);
            write_max(&num_vertices, e.second+1);
        }
        #pragma omp parallel for
        for(uint64_t i=0;i<del_raw_edges_len;i++)
        {
            const auto &e = del_raw_edges[i];
            write_max(&num_vertices, e.first+1);
            write_max(&num_vertices, e.second+1);
        }
        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "read: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
        fprintf(stderr, "|E|=%lu\n", raw_edges_len);
    }
#if defined(BFS)
    Graph<void> graph(num_vertices, raw_edges_len, false, true);
#elif defined(WCC)
    Graph<void> graph(num_vertices, raw_edges_len, true, false);
#elif defined(SSSP) || defined(SSWP)
    Graph<uint64_t> graph(num_vertices, raw_edges_len, false, true);
#endif

    {
        auto start = std::chrono::system_clock::now();
        #pragma omp parallel for
        for(uint64_t i=0;i<imported_edges;i++)
        {
            const auto &e = raw_edges[i];
#if defined(BFS) || defined(WCC)
            graph.add_edge({e.first, e.second}, directed);
#elif defined(SSSP) || defined(SSWP)
            graph.add_edge({e.first, e.second, (e.first+e.second)%128}, directed);
#endif
        }
        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "add: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }

    auto labels = graph.alloc_vertex_tree_array<uint64_t>();
#if defined(BFS)
    const uint64_t MAXL = 65536;
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
        return std::make_pair(src_data+1 < dst_data, src_data + 1);
    };
    auto active_result_func = [](uint64_t old_result, uint64_t src, uint64_t dst, uint64_t src_data, uint64_t old_dst_data, uint64_t new_dst_data) -> uint64_t
    {
        return old_result+1;
    };
    auto equal_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, decltype(graph)::adjedge_type adjedge) -> bool
    {
        return src_data + 1 == dst_data;
    };
    auto init_label_func = [=](uint64_t vid) -> std::pair<uint64_t, bool>
    {
        return {vid==root?0:MAXL, vid==root};
    };
#elif defined(WCC)
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
    std::vector<uint64_t> inits(num_vertices);
    for(uint64_t i=0;i<num_vertices;i++) inits[i] = i;
    std::random_shuffle(inits.begin(), inits.end());
    auto init_label_func = [&inits](uint64_t vid) -> std::pair<uint64_t, bool>
    {
        return {inits[vid], true};
    };
#elif defined(SSSP)
    const uint64_t MAXL = 134217728;
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
        return std::make_pair(src_data+adjedge.data < dst_data, src_data + adjedge.data);
    };
    auto active_result_func = [](uint64_t old_result, uint64_t src, uint64_t dst, uint64_t src_data, uint64_t old_dst_data, uint64_t new_dst_data) -> uint64_t
    {
        return old_result+1;
    };
    auto equal_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, decltype(graph)::adjedge_type adjedge) -> bool
    {
        return src_data + adjedge.data == dst_data;
    };
    auto init_label_func = [=](uint64_t vid) -> std::pair<uint64_t, bool>
    {
        return {vid==root?0:MAXL, vid==root};
    };
#elif defined(SSWP)
    const uint64_t MAXL = 134217728;
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
        return std::make_pair(std::min(src_data, adjedge.data) > dst_data, std::min(src_data, adjedge.data));
    };
    auto active_result_func = [](uint64_t old_result, uint64_t src, uint64_t dst, uint64_t src_data, uint64_t old_dst_data, uint64_t new_dst_data) -> uint64_t
    {
        return old_result+1;
    };
    auto equal_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, decltype(graph)::adjedge_type adjedge) -> bool
    {
        return std::min(src_data, adjedge.data) == dst_data;
    };
    auto init_label_func = [=](uint64_t vid) -> std::pair<uint64_t, bool>
    {
        return {vid==root?MAXL:0, vid==root};
    };
#endif

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

    uint64_t unsafe_add_size = 0, unsafe_del_size = 0;
    {
        auto start = std::chrono::system_clock::now();
        uint64_t add = 0, del = 0;
        for(bool is_add = true; add < add_raw_edges_len || del < del_raw_edges_len; is_add = !is_add)
        {
            if(is_add)
            {
                if(add >= add_raw_edges_len) continue;
                const auto &e = add_raw_edges[add];
#if defined(BFS) || defined(WCC)
                decltype(graph)::edge_type edge = {e.first, e.second};
#elif defined(SSSP) || defined(SSWP)
                decltype(graph)::edge_type edge = {e.first, e.second, (e.first+e.second)%128};
#endif
                if(graph.add_edge(edge, directed) == 0 && graph.need_update_tree_add(update_func, labels, edge, directed))
                {
                    unsafe_add_size++;
                    graph.update_tree_add<uint64_t, uint64_t>(
                        continue_reduce_func,
                        update_func,
                        active_result_func,
                        labels, edge, directed
                    );
                }
                add++;
            }
            else
            {
                if(del >= del_raw_edges_len) continue;
                const auto &e = del_raw_edges[del];
#if defined(BFS) || defined(WCC)
                decltype(graph)::edge_type edge = {e.first, e.second};
#elif defined(SSSP) || defined(SSWP)
                decltype(graph)::edge_type edge = {e.first, e.second, (e.first+e.second)%128};
#endif
                if(graph.del_edge(edge, directed) == 1 && graph.need_update_tree_del(labels, edge, directed))
                {
                    unsafe_del_size++;
                    graph.update_tree_del<uint64_t, uint64_t>(
                        init_label_func,
                        continue_reduce_func,
                        update_func,
                        active_result_func,
                        equal_func,
                        labels, edge, directed
                    );
                }
                del++;
            }
        }
        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "exec: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }

    fprintf(stderr, "total add: %lu\n", add_raw_edges_len);
    fprintf(stderr, "total del: %lu\n", del_raw_edges_len);
    fprintf(stderr, "unsafe add: %lu\n", unsafe_add_size);
    fprintf(stderr, "unsafe del: %lu\n", unsafe_del_size);
    fprintf(stderr, "unsafe add rate: %lf\n", (double)unsafe_add_size/add_raw_edges_len);
    fprintf(stderr, "unsafe del rate: %lf\n", (double)unsafe_del_size/del_raw_edges_len);
    fprintf(stderr, "unsafe rate: %lf\n", (double)(unsafe_add_size+unsafe_del_size)/(add_raw_edges_len+del_raw_edges_len));

    return 0;
}
