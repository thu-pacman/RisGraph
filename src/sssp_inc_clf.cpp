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
#include <fcntl.h>
#include <unistd.h>
#include <immintrin.h>
#include <omp.h>
#include "core/type.hpp"
#include "core/graph.hpp"
#include "core/io.hpp"
#include <tbb/parallel_sort.h>

int main(int argc, char** argv)
{
    if(argc <= 3)
    {
        fprintf(stderr, "usage: %s graph root imported_rate\n", argv[0]);
        exit(1);
    }
    std::pair<uint64_t, uint64_t> *raw_edges;
    uint64_t root = std::stoull(argv[2]);
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
    Graph<uint64_t> graph(num_vertices, raw_edges_len, false, true);
    //std::random_shuffle(raw_edges.begin(), raw_edges.end());
    double imported_rate = std::stod(argv[3]);
    uint64_t imported_edges = raw_edges_len*imported_rate;
    {
        auto start = std::chrono::system_clock::now();
        #pragma omp parallel for
        for(uint64_t i=0;i<imported_edges;i++)
        {
            const auto &e = raw_edges[i];
            graph.add_edge({e.first, e.second, (e.first+e.second)%128}, true);
        }
        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "add: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }

    auto labels = graph.alloc_vertex_tree_array<uint64_t>();
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

    std::atomic_uint64_t add_edge_len(0), del_edge_len(0);
    std::vector<decltype(graph)::edge_type> add_edge, del_edge;
    std::mutex add_mutex, del_mutex;
    {
        auto start = std::chrono::system_clock::now();
        #pragma omp parallel for
        for(uint64_t i=imported_edges;i<raw_edges_len;i++)
        {
            {
                const auto &e = raw_edges[i];
                if(graph.need_update_tree_add(update_func, labels, {e.first, e.second, (e.first+e.second)%128}, true))
                {
                    atomic_append(add_edge, add_edge_len, {e.first, e.second, (e.first+e.second)%128}, add_mutex);
                }
                else
                {
                    graph.add_edge({e.first, e.second, (e.first+e.second)%128}, true);
                }
            }

            {
                const auto &e = raw_edges[i-imported_edges];
                if(graph.need_update_tree_del(labels, {e.first, e.second, (e.first+e.second)%128}, true))
                {
                    atomic_append(del_edge, del_edge_len, {e.first, e.second, (e.first+e.second)%128}, del_mutex);
                }
                else
                {
                    graph.del_edge({e.first, e.second, (e.first+e.second)%128}, true);
                }
            }
        };
        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "exec safe: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
        auto cmp = [](const decltype(graph)::edge_type &a, const decltype(graph)::edge_type &b)
        {
            if(a.src == b.src)
                return a.dst < b.dst;
            else
                return a.src < b.src;
        };
        tbb::parallel_sort(add_edge.begin(), add_edge.begin()+add_edge_len.load(), cmp);
        tbb::parallel_sort(del_edge.begin(), del_edge.begin()+del_edge_len.load(), cmp);
        fprintf(stderr, "unsafe_add: %lu\n", add_edge_len.load());
        fprintf(stderr, "unsafe_del: %lu\n", del_edge_len.load());
    }

    std::vector<uint64_t> latency;
    {
        auto start = std::chrono::system_clock::now();
        uint64_t min_len = std::min(add_edge_len.load(), del_edge_len.load());
        for(uint64_t i=0;i<min_len;i++)
        {
            {
                auto start = std::chrono::system_clock::now();
                const auto &e = add_edge[i];
                if(graph.add_edge(e, true) == 0)
                {
                    graph.update_tree_add<uint64_t, uint64_t>(
                        continue_reduce_func,
                        update_func,
                        active_result_func,
                        labels, e, true
                    );
                }
                auto end = std::chrono::system_clock::now();
                //if(end-start > std::chrono::milliseconds(5))
                //{
                //    fprintf(stderr, "add %ld->%ld, %lfms\n", e.src, e.dst, 1e-3*std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
                //}
                latency.emplace_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
            }

            {
                auto start = std::chrono::system_clock::now();
                const auto &e = del_edge[i];
                if(graph.del_edge(e, true) == 1)
                {
                    graph.update_tree_del<uint64_t, uint64_t>(
                        init_label_func,
                        continue_reduce_func,
                        update_func,
                        active_result_func,
                        equal_func,
                        labels, e, true
                    );
                }
                auto end = std::chrono::system_clock::now();
                //if(end-start > std::chrono::milliseconds(5))
                //{
                //    fprintf(stderr, "del %ld->%ld, %lfms\n", e.src, e.dst, 1e-3*std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
                //}
                latency.emplace_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
            }
        }
        for(uint64_t i=min_len;i<add_edge_len.load();i++)
        {
            {
                auto start = std::chrono::system_clock::now();
                const auto &e = add_edge[i];
                if(graph.add_edge(e, true) == 0)
                {
                    graph.update_tree_add<uint64_t, uint64_t>(
                        continue_reduce_func,
                        update_func,
                        active_result_func,
                        labels, e, true
                    );
                }
                auto end = std::chrono::system_clock::now();
                //if(end-start > std::chrono::milliseconds(5))
                //{
                //    fprintf(stderr, "add %ld->%ld, %lfms\n", e.src, e.dst, 1e-3*std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
                //}
                latency.emplace_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
            }
        }

        for(uint64_t i=min_len;i<del_edge_len.load();i++)
        {
            {
                auto start = std::chrono::system_clock::now();
                const auto &e = del_edge[i];
                if(graph.del_edge(e, true) == 1)
                {
                    graph.update_tree_del<uint64_t, uint64_t>(
                        init_label_func,
                        continue_reduce_func,
                        update_func,
                        active_result_func,
                        equal_func,
                        labels, e, true
                    );
                }
                auto end = std::chrono::system_clock::now();
                //if(end-start > std::chrono::milliseconds(5))
                //{
                //    fprintf(stderr, "add %ld->%ld, %lfms\n", e.src, e.dst, 1e-3*std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
                //}
                latency.emplace_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
            }
        }
        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "exec unsafe: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }

    std::sort(latency.begin(), latency.end());
    uint64_t sum = 0;
    for(auto t:latency) sum += t;
    fprintf(stderr, "size:      %lu\n", latency.size());
    fprintf(stderr, "total:     %lfus\n", 1e-3*sum);
    fprintf(stderr, "mean:      %lfus\n", 1e-3*sum/latency.size());
    fprintf(stderr, "p25:       %lfus\n", 1e-3*latency[latency.size()*0.25]);
    fprintf(stderr, "p50:       %lfus\n", 1e-3*latency[latency.size()*0.50]);
    fprintf(stderr, "p75:       %lfus\n", 1e-3*latency[latency.size()*0.75]);
    fprintf(stderr, "p99:       %lfus\n", 1e-3*latency[latency.size()*0.99]);
    fprintf(stderr, "p999:      %lfus\n", 1e-3*latency[latency.size()*0.999]);
    fprintf(stderr, "p9999:     %lfus\n", 1e-3*latency[latency.size()*0.9999]);
    fprintf(stderr, "p99999:    %lfus\n", 1e-3*latency[latency.size()*0.99999]);
    fprintf(stderr, "p999999:   %lfus\n", 1e-3*latency[latency.size()*0.999999]);
    fprintf(stderr, "p9999999:  %lfus\n", 1e-3*latency[latency.size()*0.9999999]);
    fprintf(stderr, "p99999999: %lfus\n", 1e-3*latency[latency.size()*0.99999999]);
    fprintf(stderr, "max:       %lfus\n", 1e-3*latency[latency.size()-1]);

    {
        std::vector<std::atomic_uint64_t> layer_counts(MAXL);
        for(auto &a : layer_counts) a = 0;
        graph.stream_vertices<uint64_t>(
            [&](uint64_t vid)
            {
                if(labels[vid].data != MAXL)
                {
                    layer_counts[labels[vid].data]++;
                    return 1;
                }
                return 0;
            },
            graph.get_dense_active_all()
        );
        for(uint64_t i=0;i<layer_counts.size();i++)
        {
            if(layer_counts[i] > 0)
            {
                printf("%lu: %lu, ", i, layer_counts[i].load());
            }
        }
        printf("\n");
    }
    return 0;
}
