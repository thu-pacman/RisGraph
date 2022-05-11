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
#include <immintrin.h>
#include <omp.h>
#include "core/type.hpp"
#include "core/graph.hpp"
#include "core/io.hpp"

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

    {
        auto start = std::chrono::system_clock::now();

        graph.build_tree<uint64_t, uint64_t>(
            init_label_func,
            continue_reduce_func,
            update_func,
            active_result_func,
            labels
        );

        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "exec: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }

    {
        auto labels_pre = labels;
        auto start = std::chrono::system_clock::now();
        auto pre_time = start; const uint64_t print_int = 10000000;
        auto func = [&](uint64_t i)
        {
            if((i-imported_edges)%print_int == 0)
            {
                std::vector<std::atomic_uint64_t> layer_counts(MAXL);
                for(auto &a : layer_counts) a = 0;
                auto num_visited = graph.stream_vertices<uint64_t>(
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
                fflush(stdout);
                auto total_depth = graph.stream_vertices<uint64_t>(
                    [&](uint64_t vid)
                    {
                        uint64_t depth = 0;
                        for(;labels[vid].parent.nbr != num_vertices;vid = labels[vid].parent.nbr)
                        {
                            depth++;
                        }
                        return depth;
                    },
                    graph.get_dense_active_all()
                );
                auto now = std::chrono::system_clock::now();
                fprintf(stderr, "%lu, visited: %lu, average depth: %lf, time: %.6lfs, edges/sec: %.6lf\n", i-imported_edges, num_visited, (double)total_depth/num_vertices, 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(now-pre_time).count(), print_int/(1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(now-pre_time).count()));
                pre_time = now;
            }
            {
                const auto &e = raw_edges[i];
                auto old_num = graph.add_edge({e.first, e.second, (e.first+e.second)%128}, true);
                if(!old_num)
                {
                    graph.update_tree_add<uint64_t, uint64_t>(
                        continue_reduce_func,
                        update_func,
                        active_result_func,
                        labels, {e.first, e.second, (e.first+e.second)%128}, true
                    );
                }
            }

            {
                const auto &e = raw_edges[i-imported_edges];
                auto old_num = graph.del_edge({e.first, e.second, (e.first+e.second)%128}, true);
                if(old_num==1)
                {
                    graph.update_tree_del<uint64_t, uint64_t>(
                        init_label_func,
                        continue_reduce_func,
                        update_func,
                        active_result_func,
                        equal_func,
                        labels, {e.first, e.second, (e.first+e.second)%128}, true
                    );
                }
                    
            }
        };

        for(uint64_t i=imported_edges;i<raw_edges_len;i++) func(i);
        //{
        //    const uint64_t start = imported_edges;
        //    const uint64_t end = raw_edges_len;
        //    const uint64_t batch = 64;
        //    std::vector<std::thread> pool;
        //    std::atomic_uint64_t global_i = start;
        //    for(int i=0;i<omp_get_num_threads();i++)
        //    {
        //        pool.emplace_back([&]()
        //        {
        //            while(global_i.load() < end)
        //            {
        //                uint64_t begin_i = global_i.fetch_add(batch);
        //                for(uint64_t i=begin_i, j=0;i<end&&j<batch;i++,j++) func(i);
        //            }
        //        });
        //    }
        //    for(auto & t : pool)
        //    {
        //        t.join();
        //    }
        //}

        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "exec: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }

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

    {
        auto start = std::chrono::system_clock::now();

        graph.build_tree<uint64_t, uint64_t>(
            init_label_func,
            continue_reduce_func,
            update_func,
            active_result_func,
            labels
        );

        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "exec: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }
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
