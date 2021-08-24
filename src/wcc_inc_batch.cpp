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
    Graph<void> graph(num_vertices, raw_edges_len, true, false);
    //std::random_shuffle(raw_edges.begin(), raw_edges.end());
    uint64_t imported_edges = raw_edges_len*0.5;
    {
        auto start = std::chrono::system_clock::now();
        #pragma omp parallel for
        for(uint64_t i=0;i<imported_edges;i++)
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

    std::vector<char> WAL;
    auto WAL_fd = open("WAL.log", O_CREAT | O_RDWR | O_TRUNC, 0664);
    assert(WAL_fd >= 0);
    {
        auto labels_pre = labels;
        auto start = std::chrono::system_clock::now();
        auto pre_time = start; const uint64_t print_int = 1000000;
        const uint64_t batch = 10;
        auto func = [&](uint64_t i)
        {
            if((i-imported_edges)%print_int == 0)
            {
                auto num_changes = graph.stream_vertices<uint64_t>(
                    [&](uint64_t vid)
                    {
                        if(labels[vid].data != labels_pre[vid].data)
                        {
                            labels_pre[vid].data = labels[vid].data;
                            return 1;
                        }
                        return 0;
                    },
                    graph.get_dense_active_all()
                );
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
                fprintf(stderr, "%lu, changes: %lu, average depth: %lf, time: %.6lfs, edges/sec: %.6lf\n", i-imported_edges, num_changes, (double)total_depth/num_vertices, 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(now-pre_time).count(), print_int/(1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(now-pre_time).count()));
                pre_time = now;
            }

            std::atomic_uint64_t WAL_len(0);
            std::mutex WAL_mutex;
            const char * BEG = "BEG";
            const char * ADD = "ADD";
            const char * DEL = "DEL";
            const char * END = "END";
            atomic_append(WAL, WAL_len, BEG, strlen(BEG)+1, WAL_mutex);

            {
                static std::vector<decltype(graph)::edge_type> added_edges;
                added_edges.resize(batch);
                std::atomic_uint64_t length(0);
                atomic_append(WAL, WAL_len, ADD, strlen(ADD)+1, WAL_mutex);
                THRESHOLD_OPENMP("omp parallel for", batch, 
                    for(uint64_t j=0;j<batch;j++)
                    {   
                        if(i+j>=raw_edges_len) continue;
                        const auto &e = raw_edges[i+j];
                        decltype(graph)::edge_type edge = {e.first, e.second};
                        atomic_append(WAL, WAL_len, &edge, sizeof(edge), WAL_mutex);
                        auto old_num = graph.add_edge({e.first, e.second}, false);
                        if(!old_num) added_edges[length.fetch_add(1)] = {e.first, e.second};
                    }
                );
                added_edges.resize(length.load());
                graph.update_tree_add<uint64_t, uint64_t>(
                    continue_reduce_func,
                    update_func,
                    active_result_func,
                    labels, added_edges, false
                );
            }

            {
                static std::vector<decltype(graph)::edge_type> deled_edges;
                deled_edges.resize(batch);
                std::atomic_uint64_t length(0);
                atomic_append(WAL, WAL_len, DEL, strlen(DEL)+1, WAL_mutex);
                THRESHOLD_OPENMP("omp parallel for", batch, 
                    for(uint64_t j=0;j<batch;j++)
                    {   
                        if(i+j>=raw_edges_len) continue;
                        const auto &e = raw_edges[i+j-imported_edges];
                        decltype(graph)::edge_type edge = {e.first, e.second};
                        atomic_append(WAL, WAL_len, &edge, sizeof(edge), WAL_mutex);
                        auto old_num = graph.del_edge({e.first, e.second}, false);
                        if(old_num==1) deled_edges[length.fetch_add(1)] = {e.first, e.second};
                    }
                );
                deled_edges.resize(length.load());
                graph.update_tree_del<uint64_t, uint64_t>(
                    init_label_func,
                    continue_reduce_func,
                    update_func,
                    active_result_func,
                    equal_func,
                    labels, deled_edges, false
                );
            }

            atomic_append(WAL, WAL_len, END, strlen(END)+1, WAL_mutex);
            //auto ret = write(WAL_fd, WAL.data(), WAL_len.load());
            //assert((uint64_t)ret == WAL_len.load());
            //ret = fdatasync(WAL_fd);
            //assert(ret == 0);
        };

        for(uint64_t i=imported_edges;i<raw_edges_len;i+=batch) func(i);
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
