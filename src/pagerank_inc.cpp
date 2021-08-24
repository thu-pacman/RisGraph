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
#include <cmath>

const double d = (double)0.85;
const double eps_total = 1e-6;
const double eps_local = 1e-6;
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
    Graph<void> graph(num_vertices, raw_edges_len, false, true);
    //std::random_shuffle(raw_edges.begin(), raw_edges.end());
    uint64_t imported_edges = raw_edges_len/2;
    {
        auto start = std::chrono::system_clock::now();
        #pragma omp parallel for
        for(uint64_t i=0;i<imported_edges;i++)
        {
            const auto &e = raw_edges[i];
            graph.add_edge({e.first, e.second}, true);
        }
        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "add: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }

    auto curr = graph.alloc_vertex_array<double>();
    auto next = graph.alloc_vertex_array<double>();
    auto labels = graph.alloc_vertex_array<uint64_t>();
    double one_over_n = 1.0 / num_vertices;
    graph.fill_vertex_array(labels, 0lu);

    {
        auto start = std::chrono::system_clock::now();
        auto delta = graph.stream_vertices<double>(
            [&](uint64_t vid)
            {
                curr[vid] = one_over_n;
                auto outgoing_degree = graph.get_outgoing_degree(vid);
                if(outgoing_degree > 0)
                {
                    curr[vid] /= outgoing_degree;
                }
                return one_over_n;
            },
            graph.get_dense_active_all()
        );

        for(uint64_t i=0;i<num_vertices && delta > eps_total;i++)
        {
            fprintf(stderr, "delta(%lu) = %.6le\n", i, delta);
            graph.fill_vertex_array(next, 0.0);
            delta = graph.stream_edges<uint64_t>(
                [&](uint64_t src, const decltype(graph)::adjlist_range_type &outgoing_range)
                {
                    for(auto iter=outgoing_range.first;iter!=outgoing_range.second;iter++) 
                    {
                        auto edge = *iter;
                        const uint64_t dst = edge.nbr;
                        if(edge.num > 0) write_add(&next[dst], curr[src]*edge.num);
                    }
                    return 0;
                },
                [&](uint64_t dst, const decltype(graph)::adjlist_range_type &incoming_range)
                {
                    double sum = 0;
                    for(auto iter=incoming_range.first;iter!=incoming_range.second;iter++) 
                    {
                        auto edge = *iter;
                        const uint64_t src = edge.nbr;
                        if(edge.num > 0) sum += curr[src]*edge.num;
                    }
                    next[dst] = sum;
                    return 0;
                },
                graph.get_dense_active_all()
            );
            delta = graph.stream_vertices<double>(
                [&](uint64_t vid)
                {
                    next[vid] = (1 - d) * one_over_n + d * next[vid];
                    auto outgoing_degree = graph.get_outgoing_degree(vid);
                    if(outgoing_degree > 0)
                    {
                        next[vid] /= outgoing_degree;
                        return std::fabs(next[vid]-curr[vid])*outgoing_degree;
                    }
                    return std::fabs(next[vid]-curr[vid]);
                },
                graph.get_dense_active_all()
            );
            std::swap(curr, next);
        }
        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "exec: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }

    {
        auto start = std::chrono::system_clock::now();
        auto pre_time = start; const uint64_t print_int = 10000;
        auto &active_in = graph.get_sparse_active_in();
        auto &active_out = graph.get_sparse_active_out();
        uint64_t cur_label = 0;
        std::atomic_uint64_t length_in, length_out;
        uint64_t total_length = 0;
        for(uint64_t i=imported_edges;i<raw_edges_len;i++)
        {
            if((i-imported_edges)%print_int == 0)
            {
                {
                    double maxpr = 0.0;
                    uint64_t max_idx = 0;
                    double sum = graph.stream_vertices<double>(
                        [&](uint64_t vid)
                        {
                            auto outgoing_degree = graph.get_outgoing_degree(vid);
                            double pr = curr[vid];
                            if(outgoing_degree > 0)
                            {
                                pr *= outgoing_degree;
                            }
                            if(write_max(&maxpr, pr))
                            {
                                max_idx = vid;
                            }
                            return pr;
                        },
                        graph.get_dense_active_all()
                    );
                    auto now = std::chrono::system_clock::now();
                    fprintf(stderr, "%lu, sum: %le, maxpr[%lu]: %le, average_length: %lf, time: %.6lfs, edges/sec: %.6lf\n", i-imported_edges, sum, max_idx, maxpr, (double)total_length/print_int, 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(now-pre_time).count(), print_int/(1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(now-pre_time).count()));
                    pre_time = now;
                    total_length = 0;
                }
            }
            if((i-imported_edges)%(print_int*100) == 0)
            {
                double delta = 1;
                for(uint64_t i=0;i<num_vertices && delta > eps_total;i++)
                {
                    fprintf(stderr, "delta(%lu) = %.6le\n", i, delta);
                    graph.fill_vertex_array(next, 0.0);
                    delta = graph.stream_edges<uint64_t>(
                        [&](uint64_t src, const decltype(graph)::adjlist_range_type &outgoing_range)
                        {
                            for(auto iter=outgoing_range.first;iter!=outgoing_range.second;iter++) 
                            {
                                auto edge = *iter;
                                const uint64_t dst = edge.nbr;
                                if(edge.num > 0) write_add(&next[dst], curr[src]*edge.num);
                            }
                            return 0;
                        },
                        [&](uint64_t dst, const decltype(graph)::adjlist_range_type &incoming_range)
                        {
                            double sum = 0;
                            for(auto iter=incoming_range.first;iter!=incoming_range.second;iter++) 
                            {
                                auto edge = *iter;
                                const uint64_t src = edge.nbr;
                                if(edge.num > 0) sum += curr[src]*edge.num;
                            }
                            next[dst] = sum;
                            return 0;
                        },
                        graph.get_dense_active_all()
                    );
                    delta = graph.stream_vertices<double>(
                        [&](uint64_t vid)
                        {
                            next[vid] = (1 - d) * one_over_n + d * next[vid];
                            auto outgoing_degree = graph.get_outgoing_degree(vid);
                            if(outgoing_degree > 0)
                            {
                                next[vid] /= outgoing_degree;
                                return std::fabs(next[vid]-curr[vid])*outgoing_degree;
                            }
                            return std::fabs(next[vid]-curr[vid]);
                        },
                        graph.get_dense_active_all()
                    );
                    std::swap(curr, next);
                }
                {
                    double maxpr = 0.0;
                    uint64_t max_idx = 0;
                    double sum = graph.stream_vertices<double>(
                        [&](uint64_t vid)
                        {
                            auto outgoing_degree = graph.get_outgoing_degree(vid);
                            double pr = curr[vid];
                            if(outgoing_degree > 0)
                            {
                                pr *= outgoing_degree;
                            }
                            if(write_max(&maxpr, pr))
                            {
                                max_idx = vid;
                            }
                            return pr;
                        },
                        graph.get_dense_active_all()
                    );
                    auto now = std::chrono::system_clock::now();
                    fprintf(stderr, "%lu, sum: %le, maxpr[%lu]: %le, average_length: %lf, time: %.6lfs, edges/sec: %.6lf\n", i-imported_edges, sum, max_idx, maxpr, (double)total_length/print_int, 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(now-pre_time).count(), print_int/(1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(now-pre_time).count()));
                    pre_time = now;
                    total_length = 0;
                }
            }
            
            length_in.store(0, std::memory_order_relaxed);
            cur_label++;

            {
                const auto &e = raw_edges[i];
                graph.add_edge({e.first, e.second}, true);
                atomic_append(active_in, length_in, e.first, graph.get_global_lock());
                atomic_append(active_in, length_in, e.second, graph.get_global_lock());
            }

            {
                const auto &e = raw_edges[i-imported_edges];
                graph.del_edge({e.first, e.second}, true);
                atomic_append(active_in, length_in, e.first, graph.get_global_lock());
                atomic_append(active_in, length_in, e.second, graph.get_global_lock());
            }
            
            while(length_in.load() > 0)
            {
                cur_label++;
                length_out.store(0, std::memory_order_relaxed);
                total_length += length_in.load();

                graph.stream_vertices<uint64_t>(
                    [&](uint64_t vid)
                    {
                        double sum = 0;
                        auto incoming_range = graph.get_incoming_adjlist_range(vid);
                        for(auto iter=incoming_range.first;iter!=incoming_range.second;iter++) 
                        {
                            auto edge = *iter;
                            const uint64_t src = edge.nbr;
                            if(edge.num > 0) sum += curr[src]*edge.num;
                        }
                        next[vid] = sum;
                        return 0;
                    },
                    active_in, length_in.load(std::memory_order_relaxed)
                );
                graph.stream_vertices<double>(
                    [&](uint64_t vid)
                    {
                        next[vid] = (1 - d) * one_over_n + d * next[vid];
                        auto outgoing_degree = graph.get_outgoing_degree(vid);
                        double local_delta;
                        if(outgoing_degree > 0)
                        {
                            next[vid] /= outgoing_degree;
                            local_delta = std::fabs(next[vid]-curr[vid])*outgoing_degree;
                        }
                        else
                        {
                            local_delta = std::fabs(next[vid]-curr[vid]);
                        }
                        //if(local_delta > eps_local)
                        if(std::fabs(next[vid]-curr[vid])/curr[vid] > 1e-1)
                        {
                            auto old_val = labels[vid];
                            if(old_val != cur_label && cas(&labels[vid], old_val, cur_label))
                            {
                                atomic_append(active_out, length_out, vid, graph.get_global_lock());
                            }
                            auto outgoing_range = graph.get_outgoing_adjlist_range(vid);
                            for(auto iter=outgoing_range.first;iter!=outgoing_range.second;iter++) 
                            {
                                auto edge = *iter;
                                const uint64_t dst = edge.nbr;
                                if(edge.num > 0) 
                                {
                                    auto old_val = labels[dst];
                                    if(old_val != cur_label && cas(&labels[dst], old_val, cur_label))
                                    {
                                        atomic_append(active_out, length_out, dst, graph.get_global_lock());
                                    }
                                }
                                
                            }
                        }
                        return local_delta;
                    },
                    active_in, length_in.load(std::memory_order_relaxed)
                );
                std::swap(curr, next);
                std::swap(active_in, active_out);
                length_in.store(length_out.load(std::memory_order_relaxed), std::memory_order_relaxed);
            }
        }
        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "exec: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }

    {
        double sum = graph.stream_vertices<double>(
            [&](uint64_t vid)
            {
                auto outgoing_degree = graph.get_outgoing_degree(vid);
                if(outgoing_degree > 0)
                {
                    curr[vid] *= outgoing_degree;
                }
                return curr[vid];
            },
            graph.get_dense_active_all()
        );
        uint64_t max_vid = 0;
        for(uint64_t i=0;i<num_vertices;i++)
        {
            if(curr[i] > curr[max_vid]) max_vid = i;
        }
        fprintf(stderr, "max_pr[%lu] = %le, sum = %le\n", max_vid, curr[max_vid], sum);
    }

    return 0;
}
