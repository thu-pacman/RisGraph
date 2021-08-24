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
    uint64_t imported_edges = raw_edges_len;
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
    double one_over_n = 1.0 / num_vertices;

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

        for(uint64_t i=0;i<num_vertices && delta>1e-6;i++)
        {
            fprintf(stderr, "delta(%lu) = %le\n", i, delta);
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
