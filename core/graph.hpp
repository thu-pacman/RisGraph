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

#pragma once
#include <cstdio>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <exception>
#include <functional>
#include <mutex>
#include <atomic>
#include <immintrin.h>
#include <algorithm>
#include <omp.h>
#include <tbb/tbb.h>
#include <sparsehash/dense_hash_map>
#include <cmath>
#include "type.hpp"
#include "bitmap.hpp"
#include "atomic.hpp"
#include "actset.hpp"
#include "mvvec.hpp"
#include "storage.hpp"
#include "io.hpp"

template <typename EdgeData = void>
class Graph 
{
public:
    using adjedge_type = AdjEdge<EdgeData>;
    using edge_type = Edge<EdgeData>;
    using Storage = IndexedEdgeStorage<adjedge_type, storage::data::Vector, storage::index::DenseHashMap>;
    //using Storage = IndexOnlyStorage<adjedge_type, storage::data::Vector, storage::index::DenseHashMap>;
    using adjlist_type = typename Storage::adjlist_type;
    using adjlist_iter_type = typename Storage::adjlist_iter_type;
    using adjlist_range_type = std::pair<adjlist_iter_type, adjlist_iter_type>;
    using lock_type = typename Storage::lock_type;

    template<typename VertexData>
    struct VertexTree
    {
        adjedge_type parent;
        VertexData data;
    };

    Graph(uint64_t _vertices, uint64_t _hint_edges, bool _symmetric, bool _dual, bool _trace_modified = false)
        : vertices(_vertices), dense_threshold(vertices), symmetric(_symmetric), dual(_dual),
        outgoing(), incoming(),
        edges(_hint_edges==0?_vertices*16:_hint_edges),
        dense_active_all(vertices),
        active_in(vertices), active_out(vertices), active_tree(vertices), 
        invalidated(vertices), invalidated_idx(0),
        offsets(), empty_parent(), modified(vertices), trace_modified(_trace_modified)
    {
        int ncpus = omp_get_max_threads();
        cpu_set_t cpuset_full, cpuset;
        pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        CPU_ZERO(&cpuset_full);
        #pragma omp parallel
        {
            #pragma omp critical
            {
                cpu_set_t local_cpuset;
                pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &local_cpuset);
                CPU_OR(&cpuset_full, &cpuset_full, &local_cpuset);
                thread_id.local() = omp_get_thread_num();
            }
        }
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset_full);
        tbb::task_scheduler_init init(ncpus);
        //task_arena.initialize(ncpus);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

        fprintf(stderr, "|V|=%lu\n", _vertices);
        empty_parent.nbr = vertices;

        outgoing.resize(vertices);

        if(!symmetric) 
        {
            incoming.resize(vertices);
        }

        dense_active_all.fill();
    }

    uint64_t get_thread_id()
    {
        return thread_id.local();
    }

    lock_type& get_lock(uint64_t vid)
    {
        return outgoing.get_lock(vid);
    }

    std::mutex& get_global_lock()
    {
        return mutex;
    }

    ActiveSet &get_modified()
    {
        return modified;
    }

    uint64_t get_modified_length()
    {
        return modified.get_sparse_length();
    }

    void clear_modified()
    {
        modified.clear();
    }

    void transpose()
    {
        if(!symmetric)
        {
            std::swap(outgoing, incoming);
        }
    }

    Bitmap& get_dense_active_in()
    {
        return active_in.get_dense_ref();
    }

    Bitmap& get_dense_active_out()
    {
        return active_out.get_dense_ref();
    }

    Bitmap& get_dense_active_all()
    {
        return dense_active_all;
    }

    std::vector<uint64_t>& get_sparse_active_in()
    {
        return active_in.get_sparse_ref();
    }

    std::vector<uint64_t>& get_sparse_active_out()
    {
        return active_out.get_sparse_ref();
    }

    uint64_t get_incoming_degree(uint64_t vid)
    {
        if(symmetric) return get_outgoing_degree(vid);
        return incoming.get_degree(vid);
    }

    uint64_t get_outgoing_degree(uint64_t vid)
    {
        return outgoing.get_degree(vid);
    }

    adjlist_type& get_outgoing_adjlist(uint64_t vid)
    {
        return outgoing.get_adjlist(vid);
    }

    adjlist_type& get_incoming_adjlist(uint64_t vid)
    {
        if(symmetric) return get_outgoing_adjlist(vid);
        return incoming.get_adjlist(vid);
    }

    adjlist_range_type get_outgoing_adjlist_range(uint64_t vid)
    {
        return outgoing.get_adjlist_iter(vid);
    }

    adjlist_range_type get_incoming_adjlist_range(uint64_t vid)
    {
        if(symmetric) return get_outgoing_adjlist_range(vid);
        return incoming.get_adjlist_iter(vid);
    }

    uint64_t get_edge_num(edge_type e)
    {
        return outgoing.get_edge_num(e.src, e);
    }

    uint64_t add_edge(edge_type e, bool directed = true)
    {
        if(e.src >= vertices || e.dst >= vertices) throw std::runtime_error("VertexId error.");
        if(!directed)
        {
            add_edge(e, true);
            std::swap(e.src, e.dst);
            return add_edge(e, true) - (e.src == e.dst);
        }

        uint64_t current_size = 0;
        adjedge_type ae = e;

        {
            current_size = outgoing.update_edge(ae, e.src, 1);
        }

        if(!symmetric)
        {
            ae.nbr = e.src;
            incoming.update_edge(ae, e.dst, 1);
        }

        return current_size;
    }

    uint64_t del_edge(edge_type e, bool directed = true)
    {
        if(e.src >= vertices || e.dst >= vertices) throw std::runtime_error("VertexId error.");
        if(!directed)
        {
            del_edge(e, true);
            std::swap(e.src, e.dst);
            return del_edge(e, true);
        }

        uint64_t current_size = 0;
        adjedge_type ae = e;

        {
            current_size = outgoing.update_edge(ae, e.src, -1);
        }

        if(!symmetric)
        {
            ae.nbr = e.src;
            incoming.update_edge(ae, e.dst, -1);
        }

        return current_size;
    }

    template<typename VertexData>
    std::vector<VertexTree<VertexData>> alloc_vertex_tree_array()
    {
        std::vector<VertexTree<VertexData>> ta(vertices);
        #pragma omp parallel for
        for(uint64_t i=0;i<vertices;i++) ta[i].parent = empty_parent;
        return ta;
    }

    template<typename VertexData>
    void fill_vertex_tree_array(std::vector<VertexTree<VertexData>> &ta, VertexData value)
    {
        #pragma omp parallel for
        for(uint64_t i=0;i<vertices;i++) {
            ta[i].data = value;
        }
    }

    template<typename VertexData>
    std::vector<VertexData> alloc_vertex_array()
    {
        std::vector<VertexData> a;
        a.resize(vertices);
        return a;
    }

    template<typename VertexData>
    void fill_vertex_array(std::vector<VertexData> &a, VertexData value)
    {
        #pragma omp parallel for
        for(uint64_t i=0;i<vertices;i++) {
            a[i] = value;
        }
    }

    Bitmap alloc_vertex_bitmap()
    {
        return Bitmap(vertices);
    }

    template<typename VertexData>
    MVVec<VertexData> alloc_history_array()
    {
        return MVVec<VertexData>(vertices);
    }

    template<typename R>
    R stream_vertices(std::function<R(uint64_t)> process, const Bitmap &active)
    {
        R reducer = 0;
        #pragma omp parallel for schedule(dynamic, 64) reduction(+:reducer)
        for(uint64_t word_i=0;word_i<WORD_OFFSET(vertices)+1;word_i++)
        {
            uint64_t v_i = BEGIN_OF_WORD(word_i);
            uint64_t word = active.data[word_i];
            while(word != 0)
            {
                if(word & 1)
                {
                    reducer += process(v_i);
                }
                v_i++;
                word >>= 1;
            }
        }
        return reducer;
    }

    template<typename R>
    R stream_edges(std::function<R(uint64_t, const adjlist_range_type &range)> sparse_process, std::function<R(uint64_t, const adjlist_range_type &range)> dense_process, const Bitmap &active)
    {
        R reducer = 0;
        uint64_t active_edges = stream_vertices<uint64_t>(
            [&](uint64_t vid)
            {
                return outgoing.get_degree(vid);
            },
            active
        );
        bool sparse = sparse_process && (active_edges < dense_threshold || active_edges < edges/20 || !dual || !dense_process);
        //fprintf(stderr, "%lu %lu %s\n", active_edges, edges.load(), sparse?"sparse":"dense");
        if(sparse)
        {
            #pragma omp parallel for schedule(dynamic, (active_edges<edges/200)?65536:64) reduction(+:reducer)
            for(uint64_t word_i=0;word_i<WORD_OFFSET(vertices)+1;word_i++)
            {
                uint64_t v_i = BEGIN_OF_WORD(word_i);
                uint64_t word = active.data[word_i];
                while(word != 0)
                {
                    if(word & 1)
                    {
                        if(outgoing.get_degree(v_i))
                        {
                            reducer += sparse_process(v_i, outgoing.get_adjlist_iter(v_i));
                        }
                    }
                    v_i++;
                    word >>= 1;
                }
            }
            //reducer = tbb::parallel_reduce(tbb::blocked_range<uint64_t>(0lu, WORD_OFFSET(vertices)+1, (active_edges<edges/200)?65536:64), reducer, 
            //[&](const tbb::blocked_range<uint64_t> &range, R res) -> R
            //{
            //    for(uint64_t word_i=range.begin();word_i!=range.end();word_i++)
            //    {
            //        uint64_t v_i = BEGIN_OF_WORD(word_i);
            //        uint64_t word = active.data[word_i];
            //        while(word != 0)
            //        {
            //            if(word & 1)
            //            {
            //                if(outgoing.get_degree(v_i)) res += sparse_process(v_i, outgoing.get_adjlist(v_i));
            //            }
            //            v_i++;
            //            word >>= 1;
            //        }
            //    }
            //    return res;
            //},
            //[](R x, R y) -> R
            //{
            //    return x+y;
            //});
        }
        else
        {
            #pragma omp parallel for schedule(dynamic, 64) reduction(+:reducer)
            for(uint64_t v_i=0;v_i<vertices;v_i++)
            {
                if(symmetric)
                {
                    if(outgoing.get_degree(v_i)) reducer += dense_process(v_i, outgoing.get_adjlist_iter(v_i));
                }
                else
                {
                    if(incoming.get_degree(v_i)) reducer += dense_process(v_i, incoming.get_adjlist_iter(v_i));
                }
            }
        }
        return reducer;
    }

    //template<typename R>
    //R stream_vertices(std::function<R(uint64_t)> process, const std::vector<uint64_t> &active, const uint64_t &length)
    template<typename R, typename Process>
    R stream_vertices(Process process, const std::vector<uint64_t> &active, const uint64_t &length)
    {
        R reducer = 0;
        if(length < OPENMP_THRESHOLD)
        {
            for(uint64_t i=0;i<length;i++)
            {
                uint64_t v_i = active[i];
                reducer += process(v_i);
            }
        }
        else
        {
            reducer = tbb::parallel_reduce(tbb::blocked_range<uint64_t>(0lu, length, 256), reducer, 
            [&](const tbb::blocked_range<uint64_t> &range, R res) -> R
            {
                for(uint64_t i=range.begin();i!=range.end();i++)
                {
                    uint64_t v_i = active[i];
                    res += process(v_i);
                }
                return res;
            },
            [](R x, R y) -> R
            {
                return x+y;
            },
            affinity_partitioner);
        }
        return reducer;
    }

    template<typename R>
    R stream_edges(std::function<R(uint64_t, const adjlist_range_type &range)> sparse_process, std::function<R(uint64_t, const adjlist_range_type &range)> dense_process, const std::vector<uint64_t> &active, const uint64_t &length, uint64_t active_edges = (uint64_t)-1)
    {
        R reducer = 0;
        if(active_edges == (uint64_t)-1)
        {
            active_edges = stream_vertices<uint64_t>(
                [&](uint64_t vid)
                {
                    return outgoing.get_degree(vid);
                },
                active, length
            );
        }
        bool sparse = sparse_process && (active_edges < dense_threshold || active_edges < edges/20 || !dual || !dense_process);
        //fprintf(stderr, "%lu %lu %s\n", active_edges, edges.load(), sparse?"sparse":"dense");
        if(sparse)
        {
            //THRESHOLD_OPENMP("omp parallel for schedule(dynamic, 64) reduction(+:reducer)", length, 
            //    for(uint64_t i=0;i<length;i++)
            //    {
            //        uint64_t v_i = active[i];
            //        if(outgoing.get_degree(v_i)) reducer += sparse_process(v_i, outgoing.get_adjlist(v_i));
            //    }
            //);
            if(length < OPENMP_THRESHOLD)
            {
                for(uint64_t i=0;i<length;i++)
                {
                    uint64_t v_i = active[i];
                    if(outgoing.get_degree(v_i)) reducer += sparse_process(v_i, outgoing.get_adjlist_iter(v_i));
                }
            }
            else
            {
                reducer = tbb::parallel_reduce(tbb::blocked_range<uint64_t>(0lu, length, 64), reducer, 
                [&](const tbb::blocked_range<uint64_t> &range, R res) -> R
                {
                    for(uint64_t i=range.begin();i!=range.end();i++)
                    {
                        uint64_t v_i = active[i];
                        if(outgoing.get_degree(v_i)) res += sparse_process(v_i, outgoing.get_adjlist_iter(v_i));
                    }
                    return res;
                },
                [](R x, R y) -> R
                {
                    return x+y;
                },
                affinity_partitioner);
            }
        }
        else
        {
            #pragma omp parallel for schedule(dynamic, 32) reduction(+:reducer)
            for(uint64_t v_i=0;v_i<vertices;v_i++)
            {
                if(symmetric)
                {
                    if(outgoing.get_degree(v_i)) reducer += dense_process(v_i, outgoing.get_adjlist_iter(v_i));
                }
                else
                {
                    if(incoming.get_degree(v_i)) reducer += dense_process(v_i, incoming.get_adjlist_iter(v_i));
                }
            }
        }
        return reducer;
    }

    //template<typename R>
    //R stream_edges_sparse(std::function<R(uint64_t, const adjedge_type &)> process, const std::vector<uint64_t> &active, const uint64_t &length)
    template<typename R, typename Process>
    R stream_edges_sparse(Process process, const std::vector<uint64_t> &active, const uint64_t &length, uint64_t active_edges = (uint64_t)-1)
    {
        R reducer = 0;
        if(active_edges == (uint64_t)-1)
        {
            active_edges = stream_vertices<uint64_t>(
                [&](uint64_t vid)
                {
                    return outgoing.get_degree(vid);
                },
                active, length
            );
        }
        if(active_edges > OPENMP_THRESHOLD)
        {
            if(offsets.size() < length+1) offsets.resize(length+1);
            for(uint64_t i=1;i<=length;i++)
            {
                offsets[i] = offsets[i-1] + outgoing.get_adjlist(active[i-1]).size();
            }
            tbb::enumerable_thread_specific<uint64_t> next_a(1);
            //thread_local uint64_t next = 1;
            reducer = tbb::parallel_reduce(tbb::blocked_range<uint64_t>(0lu, offsets[length], 256), reducer, 
            [&](const tbb::blocked_range<uint64_t> &range, R res) -> R
            {
                uint64_t next = next_a.local();
                if(next > length || offsets[next-1] > range.begin()) next = (length < 32) ? 1 : std::upper_bound(offsets.begin(), offsets.begin()+length+1, range.begin())-offsets.begin();
                for(uint64_t i=range.begin();i!=range.end();i++)
                {
                    while(offsets[next] <= i) next++;
                    uint64_t v_i = active[next-1];
                    res += process(v_i, outgoing.get_adjlist(v_i)[i-offsets[next-1]]);
                }
                next_a.local() = next;
                return res;
            },
            [](R x, R y) -> R
            {
                return x+y;
            },
            affinity_partitioner);
        }
        else
        {
            for(uint64_t i=0;i<length;i++)
            {
                uint64_t v_i = active[i];
                if(outgoing.get_degree(v_i))
                {
                    for(auto e:outgoing.get_adjlist(v_i)) reducer += process(v_i, e);
                }
            }
        }
        return reducer;
    }

    template<typename R, typename Process>
    R stream_vertices_hybrid(Process process, ActiveSet &active)
    {
        if(active.is_dense())
        {
            return stream_vertices<R>(process, active.get_dense());
        }
        else
        {
            return stream_vertices<R>(process, active.get_sparse(), active.get_sparse_length());
        }
    }

    //TODO pull
    template<typename R, typename ProcessEdge>
    R stream_edges_hybrid(std::function<R(uint64_t, const adjlist_range_type &range)> process_push, std::function<R(uint64_t, const adjlist_range_type &range)> process_pull, ProcessEdge process_edge, ActiveSet &active)
    {
        if(active.is_dense()) 
        {
            //fprintf(stderr, "stream_edges_dense >= %lu\n", active.get_sparse_length());
            return stream_edges<R>(process_push, process_pull, active.get_dense());
        }
        else
        {
            //return stream_edges<R>(process_push, nullptr, active.get_sparse(), active.get_sparse_length());
            //return stream_edges_sparse<R>(process_edge, active.get_sparse(), active.get_sparse_length());
            uint64_t active_edges = stream_vertices<uint64_t>(
                [&](uint64_t vid)
                {
                    return outgoing.get_degree(vid);
                },
                active.get_sparse(), active.get_sparse_length()
            );
            if(active_edges < 16384 || !Storage::edge_random_accessable)
            {
                return stream_edges<R>(process_push, nullptr, active.get_sparse(), active.get_sparse_length());
            }
            double x = log(active.get_sparse_length()), y = log(active_edges);
            double predict = -2.20754644*x+0.58438928*y+14.45252841;
            //uint64_t active_edges = (uint64_t) -1;
            //const uint64_t vertex_centric_threshold = 2048000;//1024000;
            //if(active.get_sparse_length() > vertex_centric_threshold || !Storage::edge_random_accessable)
            if(predict < 0)
            {
                //fprintf(stderr, "stream_edges_sparse_vertex %lu\n", active.get_sparse_length());
                return stream_edges<R>(process_push, nullptr, active.get_sparse(), active.get_sparse_length(), active_edges);
            }
            else
            {
                return stream_edges_sparse<R>(process_edge, active.get_sparse(), active.get_sparse_length(), active_edges);
            }
        }
    }

    //template <typename R, typename DataType>
    //R build_tree_raw(
    //        std::function<std::pair<DataType, bool>(uint64_t vid)> init_label_func, 
    //        std::function<std::pair<bool, R>(uint64_t depth, R total_result, R local_result)> continue_reduce_func, 
    //        std::function<std::pair<bool, DataType>(uint64_t src, uint64_t dst, DataType src_data, DataType dst_data, adjedge_type adjedge)> update_func, 
    //        std::function<R(R old_result, uint64_t src, uint64_t dst, DataType src_data, DataType old_dst_data, DataType new_dst_data)> active_result_func, 
    //        std::vector<DataType> &labels)
    template <typename R, typename DataType, typename InitLabelFunc, typename ContinueReduceFunc, typename UpdateFunc, typename ActiveResultFunc>
    R build_tree_raw(
            InitLabelFunc init_label_func, 
            ContinueReduceFunc continue_reduce_func, 
            UpdateFunc update_func, 
            ActiveResultFunc active_result_func, 
            std::vector<DataType> &labels)
    {
        auto &active_all = get_dense_active_all();
        auto &active_in = get_dense_active_in();
        auto &active_out = get_dense_active_out();
        active_in.clear();
        stream_vertices<uint64_t>(
            [&](uint64_t vid)
            {
                labels[vid] = init_label_func(vid).first;
                if(init_label_func(vid).second) active_in.set_bit(vid);
                return 0;
            },
            active_all
        );
        R total_result = 0;
        for(uint64_t i=0;true;i++)
        {
            active_out.clear();
            R local_result = stream_edges<R>(
                [&](uint64_t src, const adjlist_range_type &outgoing_range)
                {
                    R result = 0;
                    for(auto iter = outgoing_range.first;iter != outgoing_range.second; iter++) 
                    {
                        auto edge = *iter;
                        if(edge.num > 0)
                        {
                            uint64_t dst = edge.nbr;
                            auto src_data = labels[src];
                            auto dst_data = labels[dst];
                            if(update_func(src, dst, src_data, dst_data, edge).first)
                            {
                                auto eup = edge; eup.nbr = src;
                                auto update_pair = update_label_raw(labels, src, dst, eup, update_func);
                                if(update_pair.first)
                                {
                                    active_out.set_bit(dst);
                                    result = active_result_func(result, src, dst, src_data, dst_data, update_pair.second);
                                }
                            }
                        }
                    }
                    return result;
                },
                [&](uint64_t dst, const adjlist_range_type &incoming_range)
                {
                    DataType new_label = labels[dst];
                    uint64_t new_src = (uint64_t)-1;
                    adjedge_type new_edge;
                    R result = 0;
                    for(auto iter = incoming_range.first;iter != incoming_range.second; iter++) 
                    {
                        auto edge = *iter;
                        if(edge.num > 0)
                        {
                            uint64_t src = edge.nbr;
                            auto src_data = labels[src];
                            auto update_pair = update_func(src, dst, src_data, new_label, edge);
                            if(update_pair.first)
                            {
                                new_label = update_pair.second;
                                new_src = src;
                                new_edge = edge;
                            }
                        }
                    }
                    if(new_src == (uint64_t)-1) return result;
                    auto src_data = labels[new_src];
                    auto dst_data = labels[dst];
                    if(update_func(new_src, dst, src_data, dst_data, new_edge).first)
                    {
                        auto src = new_src;
                        auto edge = new_edge;
                        auto eup = edge; eup.nbr = src;
                        auto update_pair = update_label_raw(labels, src, dst, eup, update_func);
                        if(update_pair.first)
                        {
                            active_out.set_bit(dst);
                            result = active_result_func(result, src, dst, src_data, dst_data, update_pair.second);
                        }
                    }
                    return result;
                },
                active_in
            );
            std::swap(active_in, active_out);
            bool is_continue;
            std::tie(is_continue, total_result) = continue_reduce_func(i, total_result, local_result);
            if(!is_continue) break;
        }
        return total_result;
    }

    //template <typename R, typename DataType>
    //R update_tree_add(
    //        std::function<std::pair<bool, R>(uint64_t depth, R total_result, R local_result)> continue_reduce_func, 
    //        std::function<std::pair<bool, DataType>(uint64_t src, uint64_t dst, DataType src_data, DataType dst_data, adjedge_type adjedge)> update_func, 
    //        std::function<R(R old_result, uint64_t src, uint64_t dst, DataType src_data, DataType old_dst_data, DataType new_dst_data)> active_result_func, 
    //        std::vector<DataType> &labels, edge_type edge, bool directed = true)
    template <typename R, typename DataType, typename ContinueReduceFunc, typename UpdateFunc, typename ActiveResultFunc>
    R update_tree_add_raw(
            ContinueReduceFunc continue_reduce_func, 
            UpdateFunc update_func, 
            ActiveResultFunc active_result_func, 
            std::vector<DataType> &labels, edge_type edge, bool directed = true)
    {
        if(!directed)
        {
            R retl = update_tree_add_raw<R, DataType>(continue_reduce_func, update_func, active_result_func, labels, edge, true);
            std::swap(edge.src, edge.dst);
            R retr = update_tree_add_raw<R, DataType>(continue_reduce_func, update_func, active_result_func, labels, edge, true);
            return continue_reduce_func(0, retl, retr).second;
        }

        R total_result = 0;
        if(update_func(edge.src, edge.dst, labels[edge.src], labels[edge.dst], edge).first) 
        {
            active_in.clear();
            adjedge_type eup = edge; eup.nbr = edge.src;
            auto src_data = labels[edge.src];
            auto dst_data = labels[edge.dst];
            auto update_pair = update_label_raw(labels, edge.src, edge.dst, eup, update_func);
            if(update_pair.first)
            {
                active_in.active(edge.dst);
                if(trace_modified) modified.active(edge.dst);
                total_result = active_result_func(total_result, edge.src, edge.dst, src_data, dst_data, update_pair.second);
            }
        }
        else
        {
            return total_result;
        }

        for(uint64_t i=0;true;i++)
        {
            active_out.clear();
            R local_result = stream_edges_hybrid<R>(
                [&](uint64_t src, const adjlist_range_type &outgoing_range)
                {
                    R result = 0;
                    for(auto iter = outgoing_range.first;iter != outgoing_range.second; iter++) 
                    {
                        auto edge = *iter;
                        if(edge.num > 0)
                        {
                            uint64_t dst = edge.nbr;
                            auto src_data = labels[src];
                            auto dst_data = labels[dst];
                            if(update_func(src, dst, src_data, dst_data, edge).first)
                            {
                                auto eup = edge; eup.nbr = src;
                                auto update_pair = update_label_raw(labels, src, dst, eup, update_func);
                                if(update_pair.first)
                                {
                                    active_out.active(dst);
                                    if(trace_modified) modified.active(dst);
                                    result = active_result_func(result, src, dst, src_data, dst_data, update_pair.second);
                                }
                            }
                        }
                    }
                    return result;
                },
                [&](uint64_t dst, const adjlist_range_type &incoming_range)
                {
                    DataType new_label = labels[dst];
                    uint64_t new_src = (uint64_t)-1;
                    adjedge_type new_edge;
                    R result = 0;
                    for(auto iter = incoming_range.first;iter != incoming_range.second; iter++) 
                    {
                        auto edge = *iter;
                        if(edge.num > 0)
                        {
                            uint64_t src = edge.nbr;
                            auto src_data = labels[src];
                            auto update_pair = update_func(src, dst, src_data, new_label, edge);
                            if(update_pair.first)
                            {
                                new_label = update_pair.second;
                                new_src = src;
                                new_edge = edge;
                            }
                        }
                    }
                    if(new_src == (uint64_t)-1) return result;
                    auto src_data = labels[new_src];
                    auto dst_data = labels[dst];
                    if(update_func(new_src, dst, src_data, dst_data, new_edge).first)
                    {
                        auto src = new_src;
                        auto edge = new_edge;
                        auto eup = edge; eup.nbr = src;
                        auto update_pair = update_label_raw(labels, src, dst, eup, update_func);
                        if(update_pair.first)
                        {
                            active_out.active(dst);
                            if(trace_modified) modified.active(dst);
                            result = active_result_func(result, src, dst, src_data, dst_data, update_pair.second);
                        }
                    }
                    return result;
                },
                [&](uint64_t src, const adjedge_type &edge)
                {
                    R result = 0;
                    if(edge.num > 0)
                    {
                        uint64_t dst = edge.nbr;
                        auto src_data = labels[src];
                        auto dst_data = labels[dst];
                        if(update_func(src, dst, src_data, dst_data, edge).first)
                        {
                            auto eup = edge; eup.nbr = src;
                            auto update_pair = update_label_raw(labels, src, dst, eup, update_func);
                            if(update_pair.first)
                            {
                                active_out.active(dst);
                                if(trace_modified) modified.active(dst);
                                result = active_result_func(result, src, dst, src_data, dst_data, update_pair.second);
                            }
                        }
                    }
                    return result;
                },
                active_in
            );
            std::swap(active_in, active_out);
            bool is_continue;
            std::tie(is_continue, total_result) = continue_reduce_func(i, total_result, local_result);
            if(!is_continue) break;
        }
        return total_result;
    }

    //template <typename R, typename DataType>
    //R build_tree(
    //        std::function<std::pair<DataType, bool>(uint64_t vid)> init_label_func, 
    //        std::function<std::pair<bool, R>(uint64_t depth, R total_result, R local_result)> continue_reduce_func, 
    //        std::function<std::pair<bool, DataType>(uint64_t src, uint64_t dst, DataType src_data, DataType dst_data, adjedge_type adjedge)> update_func, 
    //        std::function<R(R old_result, uint64_t src, uint64_t dst, DataType src_data, DataType old_dst_data, DataType new_dst_data)> active_result_func, 
    //        std::vector<VertexTree<DataType>> &labels)
    template <typename R, typename DataType, typename InitLabelFunc, typename ContinueReduceFunc, typename UpdateFunc, typename ActiveResultFunc>
    R build_tree(
            InitLabelFunc init_label_func, 
            ContinueReduceFunc continue_reduce_func, 
            UpdateFunc update_func, 
            ActiveResultFunc active_result_func, 
            std::vector<VertexTree<DataType>> &labels)
    {
        //static std::vector<VertexTree<DataType>> bak_labels = alloc_vertex_tree_array<DataType>();
        auto &active_all = get_dense_active_all();
        auto &active_in = get_dense_active_in();
        auto &active_out = get_dense_active_out();
        active_in.clear();
        stream_vertices<uint64_t>(
            [&](uint64_t vid)
            {
                labels[vid].data = init_label_func(vid).first;
                labels[vid].parent = empty_parent;
                //bak_labels[vid] = labels[vid];
                if(init_label_func(vid).second) active_in.set_bit(vid);
                return 0;
            },
            active_all
        );
        R total_result = 0;
        for(uint64_t i=0;true;i++)
        {
            //stream_vertices<uint64_t>(
            //    [&](uint64_t vid)
            //    {
            //        bak_labels[vid] = labels[vid];
            //        return 0;
            //    },
            //    active_in
            //);
            active_out.clear();
            R local_result = stream_edges<R>(
                [&](uint64_t src, const adjlist_range_type &outgoing_range)
                {
                    R result = 0;
                    for(auto iter = outgoing_range.first;iter != outgoing_range.second; iter++) 
                    {
                        auto edge = *iter;
                        if(edge.num > 0)
                        {
                            uint64_t dst = edge.nbr;
                            //auto src_data = bak_labels[src].data;
                            auto src_data = labels[src].data;
                            auto dst_data = labels[dst].data;
                            if(update_func(src, dst, src_data, dst_data, edge).first)
                            {
                                auto eup = edge; eup.nbr = src;
                                auto update_pair = update_label(labels, src, dst, eup, update_func, src_data);
                                if(update_pair.first)
                                {
                                    active_out.set_bit(dst);
                                    result = active_result_func(result, src, dst, src_data, dst_data, update_pair.second);
                                }
                            }
                        }
                    }
                    return result;
                },
                [&](uint64_t dst, const adjlist_range_type &incoming_range)
                {
                    DataType new_label = labels[dst].data;
                    uint64_t new_src = (uint64_t)-1;
                    adjedge_type new_edge;
                    R result = 0;
                    for(auto iter = incoming_range.first;iter != incoming_range.second; iter++) 
                    {
                        auto edge = *iter;
                        if(edge.num > 0)
                        {
                            uint64_t src = edge.nbr;
                            //auto src_data = bak_labels[src].data;
                            auto src_data = labels[src].data;
                            auto update_pair = update_func(src, dst, src_data, new_label, edge);
                            if(update_pair.first)
                            {
                                new_label = update_pair.second;
                                new_src = src;
                                new_edge = edge;
                            }
                        }
                    }
                    if(new_src == (uint64_t)-1) return result;
                    //auto src_data = bak_labels[new_src].data;
                    auto src_data = labels[new_src].data;
                    auto dst_data = labels[dst].data;
                    if(update_func(new_src, dst, src_data, dst_data, new_edge).first)
                    {
                        auto src = new_src;
                        auto edge = new_edge;
                        auto eup = edge; eup.nbr = src;
                        auto update_pair = update_label(labels, src, dst, eup, update_func, src_data);
                        if(update_pair.first)
                        {
                            active_out.set_bit(dst);
                            result = active_result_func(result, src, dst, src_data, dst_data, update_pair.second);
                        }
                    }
                    return result;
                },
                active_in
            );
            std::swap(active_in, active_out);
            bool is_continue;
            std::tie(is_continue, total_result) = continue_reduce_func(i, total_result, local_result);
            if(!is_continue) break;
        }
        return total_result;
    }

    template <typename DataType, typename UpdateFunc>
    bool need_update_tree_add(
            UpdateFunc update_func, 
            std::vector<VertexTree<DataType>> &labels, edge_type edge, bool directed = true)
    {
        if(!directed)
        {
            bool retl = need_update_tree_add<DataType>(update_func, labels, edge, true);
            std::swap(edge.src, edge.dst);
            bool retr = need_update_tree_add<DataType>(update_func, labels, edge, true);
            return retl || retr;
        }
        return (update_func(edge.src, edge.dst, labels[edge.src].data, labels[edge.dst].data, edge).first);
    }

    template <typename R, typename DataType, typename ContinueReduceFunc, typename UpdateFunc, typename ActiveResultFunc>
    R do_update_tree_add(
            ContinueReduceFunc continue_reduce_func, 
            UpdateFunc update_func, 
            ActiveResultFunc active_result_func, 
            std::vector<VertexTree<DataType>> &labels)
    {
        R total_result = 0;

        //task_arena.execute([&]()
        //{
        for(uint64_t i=0;true;i++)
        {
            active_out.clear();
            R local_result = stream_edges_hybrid<R>(
                [&](uint64_t src, const adjlist_range_type &outgoing_range)
                {
                    R result = 0;
                    for(auto iter = outgoing_range.first;iter != outgoing_range.second; iter++) 
                    {
                        auto edge = *iter;
                        if(edge.num > 0)
                        {
                            uint64_t dst = edge.nbr;
                            auto src_data = labels[src].data;
                            auto dst_data = labels[dst].data;
                            if(update_func(src, dst, src_data, dst_data, edge).first)
                            {
                                auto eup = edge; eup.nbr = src;
                                auto update_pair = update_label(labels, src, dst, eup, update_func);
                                if(update_pair.first)
                                {
                                    active_out.active(dst);
                                    if(trace_modified) modified.active(dst);
                                    result = active_result_func(result, src, dst, src_data, dst_data, update_pair.second);
                                }
                            }
                        }
                    }
                    return result;
                },
                [&](uint64_t dst, const adjlist_range_type &incoming_range)
                {
                    DataType new_label = labels[dst].data;
                    uint64_t new_src = (uint64_t)-1;
                    adjedge_type new_edge;
                    R result = 0;
                    for(auto iter = incoming_range.first;iter != incoming_range.second; iter++) 
                    {
                        auto edge = *iter;
                        if(edge.num > 0)
                        {
                            uint64_t src = edge.nbr;
                            auto src_data = labels[src].data;
                            auto update_pair = update_func(src, dst, src_data, new_label, edge);
                            if(update_pair.first)
                            {
                                new_label = update_pair.second;
                                new_src = src;
                                new_edge = edge;
                            }
                        }
                    }
                    if(new_src == (uint64_t)-1) return result;
                    auto src_data = labels[new_src].data;
                    auto dst_data = labels[dst].data;
                    if(update_func(new_src, dst, src_data, dst_data, new_edge).first)
                    {
                        auto src = new_src;
                        auto edge = new_edge;
                        auto eup = edge; eup.nbr = src;
                        auto update_pair = update_label(labels, src, dst, eup, update_func);
                        if(update_pair.first)
                        {
                            active_out.active(dst);
                            if(trace_modified) modified.active(dst);
                            result = active_result_func(result, src, dst, src_data, dst_data, update_pair.second);
                        }
                    }
                    return result;
                },
                [&](uint64_t src, const adjedge_type &edge)
                {
                    R result = 0;
                    if(edge.num > 0)
                    {
                        uint64_t dst = edge.nbr;
                        auto src_data = labels[src].data;
                        auto dst_data = labels[dst].data;
                        if(update_func(src, dst, src_data, dst_data, edge).first)
                        {
                            auto eup = edge; eup.nbr = src;
                            auto update_pair = update_label(labels, src, dst, eup, update_func);
                            if(update_pair.first)
                            {
                                active_out.active(dst);
                                if(trace_modified) modified.active(dst);
                                result = active_result_func(result, src, dst, src_data, dst_data, update_pair.second);
                            }
                        }
                    }
                    return result;
                },
                active_in
            );
            std::swap(active_in, active_out);
            bool is_continue;
            std::tie(is_continue, total_result) = continue_reduce_func(i, total_result, local_result);
            if(!is_continue) break;
        }
        //return 0ul;
        //});

        return total_result;
    }

    //template <typename R, typename DataType>
    //R update_tree_add(
    //        std::function<std::pair<bool, R>(uint64_t depth, R total_result, R local_result)> continue_reduce_func, 
    //        std::function<std::pair<bool, DataType>(uint64_t src, uint64_t dst, DataType src_data, DataType dst_data, adjedge_type adjedge)> update_func, 
    //        std::function<R(R old_result, uint64_t src, uint64_t dst, DataType src_data, DataType old_dst_data, DataType new_dst_data)> active_result_func, 
    //        std::vector<VertexTree<DataType>> &labels, edge_type edge, bool directed = true)
    template <typename R, typename DataType, typename ContinueReduceFunc, typename UpdateFunc, typename ActiveResultFunc>
    R update_tree_add(
            ContinueReduceFunc continue_reduce_func, 
            UpdateFunc update_func, 
            ActiveResultFunc active_result_func, 
            std::vector<VertexTree<DataType>> &labels, edge_type edge, bool directed = true)
    {
        if(!directed)
        {
            R retl = update_tree_add<R, DataType>(continue_reduce_func, update_func, active_result_func, labels, edge, true);
            std::swap(edge.src, edge.dst);
            R retr = update_tree_add<R, DataType>(continue_reduce_func, update_func, active_result_func, labels, edge, true);
            return continue_reduce_func(0, retl, retr).second;
        }

        R total_result = 0;
        if(update_func(edge.src, edge.dst, labels[edge.src].data, labels[edge.dst].data, edge).first) 
        {
            active_in.clear();
            adjedge_type eup = edge; eup.nbr = edge.src;
            auto src_data = labels[edge.src].data;
            auto dst_data = labels[edge.dst].data;
            auto update_pair = update_label(labels, edge.src, edge.dst, eup, update_func);
            if(update_pair.first)
            {
                active_in.active(edge.dst);
                if(trace_modified) modified.active(edge.dst);
                total_result = active_result_func(total_result, edge.src, edge.dst, src_data, dst_data, update_pair.second);
            }
        }
        else
        {
            return total_result;
        }

        return do_update_tree_add<R, DataType>(continue_reduce_func, update_func, active_result_func, labels);
    }

    template <typename R, typename DataType, typename ContinueReduceFunc, typename UpdateFunc, typename ActiveResultFunc>
    R update_tree_add(
            ContinueReduceFunc continue_reduce_func, 
            UpdateFunc update_func, 
            ActiveResultFunc active_result_func, 
            std::vector<VertexTree<DataType>> &labels, const std::vector<edge_type> &edges, const uint64_t &length, bool directed = true)
    {
        R total_result = 0;
        if(length == 1)
        {
            for(uint64_t i=0;i<length;i++)
            {
                auto edge = edges[i];
                total_result = update_tree_add<R>(continue_reduce_func, update_func, active_result_func, labels, edge, directed);
            }
            return total_result;
        }

        active_in.clear();
        THRESHOLD_OPENMP("omp parallel for", length, 
            for(uint64_t i=0;i<length;i++)
            {
                auto edge = edges[i];
                if(update_func(edge.src, edge.dst, labels[edge.src].data, labels[edge.dst].data, edge).first) 
                {
                    adjedge_type eup = edge; eup.nbr = edge.src;
                    auto src_data = labels[edge.src].data;
                    auto dst_data = labels[edge.dst].data;
                    auto update_pair = update_label(labels, edge.src, edge.dst, eup, update_func);
                    if(update_pair.first)
                    {
                        active_in.active(edge.dst);
                        if(trace_modified) modified.active(edge.dst);
                        total_result = active_result_func(total_result, edge.src, edge.dst, src_data, dst_data, update_pair.second);
                    }
                }
                if(directed) continue;
                std::swap(edge.src, edge.dst);
                if(update_func(edge.src, edge.dst, labels[edge.src].data, labels[edge.dst].data, edge).first) 
                {
                    adjedge_type eup = edge; eup.nbr = edge.src;
                    auto src_data = labels[edge.src].data;
                    auto dst_data = labels[edge.dst].data;
                    auto update_pair = update_label(labels, edge.src, edge.dst, eup, update_func);
                    if(update_pair.first)
                    {
                        active_in.active(edge.dst);
                        if(trace_modified) modified.active(edge.dst);
                        total_result = active_result_func(total_result, edge.src, edge.dst, src_data, dst_data, update_pair.second);
                    }
                }
            }
        );
        if(active_in.get_sparse_length() == 0) return total_result;

        return do_update_tree_add<R, DataType>(continue_reduce_func, update_func, active_result_func, labels);
    }

    template <typename DataType>
    bool need_update_tree_del(
            std::vector<VertexTree<DataType>> &labels, edge_type edge, bool directed = true)
    {
        if(!directed)
        {
            bool retl = need_update_tree_del<DataType>(labels, edge, true);
            std::swap(edge.src, edge.dst);
            bool retr = need_update_tree_del<DataType>(labels, edge, true);
            return retl || retr;
        }
        adjedge_type old_parent = edge; old_parent.nbr = edge.src;
        return (labels[edge.dst].parent == old_parent);
    }

    template <typename R, typename DataType, typename InitLabelFunc, typename ContinueReduceFunc, typename UpdateFunc, typename ActiveResultFunc, typename EqualFunc>
    R do_update_tree_del(
            InitLabelFunc init_label_func, 
            ContinueReduceFunc continue_reduce_func, 
            UpdateFunc update_func, 
            ActiveResultFunc active_result_func, 
            EqualFunc equal_func, 
            std::vector<VertexTree<DataType>> &labels)
    {
        R total_result = 0;

        //task_arena.execute([&]()
        //{
        uint64_t size_tree = 0;
        bool need_recompute = false;
        for(uint64_t i=0, active_vertices = active_in.get_sparse_length();i<vertices && active_vertices;i++)
        {
            if(need_recompute || (i <= 10 && size_tree+active_vertices > 0.05*vertices && size_tree+active_vertices > 2000000))
            {
                //auto total_depth = stream_vertices<uint64_t>(
                //    [&](uint64_t vid)
                //    {
                //        uint64_t depth = 0;
                //        for(;!(labels[vid].parent == empty_parent);vid = labels[vid].parent.nbr)
                //        {
                //            depth++;
                //        }
                //        return depth;
                //    },
                //    get_dense_active_all()
                //);
                //fprintf(stderr, "Depth: %lf\n", (double)total_depth/vertices);
                fprintf(stderr, "Re Computing: %lu %lu %lu\n", i, size_tree, active_in.get_sparse_length());
                stream_vertices<uint64_t>(
                    [&](uint64_t vid)
                    {
                        invalidated[vid] = 0;
                        return 0;
                    },
                    get_dense_active_all()
                );
                DataType *backup_labels = (DataType*)mmap_alloc(sizeof(DataType)*vertices);
                if(trace_modified)
                {
                    stream_vertices<uint64_t>(
                        [&](uint64_t vid)
                        {
                            backup_labels[vid] = labels[vid].data;
                            return 0;
                        },
                        get_dense_active_all()
                    );
                }
                auto ret = build_tree<R, DataType>(
                    init_label_func,
                    continue_reduce_func,
                    update_func,
                    active_result_func,
                    labels
                );
                if(trace_modified)
                {
                    stream_vertices<uint64_t>(
                        [&](uint64_t vid)
                        {
                            if(trace_modified && labels[vid].data != backup_labels[vid]) 
                                modified.active(vid);
                            return 0;
                        },
                        get_dense_active_all()
                    );
                }
                mmap_free(backup_labels, sizeof(DataType)*vertices);
                //total_depth = stream_vertices<uint64_t>(
                //    [&](uint64_t vid)
                //    {
                //        uint64_t depth = 0;
                //        for(;!(labels[vid].parent == empty_parent);vid = labels[vid].parent.nbr)
                //        {
                //            depth++;
                //        }
                //        return depth;
                //    },
                //    get_dense_active_all()
                //);
                //fprintf(stderr, "Depth: %lf\n", (double)total_depth/vertices);
                return ret;
            }
            transpose();
            if(i <= 5 || active_vertices < 8192*2) 
            {
                stream_edges_hybrid<uint64_t>(
                    [&](uint64_t dst, const adjlist_range_type &incoming_range)
                    {
                        R result = 0;
                        for(auto iter = incoming_range.first;iter != incoming_range.second; iter++) 
                        {
                            auto edge = *iter;
                            if(edge.num > 0)
                            {
                                uint64_t src = edge.nbr;
                                auto src_data = labels[src].data;
                                auto dst_data = labels[dst].data;
                                if(equal_func(src, dst, src_data, dst_data, edge) && invalidated[dst] == invalidated_idx && !has_parents(labels, src, need_recompute) && cas(&invalidated[dst], invalidated_idx, -invalidated_idx))
                                {
                                    labels[dst].parent = edge;
                                }
                            }
                        }
                        return result;
                    },
                    nullptr,
                    [&](uint64_t dst, const adjedge_type &edge)
                    {
                        uint64_t src = edge.nbr;
                        auto src_data = labels[src].data;
                        auto dst_data = labels[dst].data;
                        if(edge.num > 0 && equal_func(src, dst, src_data, dst_data, edge) && invalidated[dst] == invalidated_idx && !has_parents(labels, src, need_recompute) && cas(&invalidated[dst], invalidated_idx, -invalidated_idx))
                        {
                            labels[dst].parent = edge;
                        }
                        return 0;
                    },
                    active_in
                );
            }
            transpose();
            if(need_recompute) continue;
            active_out.clear();
            size_tree += stream_vertices_hybrid<uint64_t>(
                [&](uint64_t vid)
                {
                    if(invalidated[vid] == invalidated_idx)
                    {
                        active_out.active(vid);
                        active_tree.active(vid);
                        return 1;
                    }
                    else if(invalidated[vid] == -invalidated_idx)
                    {
                        invalidated[vid] = 0;
                    }
                    return 0;
                },
                active_in
            );
            active_in.clear();
            active_vertices = stream_edges_hybrid<uint64_t>(
                [&](uint64_t src, const adjlist_range_type &outgoing_range)
                {
                    R result = 0;
                    uint64_t activated = 0;
                    for(auto iter = outgoing_range.first;iter != outgoing_range.second; iter++) 
                    {
                        auto edge = *iter;
                        if(edge.num > 0)
                        {
                            uint64_t dst = edge.nbr;
                            edge.nbr = src;
                            if(labels[dst].parent == edge && cas(&invalidated[dst], 0l, invalidated_idx))
                            {
                                active_in.active(dst);
                                activated ++;
                            }
                        }
                    }
                    return activated;
                },
                nullptr,
                [&](uint64_t src, adjedge_type edge)
                {
                    uint64_t activated = 0;
                    uint64_t dst = edge.nbr;
                    edge.nbr = src;
                    if(labels[dst].parent == edge && cas(&invalidated[dst], 0l, invalidated_idx))
                    {
                        active_in.active(dst);
                        activated ++;
                    }
                    return activated;
                },
                active_out
            );
        }

        stream_vertices_hybrid<uint64_t>(
            [&](uint64_t vid)
            {
                labels[vid].data = init_label_func(vid).first;
                labels[vid].parent = empty_parent;
                if(trace_modified) modified.active(vid);
                invalidated[vid] = 0;
                return 1;
            },
            active_tree
        );
        std::swap(active_in, active_tree);

        transpose();
        R local_result = stream_edges_hybrid<R>(
            [&](uint64_t dst, const adjlist_range_type &incoming_range)
            {
                DataType new_label = labels[dst].data;
                uint64_t new_src = (uint64_t)-1;
                adjedge_type new_edge;
                R result = 0;
                for(auto iter = incoming_range.first;iter != incoming_range.second; iter++) 
                {
                    auto edge = *iter;
                    if(edge.num > 0)
                    {
                        uint64_t src = edge.nbr;
                        auto src_data = labels[src].data;
                        auto update_pair = update_func(src, dst, src_data, new_label, edge);
                        if(update_pair.first)
                        {
                            new_label = update_pair.second;
                            new_src = src;
                            new_edge = edge;
                        }
                    }
                }
                if(new_src == (uint64_t)-1) return result;
                auto src_data = labels[new_src].data;
                auto dst_data = labels[dst].data;
                if(update_func(new_src, dst, src_data, dst_data, new_edge).first)
                {
                    auto src = new_src;
                    auto edge = new_edge;
                    auto eup = edge; eup.nbr = src;
                    auto update_pair = update_label(labels, src, dst, eup, update_func);
                    if(update_pair.first)
                    {
                        result = active_result_func(result, src, dst, src_data, dst_data, update_pair.second);
                    }
                }
                return result;
            },
            nullptr,
            [&](uint64_t dst, const adjedge_type &edge)
            {
                DataType new_label = labels[dst].data;
                uint64_t new_src = (uint64_t)-1;
                adjedge_type new_edge;
                R result = 0;
                if(edge.num > 0)
                {
                    uint64_t src = edge.nbr;
                    auto src_data = labels[src].data;
                    auto update_pair = update_func(src, dst, src_data, new_label, edge);
                    if(update_pair.first)
                    {
                        new_label = update_pair.second;
                        new_src = src;
                        new_edge = edge;
                    }
                }
                if(new_src == (uint64_t)-1) return result;
                auto src_data = labels[new_src].data;
                auto dst_data = labels[dst].data;
                if(update_func(new_src, dst, src_data, dst_data, new_edge).first)
                {
                    auto src = new_src;
                    auto edge = new_edge;
                    auto eup = edge; eup.nbr = src;
                    auto update_pair = update_label(labels, src, dst, eup, update_func);
                    if(update_pair.first)
                    {
                        result = active_result_func(result, src, dst, src_data, dst_data, update_pair.second);
                    }
                }
                return result;
            },
            active_in
        );
        transpose();
        total_result = std::get<1>(continue_reduce_func(0, total_result, local_result));

        for(uint64_t i=0;true;i++)
        {
            active_out.clear();
            R local_result = stream_edges_hybrid<R>(
                [&](uint64_t src, const adjlist_range_type &outgoing_range)
                {
                    R result = 0;
                    for(auto iter = outgoing_range.first;iter != outgoing_range.second; iter++) 
                    {
                        auto edge = *iter;
                        if(edge.num > 0)
                        {
                            uint64_t dst = edge.nbr;
                            auto src_data = labels[src].data;
                            auto dst_data = labels[dst].data;
                            if(update_func(src, dst, src_data, dst_data, edge).first)
                            {
                                auto eup = edge; eup.nbr = src;
                                auto update_pair = update_label(labels, src, dst, eup, update_func);
                                if(update_pair.first)
                                {
                                    active_out.active(dst);
                                    result = active_result_func(result, src, dst, src_data, dst_data, update_pair.second);
                                }
                            }
                        }
                    }
                    return result;
                },
                [&](uint64_t dst, const adjlist_range_type &incoming_range)
                {
                    DataType new_label = labels[dst].data;
                    uint64_t new_src = (uint64_t)-1;
                    adjedge_type new_edge;
                    R result = 0;
                    for(auto iter = incoming_range.first;iter != incoming_range.second; iter++) 
                    {
                        auto edge = *iter;
                        if(edge.num > 0)
                        {
                            uint64_t src = edge.nbr;
                            auto src_data = labels[src].data;
                            auto update_pair = update_func(src, dst, src_data, new_label, edge);
                            if(update_pair.first)
                            {
                                new_label = update_pair.second;
                                new_src = src;
                                new_edge = edge;
                            }
                        }
                    }
                    if(new_src == (uint64_t)-1) return result;
                    auto src_data = labels[new_src].data;
                    auto dst_data = labels[dst].data;
                    if(update_func(new_src, dst, src_data, dst_data, new_edge).first)
                    {
                        auto src = new_src;
                        auto edge = new_edge;
                        auto eup = edge; eup.nbr = src;
                        auto update_pair = update_label(labels, src, dst, eup, update_func);
                        if(update_pair.first)
                        {
                            active_out.active(dst);
                            result = active_result_func(result, src, dst, src_data, dst_data, update_pair.second);
                        }
                    }
                    return result;
                },
                [&](uint64_t src, const adjedge_type &edge)
                {
                    R result = 0;
                    if(edge.num > 0)
                    {
                        uint64_t dst = edge.nbr;
                        auto src_data = labels[src].data;
                        auto dst_data = labels[dst].data;
                        if(update_func(src, dst, src_data, dst_data, edge).first)
                        {
                            auto eup = edge; eup.nbr = src;
                            auto update_pair = update_label(labels, src, dst, eup, update_func);
                            if(update_pair.first)
                            {
                                active_out.active(dst);
                                result = active_result_func(result, src, dst, src_data, dst_data, update_pair.second);
                            }
                        }
                    }
                    return result;
                },
                active_in
            );
            std::swap(active_in, active_out);
            bool is_continue;
            std::tie(is_continue, total_result) = continue_reduce_func(i, total_result, local_result);
            if(!is_continue) break;
        }
        //return 0ul;
        //});
        return total_result;
    }

    //template <typename R, typename DataType>
    //R update_tree_del(
    //        std::function<std::pair<DataType, bool>(uint64_t vid)> init_label_func, 
    //        std::function<std::pair<bool, R>(uint64_t depth, R total_result, R local_result)> continue_reduce_func, 
    //        std::function<std::pair<bool, DataType>(uint64_t src, uint64_t dst, DataType src_data, DataType dst_data, adjedge_type adjedge)> update_func, 
    //        std::function<R(R old_result, uint64_t src, uint64_t dst, DataType src_data, DataType old_dst_data, DataType new_dst_data)> active_result_func, 
    //        std::function<bool(uint64_t src, uint64_t dst, DataType src_data, DataType dst_data, adjedge_type adjedge)> equal_func, 
    //        std::vector<VertexTree<DataType>> &labels, edge_type edge, bool directed = true)
    template <typename R, typename DataType, typename InitLabelFunc, typename ContinueReduceFunc, typename UpdateFunc, typename ActiveResultFunc, typename EqualFunc>
    R update_tree_del(
            InitLabelFunc init_label_func, 
            ContinueReduceFunc continue_reduce_func, 
            UpdateFunc update_func, 
            ActiveResultFunc active_result_func, 
            EqualFunc equal_func, 
            std::vector<VertexTree<DataType>> &labels, edge_type edge, bool directed = true)
    {
        if(!directed)
        {
            R retl = update_tree_del<R, DataType>(init_label_func, continue_reduce_func, update_func, active_result_func, equal_func, labels, edge, true);
            std::swap(edge.src, edge.dst);                         
            R retr = update_tree_del<R, DataType>(init_label_func, continue_reduce_func, update_func, active_result_func, equal_func, labels, edge, true);
            return continue_reduce_func(0, retl, retr).second;
        }

        R total_result = 0;
        DataType old_label = labels[edge.dst].data;
        adjedge_type old_parent = edge; old_parent.nbr = edge.src;
        if(labels[edge.dst].parent == old_parent) 
        {
            active_in.clear();
            active_tree.clear();
            active_in.active(edge.dst);
            invalidated_idx++;
            invalidated[edge.dst] = invalidated_idx;
        }
        else
        {
            return total_result;
        }

        return do_update_tree_del<R, DataType>(init_label_func, continue_reduce_func, update_func, active_result_func, equal_func, labels);
    }

    template <typename R, typename DataType, typename InitLabelFunc, typename ContinueReduceFunc, typename UpdateFunc, typename ActiveResultFunc, typename EqualFunc>
    R update_tree_del(
            InitLabelFunc init_label_func, 
            ContinueReduceFunc continue_reduce_func, 
            UpdateFunc update_func, 
            ActiveResultFunc active_result_func, 
            EqualFunc equal_func, 
            std::vector<VertexTree<DataType>> &labels, const std::vector<edge_type> &edges, const uint64_t &length, bool directed = true)
    {
        R total_result = 0;
        if(length == 1)
        {
            for(uint64_t i=0;i<length;i++)
            {
                auto edge = edges[i];
                total_result = update_tree_del<R>(init_label_func, continue_reduce_func, update_func, active_result_func, equal_func, labels, edge, directed);
            }
            return total_result;
        }

        active_in.clear();
        active_tree.clear();
        invalidated_idx++;
        THRESHOLD_OPENMP("omp parallel for", length, 
            for(uint64_t i=0;i<length;i++)
            {
                auto edge = edges[i];
                {
                    uint64_t old_root = edge.dst;
                    adjedge_type old_parent = edge; old_parent.nbr = edge.src;
                    if(labels[edge.dst].parent == old_parent)
                    {
                        auto old_val = invalidated[old_root];
                        auto new_val = invalidated_idx;
                        if(old_val != new_val && cas(&invalidated[old_root], old_val, new_val))
                        {
                            active_in.active(old_root);
                        }
                    }
                }
                if(directed) continue;
                std::swap(edge.src, edge.dst);
                {
                    uint64_t old_root = edge.dst;
                    adjedge_type old_parent = edge; old_parent.nbr = edge.src;
                    if(labels[edge.dst].parent == old_parent)
                    {
                        auto old_val = invalidated[old_root];
                        auto new_val = invalidated_idx;
                        if(old_val != new_val && cas(&invalidated[old_root], old_val, new_val))
                        {
                            active_in.active(old_root);
                        }
                    }
                }
            }
        );
        if(active_in.get_sparse_length() == 0) return total_result;

        return do_update_tree_del<R, DataType>(init_label_func, continue_reduce_func, update_func, active_result_func, equal_func, labels);
    }

private:
    const uint64_t vertices;
    const bool symmetric, dual;
    const uint64_t dense_threshold;

    Storage outgoing, incoming;
    std::atomic_uint64_t edges;
    Bitmap dense_active_all;
    ActiveSet active_in, active_out, active_tree, modified;
    std::vector<int64_t> invalidated;
    int64_t invalidated_idx;
    std::vector<uint64_t> offsets;
    adjedge_type empty_parent;
    std::mutex mutex;
    bool trace_modified;
    //tbb::task_arena task_arena;
    tbb::affinity_partitioner affinity_partitioner;
    tbb::enumerable_thread_specific<uint64_t> thread_id;

    //template <typename DataType>
    //std::pair<bool, DataType> update_label_raw(std::vector<DataType> &labels, uint64_t src, uint64_t dst, adjedge_type eup, std::function<std::pair<bool, DataType>(uint64_t src, uint64_t dst, DataType src_data, DataType dst_data, adjedge_type aejedge)> update_func)
    template<typename DataType, typename UpdateFunc>
    std::pair<bool, DataType> update_label_raw(std::vector<DataType> &labels, uint64_t src, uint64_t dst, adjedge_type eup, UpdateFunc update_func)
    {
        const DataType src_data = labels[src];
        std::pair<bool, DataType> ret; ret.first = false;
        while(true)
        {
            const DataType old_val = labels[dst];
            auto update_pair = update_func(src, dst, src_data, old_val, eup);
            if(update_pair.first)
            {
                if(!cas(&labels[dst], old_val, update_pair.second)) continue;
                ret.first = true; ret.second = update_pair.second;
            }
            else
            {
                break;
            }
        }
        return ret;
    };

    //template <typename DataType>
    //std::pair<bool, DataType> update_label(std::vector<VertexTree<DataType>> &labels, uint64_t src, uint64_t dst, adjedge_type eup, std::function<std::pair<bool, DataType>(uint64_t src, uint64_t dst, DataType src_data, DataType dst_data, adjedge_type aejedge)> update_func)
    template<typename DataType, typename UpdateFunc>
    inline typename std::enable_if<!(sizeof(VertexTree<DataType>) <= 16 && std::is_scalar<DataType>::value), std::pair<bool, DataType>>::type update_label(std::vector<VertexTree<DataType>> &labels, uint64_t src, uint64_t dst, adjedge_type eup, UpdateFunc update_func)
    {
        std::unique_lock<lock_type> lock(get_lock(dst), std::defer_lock);
        const DataType src_data = labels[src].data;
        std::pair<bool, DataType> ret; ret.first = false;
        while(true)
        {
            uint32_t status;
#ifdef ARCH_HAS_RTM
            if ((status = _xbegin()) != _XBEGIN_STARTED)
            {
                if(status == _XABORT_CONFLICT || status == _XABORT_RETRY) continue;
                lock.lock();
            }
#else
            lock.lock();
#endif
            compiler_fence();
            auto update_pair = update_func(src, dst, src_data, labels[dst].data, eup);
            if(update_pair.first)
            {
                labels[dst].data = update_pair.second;
                labels[dst].parent = eup;
                ret.first = true; ret.second = update_pair.second;
            }
#ifdef ARCH_HAS_RTM
            if(_xtest()) _xend();
#endif
            break;
        }
        return ret;
    };

    template<typename DataType, typename UpdateFunc>
    inline typename std::enable_if<sizeof(VertexTree<DataType>) <= 16 && std::is_scalar<DataType>::value, std::pair<bool, DataType>>::type update_label(std::vector<VertexTree<DataType>> &labels, uint64_t src, uint64_t dst, adjedge_type eup, UpdateFunc update_func)
    {
        const DataType src_data = labels[src].data;
        std::pair<bool, DataType> ret; ret.first = false;
        while(true)
        {
            compiler_fence();
            auto old_val = labels[dst];
            auto update_pair = update_func(src, dst, src_data, old_val.data, eup);
            if(update_pair.first)
            {
                VertexTree<DataType> new_val = {eup, update_pair.second};
                bool success = cas(&labels[dst], old_val, new_val);
                if(success)
                {
                    ret.first = true;
                    ret.second = update_pair.second;
                }
                else
                {
                    continue;
                }
            }
            break;
        }
        return ret;
    }

    template<typename DataType, typename UpdateFunc>
    inline typename std::enable_if<!(sizeof(VertexTree<DataType>) <= 16 && std::is_scalar<DataType>::value), std::pair<bool, DataType>>::type update_label(std::vector<VertexTree<DataType>> &labels, uint64_t src, uint64_t dst, adjedge_type eup, UpdateFunc update_func, DataType src_data)
    {
        std::unique_lock<lock_type> lock(get_lock(dst), std::defer_lock);
        std::pair<bool, DataType> ret; ret.first = false;
        while(true)
        {
            uint32_t status;
#ifdef ARCH_HAS_RTM
            if ((status = _xbegin()) != _XBEGIN_STARTED)
            {
                if(status == _XABORT_CONFLICT || status == _XABORT_RETRY) continue;
                lock.lock();
            }
#else
            lock.lock();
#endif
            compiler_fence();
            auto update_pair = update_func(src, dst, src_data, labels[dst].data, eup);
            if(update_pair.first)
            {
                labels[dst].data = update_pair.second;
                labels[dst].parent = eup;
                ret.first = true; ret.second = update_pair.second;
            }
#ifdef ARCH_HAS_RTM
            if(_xtest()) _xend();
#endif
            break;
        }
        return ret;
    };

    template<typename DataType, typename UpdateFunc>
    inline typename std::enable_if<sizeof(VertexTree<DataType>) <= 16 && std::is_scalar<DataType>::value, std::pair<bool, DataType>>::type update_label(std::vector<VertexTree<DataType>> &labels, uint64_t src, uint64_t dst, adjedge_type eup, UpdateFunc update_func, DataType src_data)
    {
        std::pair<bool, DataType> ret; ret.first = false;
        while(true)
        {
            compiler_fence();
            auto old_val = labels[dst];
            auto update_pair = update_func(src, dst, src_data, old_val.data, eup);
            if(update_pair.first)
            {
                VertexTree<DataType> new_val = {eup, update_pair.second};
                bool success = cas(&labels[dst], old_val, new_val);
                if(success)
                {
                    ret.first = true;
                    ret.second = update_pair.second;
                }
                else
                {
                    continue;
                }
            }
            break;
        }
        return ret;
    }

    template<typename DataType>
    bool has_parents(std::vector<VertexTree<DataType>> &labels, uint64_t x, bool &need_recompute)
    {
        uint64_t num = 0;
        while(true)
        {
            if(invalidated[x] == invalidated_idx || invalidated[x] == -invalidated_idx) return true;
            if(labels[x].parent.nbr != vertices)
                x = labels[x].parent.nbr;
            else
                break;
            if(++num > vertices*0.1) 
            {
                need_recompute = true;//throw std::runtime_error("There is a cycle in parents.");
                return true;
            }
        }
        return false;
    }
};
