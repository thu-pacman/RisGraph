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
    if(argc <= 6)
    {
        fprintf(stderr, "usage: %s graph root imported_rate timeout_ms percent clients\n", argv[0]);
        exit(1);
    }
    std::pair<uint64_t, uint64_t> *raw_edges;
    uint64_t raw_edges_len;
    std::tie(raw_edges, raw_edges_len) = mmap_binary(argv[1]);
    uint64_t root = std::stoull(argv[2]);
    const uint32_t timeout_ms = std::stoi(argv[4]);
    const double target_timeout_rate = 1-std::stod(argv[5]);
    const uint64_t clients = std::stoull(argv[6]);
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
    Graph<void> graph(num_vertices, raw_edges_len, false, true, true);
    //std::random_shuffle(raw_edges.begin(), raw_edges.end());
    double imported_rate = std::stod(argv[3]);
    uint64_t imported_edges = raw_edges_len*imported_rate;
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

    auto labels = graph.alloc_vertex_tree_array<uint64_t>();
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

    struct Update
    {
        enum class Type
        {
            Add,
            Del
        } type;
        uint32_t timeout_ms;
        decltype(graph)::edge_type edge;
        std::chrono::high_resolution_clock::time_point *time;
    };

    std::atomic_uint64_t add_edge_len(0), del_edge_len(0);
    std::vector<Update> add_edge, del_edge;
    std::mutex add_mutex, del_mutex;
    uint64_t add_global_begin = imported_edges, del_global_begin = 0;
    uint64_t total_add_size = raw_edges_len-imported_edges, total_del_size = (raw_edges_len-imported_edges)*1;
    std::atomic_bool add_setted = false , del_setted = false;
    std::chrono::high_resolution_clock::time_point add_first, del_first;

    const uint64_t max_threads = omp_get_max_threads();
    uint64_t num_changes_threshold = max_threads;
    auto need_break_func = [&](std::chrono::high_resolution_clock::time_point time)
    {
        uint64_t num_changes = add_edge_len.load(std::memory_order_relaxed) + del_edge_len.load(std::memory_order_relaxed);
        auto min_time = time;
        if(add_setted.load(std::memory_order_acquire)) min_time = std::min(min_time, add_first);
        if(del_setted.load(std::memory_order_acquire)) min_time = std::min(min_time, del_first);
        return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(time-min_time).count() > 0.8*timeout_ms || num_changes > num_changes_threshold;
        //return num_changes > num_changes_threshold;
    };

    const uint64_t clients_per_threads = (clients+max_threads-1)/max_threads;
    tbb::enumerable_thread_specific<std::vector<std::chrono::high_resolution_clock::time_point>> all_last_time(clients_per_threads);
    tbb::enumerable_thread_specific<std::vector<uint64_t>> all_is_blocked(clients_per_threads);
    tbb::enumerable_thread_specific<std::pair<uint64_t, uint64_t>> all_next_client;
    tbb::enumerable_thread_specific<std::pair<uint64_t, uint64_t>> all_current_batch;
    tbb::enumerable_thread_specific<std::vector<uint64_t>> all_latency;
    //std::vector<uint64_t> batch_time;
    std::atomic_uint64_t num_timeouts = 0, num_updates = 0;
    uint64_t last_num_timeouts = 0, last_num_updates = 0, num_loops = 0;
    uint64_t sum_nanoseconds = 0;

    auto history_labels = graph.alloc_history_array<uint64_t>();
    uint64_t label_version = 0;
    auto global_begin_time = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        auto &last_time = all_last_time.local();
        auto &is_blocked = all_is_blocked.local();
        for(auto &t : last_time) t = global_begin_time;
        for(auto &f : is_blocked) f = num_loops;
        uint64_t local_begin = clients_per_threads*omp_get_thread_num();
        if(local_begin < clients)
        {
            uint64_t num_clients = std::min(clients_per_threads*(omp_get_thread_num()+1), clients)-local_begin;
            all_next_client.local() = std::make_pair(0ul, num_clients);
        }
        else
        {
            all_next_client.local() = std::make_pair(0ul, 0ul);
        }
        all_current_batch.local() = std::make_pair(add_global_begin+omp_get_thread_num(), del_global_begin+omp_get_thread_num());
    }
    {
        auto start = std::chrono::system_clock::now();
        while(num_updates.load() < total_add_size+total_del_size)
        {
            num_loops++;
            auto start = std::chrono::system_clock::now();
            std::atomic_uint64_t batch_size = 0;
            //uint64_t cur_batch = global_batch;
            {
                auto start = std::chrono::system_clock::now();
                add_edge_len = 0; del_edge_len = 0;
                add_setted = false; del_setted = false;
                std::atomic_int stopped_clients = 0;
                #pragma omp parallel reduction(+: sum_nanoseconds)
                {
                    bool is_break = false;
                    int num_threads = omp_get_num_threads();
                    auto &last_time = all_last_time.local();
                    auto &is_blocked = all_is_blocked.local();
                    uint64_t next_client = all_next_client.local().first;
                    uint64_t num_clients = all_next_client.local().second;
                    uint64_t left_clients = num_clients;
                    bool is_add = num_loops & 1;
                    bool add_end = false, del_end = false;
                    uint64_t local_batch_size = 0;
                    uint64_t add_batch = all_current_batch.local().first;
                    uint64_t del_batch = all_current_batch.local().second;
                    auto &latency = all_latency.local();
                    while(num_clients && (!add_end || !del_end))
                    {
                        is_add = !is_add;
                        auto time = std::chrono::high_resolution_clock::now();

                        bool need_break = need_break_func(time) || !left_clients; 

                        if(need_break && !is_break)
                        {
                            is_break = true;
                            stopped_clients.fetch_add(1, std::memory_order_relaxed);
                        }
                        if(is_break)// && stopped_clients.load(std::memory_order_relaxed) == num_threads)
                        {
                            break;
                        }

                        while(is_blocked[next_client = (next_client+1)%num_clients] == num_loops);
                        //while(is_blocked[next_client] == num_loops) next_client = (next_client+1)%num_clients;
                        
                        if(is_add)
                        {
                            const uint64_t i = add_batch;
                            add_batch += num_threads;
                            if(i > raw_edges_len)
                            {
                                add_end = true;
                                continue;
                            }
                            local_batch_size += 1;
                            {
                                decltype(graph)::edge_type edge = {raw_edges[i].first, raw_edges[i].second};
                                if(graph.need_update_tree_add(update_func, labels, edge, true))
                                {
                                    auto idx = atomic_append(add_edge, add_edge_len, Update{Update::Type::Add, timeout_ms, edge, &last_time[next_client]}, add_mutex);
                                    if(idx == 0)
                                    {
                                        add_first = time;
                                        add_setted.store(true, std::memory_order_release);
                                    }
                                    is_blocked[next_client] = num_loops;
                                    left_clients--;
                                }
                                else
                                {
                                    graph.add_edge(edge, true);
                                    auto time = std::chrono::high_resolution_clock::now();
                                    latency.emplace_back((time - last_time[next_client]).count());
                                    sum_nanoseconds += (time - last_time[next_client]).count();
                                    if(time > last_time[next_client]+std::chrono::milliseconds(timeout_ms))
                                    {
                                        num_timeouts++;
                                    }
                                    last_time[next_client] = time;
                                }

                            }
                        }
                        else
                        {
                            const uint64_t i = del_batch;
                            del_batch += num_threads;
                            if(i > total_del_size)
                            {
                                del_end = true;
                                continue;
                            }
                            local_batch_size += 1;
                            {
                                decltype(graph)::edge_type edge = {raw_edges[i].first, raw_edges[i].second};
                                if(graph.need_update_tree_del(labels, edge, true))
                                {
                                    auto idx = atomic_append(del_edge, del_edge_len, Update{Update::Type::Del, timeout_ms, edge, &last_time[next_client]}, del_mutex);
                                    if(idx == 0)
                                    {
                                        del_first = time;
                                        del_setted.store(true, std::memory_order_release);
                                    }
                                    is_blocked[next_client] = num_loops;
                                    left_clients--;
                                }
                                else
                                {
                                    graph.del_edge(edge, true);
                                    auto time = std::chrono::high_resolution_clock::now();
                                    latency.emplace_back((time - last_time[next_client]).count());
                                    sum_nanoseconds += (time - last_time[next_client]).count();
                                    if(time > last_time[next_client]+std::chrono::milliseconds(timeout_ms))
                                    {
                                        num_timeouts++;
                                    }
                                    last_time[next_client] = time;
                                }
                            }
                        }
                    }
                    batch_size.fetch_add(local_batch_size, std::memory_order_relaxed);
                    num_updates.fetch_add(local_batch_size, std::memory_order_relaxed);
                    all_next_client.local().first = next_client;
                    all_current_batch.local() = std::make_pair(add_batch, del_batch);
                }
                auto end = std::chrono::system_clock::now();
                fprintf(stderr, "exec safe: %.6lfs, ", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
                fprintf(stderr, "unsafe_add: %6lu, ", add_edge_len.load());
                fprintf(stderr, "unsafe_del: %6lu, ", del_edge_len.load());
            }
            {
                auto &latency = all_latency.local();
                auto start = std::chrono::system_clock::now();
                uint64_t i_add = 0, i_del = 0;
                uint64_t n_add = add_edge_len.load(), n_del = del_edge_len.load();
                while(i_add < n_add || i_del < n_del)
                {
                    if(i_del >= n_del || (i_add < n_add && add_edge[i_add].time < del_edge[i_del].time))
                    {
                        label_version++;
                        graph.clear_modified();
                        const auto &e = add_edge[i_add].edge;
                        if(graph.add_edge(e, true) == 0)
                        {
                            graph.update_tree_add<uint64_t, uint64_t>(
                                continue_reduce_func,
                                update_func,
                                active_result_func,
                                labels, e, true
                            );
                        }
                        graph.stream_vertices_hybrid<uint64_t>(
                            [&](uint64_t vid)
                            {
                                history_labels.set(vid, label_version, labels[vid].data);
                                return 0;
                            }, graph.get_modified());
                        auto time = std::chrono::high_resolution_clock::now();
                        latency.emplace_back((time-*add_edge[i_add].time).count());
                        sum_nanoseconds += (time-*add_edge[i_add].time).count();
                        if(time > *add_edge[i_add].time+std::chrono::milliseconds(add_edge[i_add].timeout_ms))
                        {
                            num_timeouts++;
                        }
                        *add_edge[i_add].time = time;
                        i_add++;
                    }
                    else
                    {
                        label_version++;
                        graph.clear_modified();
                        const auto &e = del_edge[i_del].edge;
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
                        graph.stream_vertices_hybrid<uint64_t>(
                            [&](uint64_t vid)
                            {
                                history_labels.set(vid, label_version, labels[vid].data);
                                return 0;
                            }, graph.get_modified());
                        auto time = std::chrono::high_resolution_clock::now();
                        latency.emplace_back((time-*del_edge[i_del].time).count());
                        sum_nanoseconds += (time-*del_edge[i_del].time).count();
                        if(time > *del_edge[i_del].time+std::chrono::milliseconds(del_edge[i_del].timeout_ms))
                        {
                            num_timeouts++;
                        }
                        *del_edge[i_del].time = time;
                        i_del++;
                    }
                }
                auto end = std::chrono::system_clock::now();
                fprintf(stderr, "exec unsafe: %.6lfs, ", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
            }
            auto end = std::chrono::system_clock::now();
            fprintf(stderr, "batch_size: %lu, ", batch_size.load(std::memory_order_relaxed));
            fprintf(stderr, "timeouts: %lu/%lu = 1-%lf, ", num_timeouts.load(std::memory_order_relaxed), num_updates.load(std::memory_order_relaxed),
                    1-(double)num_timeouts.load(std::memory_order_relaxed)/num_updates.load(std::memory_order_relaxed));
            if(num_loops && num_loops % 3 == 0)
            {
                uint64_t global_num_timeouts = num_timeouts.load(std::memory_order_relaxed);
                uint64_t global_num_updates = num_updates.load(std::memory_order_relaxed);
                uint64_t local_num_timeouts = global_num_timeouts-last_num_timeouts;
                uint64_t local_num_updates = global_num_updates-last_num_updates;
                if((double)local_num_timeouts/local_num_updates > 0.8*target_timeout_rate)
                {
                    num_changes_threshold *= 0.9;
                    num_changes_threshold = std::max(num_changes_threshold, max_threads);
                }
                if((double)local_num_timeouts/local_num_updates < 0.8*target_timeout_rate && (double)global_num_timeouts/global_num_updates < target_timeout_rate)
                {
                    if(num_changes_threshold < 1000) num_changes_threshold += 10; else num_changes_threshold *= 1.01;
                    num_changes_threshold = std::min(num_changes_threshold, 1000000lu);
                }
                last_num_timeouts += local_num_timeouts;
                last_num_updates += local_num_updates;
            }
            fprintf(stderr, "threshold: %lu, ", num_changes_threshold);
            fprintf(stderr, "mean latency: %lf us, ", (1e-3)*sum_nanoseconds/num_updates.load(std::memory_order_relaxed));
            fprintf(stderr, "exec: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
        }
        auto end = std::chrono::system_clock::now();
        std::vector<uint64_t> latency;
        for(auto &p : all_latency)
        {
            for(auto t : p) latency.emplace_back(t);
            p.clear();
        }
        tbb::parallel_sort(latency.begin(), latency.end());
        fprintf(stderr, "exec:       %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
        fprintf(stderr, "Throughput: %lf\n", num_updates.load(std::memory_order_relaxed)/(1e-9*(end-start).count()));
        fprintf(stderr, "Success:    %lf\n", 1-(double)num_timeouts.load(std::memory_order_relaxed)/num_updates.load(std::memory_order_relaxed));
        fprintf(stderr, "Mean:       %lf us\n", (1e-3)*sum_nanoseconds/num_updates.load(std::memory_order_relaxed));
        fprintf(stderr, "P25:        %lfus\n", 1e-3*latency[latency.size()*0.25]);
        fprintf(stderr, "P50:        %lfus\n", 1e-3*latency[latency.size()*0.50]);
        fprintf(stderr, "P75:        %lfus\n", 1e-3*latency[latency.size()*0.75]);
        fprintf(stderr, "P90:        %lfus\n", 1e-3*latency[latency.size()*0.90]);
        fprintf(stderr, "P99:        %lfus\n", 1e-3*latency[latency.size()*0.99]);
        fprintf(stderr, "P999:       %lfus\n", 1e-3*latency[latency.size()*0.999]);
        fprintf(stderr, "P9999:      %lfus\n", 1e-3*latency[latency.size()*0.9999]);
        fprintf(stderr, "P99999:     %lfus\n", 1e-3*latency[latency.size()*0.99999]);
        fprintf(stderr, "P999999:    %lfus\n", 1e-3*latency[latency.size()*0.999999]);
        fprintf(stderr, "P9999999:   %lfus\n", 1e-3*latency[latency.size()*0.9999999]);
        fprintf(stderr, "P99999999:  %lfus\n", 1e-3*latency[latency.size()*0.99999999]);
        fprintf(stderr, "Max:        %lfus\n", 1e-3*latency[latency.size()-1]);
    }

    //std::sort(batch_time.begin(), batch_time.end());
    //uint64_t sum_time = 0;
    //for(auto t:batch_time) sum_time += t;
    //fprintf(stderr, "mean: %lfs, 75: %lfs, 99: %lfs, 99.9: %lf, 99.99:%lf, max:%lf\n", 1e-6*sum_time/batch_time.size(), 1e-6*batch_time[batch_time.size()*.75], 1e-6*batch_time[batch_time.size()*.99], 1e-6*batch_time[batch_time.size()*.999], 1e-6*batch_time[batch_time.size()*.9999], 1e-6*batch_time[batch_time.size()-1]);

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
                printf("%lu ", layer_counts[i].load());
            }
            else
            {
                printf("\n");
                break;
            }
        }
    }

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
                printf("%lu ", layer_counts[i].load());
            }
            else
            {
                printf("\n");
                break;
            }
        }
    }
    return 0;
}
