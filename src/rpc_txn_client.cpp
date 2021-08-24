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
#include "core/ucx_stream.hpp"
#include "core/rpc.hpp"

int main(int argc, char** argv)
{
    if(argc <= 7)
    {
        fprintf(stderr, "usage: %s graph timeout_ms clients queue_depth sleep_time_us updates_per_txn server", argv[0]);
        exit(1);
    }

    std::pair<uint64_t, uint64_t> *raw_edges, *add_raw_edges, *del_raw_edges;
    uint64_t raw_edges_len, add_raw_edges_len, del_raw_edges_len;
    std::tie(raw_edges, raw_edges_len) = mmap_binary(argv[1]);
    const uint32_t timeout_ms = std::stoi(argv[2]);
    const uint64_t clients = std::stoull(argv[3]);
    const uint32_t queue_depth = std::stoi(argv[4]);
    const uint32_t sleep_time_us = std::stoi(argv[5]);
    const uint32_t updates_per_txn = std::stoi(argv[6]);

    uint64_t imported_edges = raw_edges_len*0.9;
    if(argc > 9)
    {
        std::tie(add_raw_edges, add_raw_edges_len) = mmap_binary(argv[8]);
        std::tie(del_raw_edges, del_raw_edges_len) = mmap_binary(argv[9]);
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
        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "read: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
        fprintf(stderr, "|E|=%lu\n", raw_edges_len);
    }

    const uint64_t threads = omp_get_max_threads();
    const uint64_t clients_per_thread = (clients+threads-1)/threads;
    auto streams = UCXStream::make_ucx_stream(argv[7], 2333, threads, clients_per_thread);

    const uint64_t time_offset_sync_times = 1024;
    for(uint64_t i=0;i<time_offset_sync_times;i++)
    {
        ClientUpdateRequest request = {0, ClientUpdateRequest::Type::End, 0, 0, 0, std::chrono::high_resolution_clock::now()};
        ClientUpdateResponse response;
        streams[0].send(&request, sizeof(request), 0);
        streams[0].recv(&response, sizeof(response), 0);
    }

    tbb::enumerable_thread_specific<std::vector<uint64_t>> all_latency;
    tbb::enumerable_thread_specific<std::vector<uint64_t>> all_event_latency;
    std::atomic_uint64_t num_timeouts = 0, num_updates = 0, sum_nanoseconds = 0, sum_event_nanoseconds = 0;

    {
        auto start = std::chrono::system_clock::now();
        #pragma omp parallel
        {
            uint64_t tid = omp_get_thread_num();
            uint64_t client_begin = clients_per_thread*tid;
            uint64_t num_clients = std::min(clients_per_thread*(tid+1), clients)-client_begin;
            if(client_begin >= clients) num_clients = 0;
            uint64_t local_num_timeouts = 0, local_num_updates = 0, local_sum_nanoseconds = 0, local_sum_event_nanoseconds = 0;
            uint64_t last_print_updates = 0;
            uint64_t running_clients = num_clients;
            std::vector<boost::fibers::fiber> fibers;
            std::vector<uint64_t> wait_responses(num_clients);
            std::vector<std::unique_ptr<boost::fibers::buffered_channel<std::pair<ClientTxnUpdateRequest, std::chrono::high_resolution_clock::time_point>>>> requests;
            for(uint64_t fid=client_begin;fid<client_begin+num_clients;fid++)
            {
                requests.emplace_back(std::make_unique<boost::fibers::buffered_channel<std::pair<ClientTxnUpdateRequest, std::chrono::high_resolution_clock::time_point>>>(2)); //minimal size is 2
                fibers.emplace_back([&, fid, lfid=fid-client_begin]()
                {
                    std::pair<ClientTxnUpdateRequest, std::chrono::high_resolution_clock::time_point> channel_pair;
                    ClientTxnUpdateRequest &request = channel_pair.first;
                    auto &event_time = channel_pair.second;
                    uint64_t add = fid, del = fid;
                    event_time = std::chrono::high_resolution_clock::now();
                    while(add < add_raw_edges_len || del < del_raw_edges_len)
                    {
                        request.client_id = fid;
                        request.num_updates = 0;
                        for(bool is_add = true; add < add_raw_edges_len || del < del_raw_edges_len; is_add = !is_add)
                        {
                            if(is_add)
                            {
                                if(add >= add_raw_edges_len) continue;
                                const auto &e = add_raw_edges[add];
                                request.types[request.num_updates] = ClientTxnUpdateRequest::Type::Add;
                                request.srcs[request.num_updates] = e.first;
                                request.dsts[request.num_updates] = e.second;
                                request.datas[request.num_updates] = (e.first+e.second)%128;
                                add+=clients;
                            }
                            else
                            {
                                if(del >= del_raw_edges_len) continue;
                                const auto &e = del_raw_edges[del];
                                request.types[request.num_updates] = ClientTxnUpdateRequest::Type::Del;
                                request.srcs[request.num_updates] = e.first;
                                request.dsts[request.num_updates] = e.second;
                                request.datas[request.num_updates] = (e.first+e.second)%128;
                                del+=clients;
                            }
                            request.num_updates++;
                            if(request.num_updates >= updates_per_txn) break;
                        }
                        request.request_time = std::chrono::high_resolution_clock::now();
                        if(queue_depth != (uint32_t)-1 && request.request_time-event_time > queue_depth*std::chrono::microseconds(sleep_time_us))
                        {
                            fprintf(stderr, "Queue is full!\n");
                            throw std::runtime_error("Over Sustainable Throughput!");
                        }
                        requests[lfid]->push(channel_pair);
                        streams[tid].send(&request, sizeof(request), lfid);
                        wait_responses[lfid]++;

                        while(wait_responses[lfid]) boost::this_fiber::yield();
                        event_time += std::chrono::microseconds(sleep_time_us);
                        boost::this_fiber::sleep_until(event_time);
                    }
                    request = {(uint32_t)fid, 1, {ClientTxnUpdateRequest::Type::End}, {0}, {0}, {0}, std::chrono::high_resolution_clock::now()};
                    requests[lfid]->push(channel_pair);
                    streams[tid].send(&request, sizeof(request), lfid);
                    wait_responses[lfid]++;
                });
                fibers.emplace_back([&, fid, lfid=fid-client_begin]()
                {
                    ClientTxnUpdateResponse response;
                    while(true)
                    {
                        if(wait_responses[lfid])
                        {
                            streams[tid].recv(&response, sizeof(response), lfid);
                            assert(fid == (uint64_t)response.client_id);

                            auto channel_pair = requests[lfid]->value_pop();
                            auto &request = channel_pair.first;
                            auto &event_time = channel_pair.second;
                            if(request.types[0] == ClientTxnUpdateRequest::Type::End)
                            {
                                running_clients --;
                                break;
                            }

                            auto now = std::chrono::high_resolution_clock::now();
                            if(now-request.request_time > std::chrono::milliseconds(timeout_ms)) local_num_timeouts ++;
                            uint64_t latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now-request.request_time).count();
                            all_latency.local().emplace_back(latency_ns);
                            uint64_t event_latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now-event_time).count();
                            all_event_latency.local().emplace_back(event_latency_ns);
                            local_num_updates ++;
                            local_sum_nanoseconds += latency_ns;
                            local_sum_event_nanoseconds += event_latency_ns;
                            wait_responses[lfid]--;
                        }
                        boost::this_fiber::yield();
                    }
                });
            }
            while(num_clients && running_clients)
            {
                num_timeouts += local_num_timeouts;
                num_updates += local_num_updates;
                sum_nanoseconds += local_sum_nanoseconds;
                sum_event_nanoseconds += local_sum_event_nanoseconds;
                local_num_timeouts = 0;
                local_num_updates = 0;
                local_sum_nanoseconds = 0;
                local_sum_event_nanoseconds = 0;
                if(tid == 0 && num_updates - last_print_updates > 1000000)
                {
                    fprintf(stderr, "timeouts: %lu/%lu = 1-%lf\n", num_timeouts.load(std::memory_order_relaxed), num_updates.load(std::memory_order_relaxed), 1-(double)num_timeouts.load(std::memory_order_relaxed)/num_updates.load(std::memory_order_relaxed));
                    last_print_updates = num_updates;
                }
                boost::this_fiber::yield();
            }
            for(auto &f : fibers) f.join();
        }
        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "exec:       %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
        fprintf(stderr, "Throughput: %lf\n", num_updates.load(std::memory_order_relaxed)/(1e-9*(end-start).count()));
        fprintf(stderr, "Success:    %lf\n", 1-(double)num_timeouts.load(std::memory_order_relaxed)/num_updates.load(std::memory_order_relaxed));
        fprintf(stderr, "Mean:       %lf us\n", (1e-3)*sum_nanoseconds/num_updates.load(std::memory_order_relaxed));
        std::vector<uint64_t> latency;
        for(auto &p : all_latency)
        {
            for(auto t : p) latency.emplace_back(t);
            p.clear();
        }
        tbb::parallel_sort(latency.begin(), latency.end());
        fprintf(stderr, "P25:        %lf us\n", 1e-3*latency[latency.size()*0.25]);
        fprintf(stderr, "P50:        %lf us\n", 1e-3*latency[latency.size()*0.50]);
        fprintf(stderr, "P75:        %lf us\n", 1e-3*latency[latency.size()*0.75]);
        fprintf(stderr, "P90:        %lf us\n", 1e-3*latency[latency.size()*0.90]);
        fprintf(stderr, "P99:        %lf us\n", 1e-3*latency[latency.size()*0.99]);
        fprintf(stderr, "P999:       %lf us\n", 1e-3*latency[latency.size()*0.999]);
        fprintf(stderr, "P9999:      %lf us\n", 1e-3*latency[latency.size()*0.9999]);
        fprintf(stderr, "P99999:     %lf us\n", 1e-3*latency[latency.size()*0.99999]);
        fprintf(stderr, "P999999:    %lf us\n", 1e-3*latency[latency.size()*0.999999]);
        fprintf(stderr, "P9999999:   %lf us\n", 1e-3*latency[latency.size()*0.9999999]);
        fprintf(stderr, "P99999999:  %lf us\n", 1e-3*latency[latency.size()*0.99999999]);
        fprintf(stderr, "Max:        %lf us\n", 1e-3*latency[latency.size()-1]);

        fprintf(stderr, "Event Mean:       %lf us\n", (1e-3)*sum_event_nanoseconds/num_updates.load(std::memory_order_relaxed));
        latency.clear();
        for(auto &p : all_event_latency)
        {
            for(auto t : p) latency.emplace_back(t);
            p.clear();
        }
        tbb::parallel_sort(latency.begin(), latency.end());
        fprintf(stderr, "Event P25:        %lf us\n", 1e-3*latency[latency.size()*0.25]);
        fprintf(stderr, "Event P50:        %lf us\n", 1e-3*latency[latency.size()*0.50]);
        fprintf(stderr, "Event P75:        %lf us\n", 1e-3*latency[latency.size()*0.75]);
        fprintf(stderr, "Event P90:        %lf us\n", 1e-3*latency[latency.size()*0.90]);
        fprintf(stderr, "Event P99:        %lf us\n", 1e-3*latency[latency.size()*0.99]);
        fprintf(stderr, "Event P999:       %lf us\n", 1e-3*latency[latency.size()*0.999]);
        fprintf(stderr, "Event P9999:      %lf us\n", 1e-3*latency[latency.size()*0.9999]);
        fprintf(stderr, "Event P99999:     %lf us\n", 1e-3*latency[latency.size()*0.99999]);
        fprintf(stderr, "Event P999999:    %lf us\n", 1e-3*latency[latency.size()*0.999999]);
        fprintf(stderr, "Event P9999999:   %lf us\n", 1e-3*latency[latency.size()*0.9999999]);
        fprintf(stderr, "Event P99999999:  %lf us\n", 1e-3*latency[latency.size()*0.99999999]);
        fprintf(stderr, "Event Max:        %lf us\n", 1e-3*latency[latency.size()-1]);
    }

    return 0;
}
