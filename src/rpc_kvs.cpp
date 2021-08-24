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
#include <absl/container/btree_map.h>

int main(int argc, char** argv)
{
    if(argc <= 1)
    {
        fprintf(stderr, "usage: %s clients", argv[0]);
        exit(1);
    }
    const uint64_t clients = std::stoull(argv[1]);
    const uint64_t threads = omp_get_max_threads();
    std::vector<absl::btree_map<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>, bool>> graph(threads);
    std::vector<absl::btree_map<std::tuple<uint64_t, uint64_t>, std::tuple<uint64_t, uint64_t, uint64_t>>> history_vid(threads), history_version(threads);
    const uint64_t clients_per_thread = (clients+threads-1)/threads;
    auto streams = UCXStream::make_ucx_stream("", 2334, threads, clients_per_thread);
    tbb::enumerable_thread_specific<uint64_t> thread_min_version(0);
    std::atomic_uint64_t global_min_version(0);

    std::atomic_uint64_t version_id = 0, finished_version_id = 0;
    #pragma omp parallel
    {
        uint64_t tid = omp_get_thread_num();
        uint64_t client_begin = clients_per_thread*tid;
        uint64_t num_clients = std::min(clients_per_thread*(tid+1), clients)-client_begin;
        if(client_begin >= clients) num_clients = 0;
        std::vector<boost::fibers::fiber> fibers;
        std::vector<uint64_t> fiber_min_version(num_clients, 0);
        uint64_t left_clients = num_clients;
        for(uint64_t fid=client_begin;fid<client_begin+num_clients;fid++)
        {
            fibers.emplace_back([&, fid, lfid=fid-client_begin]()
            {
                KVUpdateRequest request;
                //KVUpdateResponse response;
                bool is_end = false;
                while(!is_end)
                {
                    streams[tid].recv(&request, sizeof(request), lfid);
                    assert(request.client_id == fid);
                    switch(request.type)
                    {
                        case KVUpdateRequest::Type::Add:
                        {
                            graph[tid][std::make_tuple(request.src, request.dst, request.data, request.version_id)] = true;
                            fiber_min_version[lfid] = request.version_id;
                            break;
                        }
                        case KVUpdateRequest::Type::Del:
                        {
                            graph[tid][std::make_tuple(request.src, request.dst, request.data, request.version_id)] = false;
                            fiber_min_version[lfid] = request.version_id;
                            break;
                        }
                        case KVUpdateRequest::Type::His:
                        {
                            history_vid[tid][std::make_tuple(request.src, request.version_id)] = {request.label_data, request.dst, request.data};
                            history_version[tid][std::make_tuple(request.version_id, request.src)] = {request.label_data, request.dst, request.data};
                            break;
                        }
                        case KVUpdateRequest::Type::Ping:
                        {
                            fiber_min_version[lfid] = request.version_id;
                            break;
                        }
                        case KVUpdateRequest::Type::Version:
                        {
                            KVUpdateResponse response = {(uint32_t)fid, global_min_version.load()};
                            streams[tid].send(&response, sizeof(response), lfid);
                            break;
                        }
                        case KVUpdateRequest::Type::End:
                        {
                            is_end = true;
                            left_clients--;
                            break;
                        }
                    }
                    //response.client_id = request.client_id;
                    //streams[tid].send(&response, sizeof(response), lfid);
                    //boost::this_fiber::yield();
                }
                fiber_min_version[lfid] = (uint64_t)-1;
            });
        }
        while(left_clients)
        {
            auto min_version = (uint64_t)-1;
            for(auto v : fiber_min_version) min_version = std::min(min_version, v);
            thread_min_version.local() = min_version;
            for(auto v : thread_min_version) min_version = std::min(min_version, v);
            global_min_version = min_version;
            boost::this_fiber::yield();
        }
        for(auto &f : fibers) f.join();
        thread_min_version.local() = (uint64_t)-1;

    }
    return 0;
}
