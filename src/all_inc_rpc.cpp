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
#include <libaio.h>
#include "core/type.hpp"
#include "core/graph.hpp"
#include "core/io.hpp"
#include "core/ucx_stream.hpp"
#include "core/rpc.hpp"

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
#if !defined(ENABLE_KVS) && !defined(ENABLE_WAL)
    if(argc <= 5)
    {
        fprintf(stderr, "usage: %s graph root timeout_ms percent clients\n", argv[0]);
        exit(1);
    }
#elif defined(ENABLE_KVS) && !defined(ENABLE_WAL)
    if(argc <= 6)
    {
        fprintf(stderr, "usage: %s graph root timeout_ms percent clients kvs_server\n", argv[0]);
        exit(1);
    }
#elif !defined(ENABLE_KVS) && defined(ENABLE_WAL)
    if(argc <= 6)
    {
        fprintf(stderr, "usage: %s graph root timeout_ms percent clients WAL_path\n", argv[0]);
        exit(1);
    }
#else
    if(argc <= 7)
    {
        fprintf(stderr, "usage: %s graph root timeout_ms percent clients WAL_path kvs_server\n", argv[0]);
        exit(1);
    }
#endif
    std::pair<uint64_t, uint64_t> *raw_edges;
    uint64_t raw_edges_len;
    std::tie(raw_edges, raw_edges_len) = mmap_binary(argv[1]);
    uint64_t root = std::stoull(argv[2]);
    const uint32_t timeout_ms = std::stoi(argv[3]);
    const double target_timeout_rate = 1-std::stod(argv[4]);
    const uint64_t clients = std::stoull(argv[5]);
#ifdef ENABLE_WAL
    const std::string wal_path = argv[6];
#endif

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
#if defined(BFS)
    Graph<void> graph(num_vertices, raw_edges_len, false, true, true);
#elif defined(WCC)
    Graph<void> graph(num_vertices, raw_edges_len, true, false, true);
#elif defined(SSSP) || defined(SSWP)
    Graph<uint64_t> graph(num_vertices, raw_edges_len, false, true, true);
#endif
    //std::random_shuffle(raw_edges.begin(), raw_edges.end());
    uint64_t imported_edges = raw_edges_len*0.9;
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
    auto init_label_func = [](uint64_t vid) -> std::pair<uint64_t, bool>
    {
        return {vid, true};
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

    const uint64_t threads = omp_get_max_threads();
    const uint64_t clients_per_thread = (clients+threads-1)/threads;
#ifdef ENABLE_KVS
#ifdef ENABLE_WAL
    auto kvs_streams = UCXStream::make_ucx_stream(argv[7], 2334, threads, clients_per_thread);
#else
    auto kvs_streams = UCXStream::make_ucx_stream(argv[6], 2334, threads, clients_per_thread);
#endif
#endif

    std::mutex history_mutex;
    auto history_labels = graph.alloc_history_array<uint64_t>();
    std::vector<uint64_t> history_vids;
    std::map<uint64_t, uint64_t> history_vid_idxes;
    std::atomic_uint64_t version_id = 0, history_vids_len = 0;
    history_vid_idxes[version_id] = history_vids_len;

    graph.stream_vertices<uint64_t>(
        [&](uint64_t vid)
        {
#ifdef ENABLE_KVS
            auto tid = graph.get_thread_id();
            auto fid = tid*clients_per_thread;
            KVUpdateRequest kvs_request = {(uint32_t)fid, KVUpdateRequest::Type::His, vid, labels[vid].parent.nbr, labels[vid].parent.data, version_id, labels[vid].data};
            kvs_streams[tid].send(&kvs_request, sizeof(kvs_request), 0);
#endif
            history_labels.set(vid, version_id, labels[vid].data);
            atomic_append(history_vids, history_vids_len, vid, history_mutex);
            return 0;
        }, graph.get_dense_active_all());

    auto streams = UCXStream::make_ucx_stream("", 2333, threads, clients_per_thread);

    const uint64_t time_offset_sync_times = 1024;
    std::chrono::high_resolution_clock::duration time_offset = std::chrono::nanoseconds(0);
    for(uint64_t i=0;i<time_offset_sync_times;i++)
    {
        ClientUpdateRequest request;
        ClientUpdateResponse response = {0, 0};
        streams[0].recv(&request, sizeof(request), 0);
        time_offset += std::chrono::high_resolution_clock::now()-request.request_time;
        streams[0].send(&response, sizeof(response), 0);
    }
    time_offset /= time_offset_sync_times;
    fprintf(stderr, "Time Offset: %ld ns\n", time_offset.count());

    struct Update
    {
        enum class Type
        {
            Add,
            Del,
            End
        } type;
        decltype(graph)::edge_type edge;
        std::chrono::high_resolution_clock::time_point time;
        uint64_t *version_id;
    };

#ifdef ENABLE_WAL
    struct WAL
    {
        enum class Type : uint32_t
        {
            Add,
            Del
        } type;
        uint32_t fid;
        uint64_t src;
        uint64_t dst;
        uint64_t data;
    };
    constexpr uint64_t WALBlockSize = 4096;
    struct WALBlock
    {
        uint64_t version;
        uint64_t num_logs;
        WAL logs[];
        void clear(uint64_t _version)
        {
            version = _version;
            num_logs = 0;
        }
        bool append(const WAL &log)
        {
            if((num_logs+1)*sizeof(WAL)+sizeof(WALBlock) > WALBlockSize) return false;
            logs[num_logs++] = log;
            return true;
        }
    };
#endif

    std::atomic_uint64_t add_edge_len(0), del_edge_len(0), batch_size(0);
    std::vector<Update> add_edge, del_edge;
    std::mutex add_mutex, del_mutex;
    std::atomic_bool add_setted = false , del_setted = false;
    std::chrono::high_resolution_clock::time_point add_first, del_first;

    uint64_t num_changes_threshold = omp_get_max_threads()*4;
    auto need_break_func = [&](std::chrono::high_resolution_clock::time_point time)
    {
        uint64_t num_changes = add_edge_len.load(std::memory_order_relaxed) + del_edge_len.load(std::memory_order_relaxed);
        auto min_time = time;
        if(add_setted.load(std::memory_order_acquire)) min_time = std::min(min_time, add_first);
        if(del_setted.load(std::memory_order_acquire)) min_time = std::min(min_time, del_first);
        return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(time-min_time).count() > 0.8*timeout_ms || num_changes > num_changes_threshold;
        //return num_changes > num_changes_threshold;
    };

    std::atomic_uint64_t num_timeouts = 0, num_updates = 0, end_threads = 0;
    std::atomic_bool loop_end = false;
    uint64_t last_num_timeouts = 0, last_num_updates = 0, num_loops = 0;

#ifdef ENABLE_KVS
    std::atomic_uint64_t version_sync_nanoseconds(0), version_sync_times(0);
#endif

    {
        auto start = std::chrono::system_clock::now();
        #pragma omp parallel
        {
            uint64_t tid = omp_get_thread_num();
            uint64_t client_begin = clients_per_thread*tid;
            uint64_t num_clients = std::min(clients_per_thread*(tid+1), clients)-client_begin;
            if(client_begin >= clients) num_clients = 0;
            uint64_t local_batch_size = 0;
            std::vector<boost::fibers::fiber> fibers;
            boost::fibers::condition_variable client_cond;
            uint64_t left_clients = num_clients, end_clients = 0;
            boost::fibers::mutex mutex;

#ifdef ENABLE_WAL
            int file_fd;
            std::string file_name = wal_path+"."+std::to_string(tid);
            if((file_fd = open(file_name.c_str(), O_RDWR | O_CREAT | O_DIRECT, 0644)) < 0) throw std::runtime_error("Open File Error");
            if(ftruncate(file_fd, 0) != 0) throw std::runtime_error("Truncate File Error");
            io_context_t io_ctx; memset(&io_ctx, 0, sizeof(io_ctx));
            if(io_setup(16, &io_ctx) != 0) throw std::runtime_error("AIO Setup Error");
            WALBlock *syncing_block = (WALBlock*) aligned_alloc(WALBlockSize, WALBlockSize);
            WALBlock *writing_block = (WALBlock*) aligned_alloc(WALBlockSize, WALBlockSize);
            uint64_t wal_block_version = 0;
            writing_block->clear(wal_block_version+1);
            struct iocb iocb;
            struct iocb* iocbp = &iocb;
            struct io_event event;
            struct timespec timespec = {0, 0};
#endif

            auto safe_begin = std::chrono::high_resolution_clock::now();
            if(num_clients == 0) end_threads++;
            std::chrono::high_resolution_clock::time_point last_print_time = std::chrono::high_resolution_clock::now();
            std::vector<std::unique_ptr<boost::fibers::buffered_channel<Update>>> requests;
            for(uint64_t fid=client_begin;fid<client_begin+num_clients;fid++)
            {
                requests.emplace_back(std::make_unique<boost::fibers::buffered_channel<Update>>(1024));
                fibers.emplace_back([&, fid, lfid=fid-client_begin]()
                {
                    ClientUpdateRequest request;
                    bool is_end = false;
                    while(!is_end)
                    {
                        streams[tid].recv(&request, sizeof(request), lfid);
                        request.request_time += time_offset;
                        switch(request.type)
                        {
                            case ClientUpdateRequest::Type::Add:
                            {
#if defined(BFS) || defined(WCC)
                                decltype(graph)::edge_type edge = {request.src, request.dst};
#elif defined(SSSP) || defined(SSWP)
                                decltype(graph)::edge_type edge = {request.src, request.dst, request.data};
#endif
                                requests[lfid]->push(Update{Update{Update::Type::Add, edge, request.request_time, nullptr}});
                                break;
                            }
                            case ClientUpdateRequest::Type::Del:
                            {
#if defined(BFS) || defined(WCC)
                                decltype(graph)::edge_type edge = {request.src, request.dst};
#elif defined(SSSP) || defined(SSWP)
                                decltype(graph)::edge_type edge = {request.src, request.dst, request.data};
#endif
                                requests[lfid]->push(Update{Update{Update::Type::Del, edge, request.request_time, nullptr}});
                                break;
                            }
                            case ClientUpdateRequest::Type::End:
                            {
                                is_end = true;
                                requests[lfid]->push(Update{Update{Update::Type::End, {}, request.request_time, nullptr}});
                                break;
                            }
                        }
                    }
                });
                fibers.emplace_back([&, fid, lfid=fid-client_begin]()
                {
                    Update update;
                    ClientUpdateResponse response;
#ifdef ENABLE_KVS
                    KVUpdateRequest kvs_request;
                    //KVUpdateResponse kvs_response;
#endif
                    bool is_end = false;
                    while(!is_end)
                    {
                        update = requests[lfid]->value_pop();
#ifdef ENABLE_WAL
                        uint64_t cur_wal_block_version = 0;
                        if(update.type != Update::Type::End)
                        {
                            WAL log = {update.type == Update::Type::Add ? WAL::Type::Add : WAL::Type::Del, 
                                       (uint32_t)fid, update.edge.src, update.edge.dst, update.edge.data};
                            while(!writing_block->append(log)) boost::this_fiber::yield();
                            cur_wal_block_version = writing_block->version;
                        }
#endif
                        switch(update.type)
                        {
                            case Update::Type::Add:
                            {
                                local_batch_size += 1;
                                if(graph.need_update_tree_add(update_func, labels, update.edge, directed))
                                {
                                    std::unique_lock<boost::fibers::mutex> lock(mutex);
                                    auto idx = atomic_append(add_edge, add_edge_len, Update{Update::Type::Add, update.edge, update.time, &response.version_id}, add_mutex);
                                    if(idx == 0)
                                    {
                                        add_first = update.time;
                                        add_setted.store(true, std::memory_order_release);
                                    }
                                    left_clients--;
                                    client_cond.wait(lock);
                                    left_clients++;
                                }
                                else
                                {
                                    response.version_id = ++version_id;
                                    graph.add_edge(update.edge, directed);
#ifdef ENABLE_KVS
                                    kvs_request = {(uint32_t)fid, KVUpdateRequest::Type::Add, update.edge.src, update.edge.dst, update.edge.data, response.version_id, 0};
                                    kvs_streams[tid].send(&kvs_request, sizeof(kvs_request), lfid);
#endif
                                }
                                break;
                            }
                            case Update::Type::Del:
                            {
                             
                                local_batch_size += 1;
                                if(graph.need_update_tree_del(labels, update.edge, directed))
                                {
                                    std::unique_lock<boost::fibers::mutex> lock(mutex);
                                    auto idx = atomic_append(del_edge, del_edge_len, Update{Update::Type::Add, update.edge, update.time, &response.version_id}, del_mutex);
                                    if(idx == 0)
                                    {
                                        del_first = update.time;
                                        del_setted.store(true, std::memory_order_release);
                                    }
                                    left_clients--;
                                    client_cond.wait(lock);
                                    left_clients++;
                                }
                                else
                                {
                                    response.version_id = ++version_id;
                                    graph.del_edge(update.edge, directed);
#ifdef ENABLE_KVS
                                    kvs_request = {(uint32_t)fid, KVUpdateRequest::Type::Del, update.edge.src, update.edge.dst, update.edge.data, response.version_id, 0};
                                    kvs_streams[tid].send(&kvs_request, sizeof(kvs_request), lfid);
#endif
                                }
                             
                                break;
                            }
                            case Update::Type::End:
                            {
#ifdef ENABLE_KVS
                                kvs_request = {(uint32_t)fid, KVUpdateRequest::Type::Ping, 0, 0, 0, (uint64_t)-1, 0};
                                kvs_streams[tid].send(&kvs_request, sizeof(kvs_request), lfid);
#endif
                                is_end = true;
                                if(++end_clients == num_clients) end_threads++;
                                break;
                            }
                        }
#ifdef ENABLE_WAL
                        while(cur_wal_block_version > wal_block_version) boost::this_fiber::yield();
#endif
                        response.client_id = fid;
                        streams[tid].send(&response, sizeof(response), lfid);
                        auto time = std::chrono::high_resolution_clock::now();
                        if(time > update.time+std::chrono::milliseconds(timeout_ms-1))
                        {
                            num_timeouts++;
                        }
                    }
                });
            }

#ifdef ENABLE_KVS
            if(num_clients) fibers.emplace_back([&]()
            {
                auto last_time = std::chrono::high_resolution_clock::now();
                auto last_version = 0lu;
                while(!loop_end)
                {
                    boost::this_fiber::sleep_for(std::chrono::microseconds(100));
                    
                    KVUpdateRequest request = {(uint32_t)client_begin, KVUpdateRequest::Type::Version, 0, 0, 0, 0, 0};
                    KVUpdateResponse response;
                    kvs_streams[tid].send(&request, sizeof(request), 0);
                    kvs_streams[tid].recv(&response, sizeof(response), 0);
                    if(response.version_id > last_version)
                    {
                        auto time = std::chrono::high_resolution_clock::now();
                        version_sync_nanoseconds += (time-last_time).count();
                        version_sync_times++;
                        last_version = version_id;
                        last_time = time;
                    }

                    boost::this_fiber::yield();

                }
            });
#endif
#ifdef ENABLE_WAL
            if(num_clients) fibers.emplace_back([&]()
            {
                while(!loop_end)
                {
                    boost::this_fiber::yield();
                    if(writing_block->num_logs == 0) continue;
                    std::swap(writing_block, syncing_block);
                    writing_block->clear(syncing_block->version+1);
                    io_prep_pwrite(&iocb, file_fd, syncing_block, WALBlockSize, (syncing_block->version-1)*WALBlockSize);
                    if(io_submit(io_ctx, 1, &iocbp) != 1) throw std::runtime_error("AIO Submit Error");
                    do
                    {
                        boost::this_fiber::yield();
                    }while(io_getevents(io_ctx, 1, 1, &event, &timespec) != 1);
                    wal_block_version = syncing_block->version;
                }
            });
#endif

            while(!loop_end)
            {
                auto time = std::chrono::high_resolution_clock::now();
                bool need_break = need_break_func(time) || !left_clients || end_clients == num_clients; 
                if(need_break)
                {
                    std::unique_lock<boost::fibers::mutex> lock(mutex);
                    batch_size += local_batch_size;
                    local_batch_size = 0;
                    #pragma omp barrier
                    if(tid == 0)
                    {
                        if(end_threads == threads) loop_end = true;
                        num_loops ++;
                        num_updates.fetch_add(batch_size);
                        auto safe_end = std::chrono::high_resolution_clock::now();
                        auto unsafe_start = std::chrono::high_resolution_clock::now();

                        uint64_t i_add = 0, i_del = 0;
                        uint64_t n_add = add_edge_len.load(), n_del = del_edge_len.load();
                        while(i_add < n_add || i_del < n_del)
                        {
                            graph.clear_modified();
                            if(i_del >= n_del || (i_add < n_add && add_edge[i_add].time < del_edge[i_del].time))
                            {
                                *add_edge[i_add].version_id = ++version_id;
                                const auto &e = add_edge[i_add].edge;
#ifdef ENABLE_KVS
                                KVUpdateRequest kvs_request = {(uint32_t)(tid*clients_per_thread), KVUpdateRequest::Type::Add, e.src, e.dst, e.data, version_id, 0};
                                kvs_streams[tid].send(&kvs_request, sizeof(kvs_request), 0);
#endif
                                if(graph.add_edge(e, directed) == 0)
                                {
                                    graph.update_tree_add<uint64_t, uint64_t>(
                                        continue_reduce_func,
                                        update_func,
                                        active_result_func,
                                        labels, e, directed
                                    );
                                }
                                i_add++;
                            }
                            else
                            {
                                *del_edge[i_del].version_id = ++version_id;
                                const auto &e = del_edge[i_del].edge;
#ifdef ENABLE_KVS
                                KVUpdateRequest kvs_request = {(uint32_t)(tid*clients_per_thread), KVUpdateRequest::Type::Del, e.src, e.dst, e.data, version_id, 0};
                                kvs_streams[tid].send(&kvs_request, sizeof(kvs_request), 0);
#endif
                                if(graph.del_edge(e, directed) == 1)
                                {
                                    graph.update_tree_del<uint64_t, uint64_t>(
                                        init_label_func,
                                        continue_reduce_func,
                                        update_func,
                                        active_result_func,
                                        equal_func,
                                        labels, e, directed
                                    );
                                }
                                i_del++;
                            }

                            history_vid_idxes[version_id] = history_vids_len;

                            graph.stream_vertices_hybrid<uint64_t>(
                                [&](uint64_t vid)
                                {
#ifdef ENABLE_KVS
                                    auto tid = graph.get_thread_id();
                                    auto fid = tid*clients_per_thread;
                                    KVUpdateRequest kvs_request = {(uint32_t)fid, KVUpdateRequest::Type::His, vid, labels[vid].parent.nbr, labels[vid].parent.data, version_id, labels[vid].data};
                                    kvs_streams[tid].send(&kvs_request, sizeof(kvs_request), 0);
#endif
                                    history_labels.set(vid, version_id, labels[vid].data);
                                    atomic_append(history_vids, history_vids_len, vid, history_mutex);
                                    return 0;
                                }, graph.get_modified());
                        }
                        auto unsafe_end = std::chrono::high_resolution_clock::now();
                        if(num_loops % 100 == 0)
                        {
                            fprintf(stderr, "interval: %.6lf s, ", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(unsafe_end-last_print_time).count());
                            fprintf(stderr, "exec safe: %.6lf s, ", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(safe_end-safe_begin).count());
                            fprintf(stderr, "unsafe_add: %6lu, ", add_edge_len.load());
                            fprintf(stderr, "unsafe_del: %6lu, ", del_edge_len.load());
                            fprintf(stderr, "exec unsafe: %.6lf s, ", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(unsafe_end-unsafe_start).count());
                            fprintf(stderr, "batch_size: %lu, ", batch_size.load(std::memory_order_relaxed));
                            fprintf(stderr, "timeouts: %lu/%lu = 1-%lf, ", num_timeouts.load(std::memory_order_relaxed), num_updates.load(std::memory_order_relaxed), 1-(double)num_timeouts.load(std::memory_order_relaxed)/num_updates.load(std::memory_order_relaxed));
                            fprintf(stderr, "threshold: %lu\n", num_changes_threshold);
#ifdef ENABLE_KVS
                            fprintf(stderr, "sync_latency: %.6lf ms\n", 1e-6*version_sync_nanoseconds/version_sync_times);
#endif
                            last_print_time = unsafe_end;
                        }

                        add_edge_len = 0; del_edge_len = 0;
                        add_setted = false; del_setted = false;
                        batch_size = 0;

                        if(num_loops % 3 == 0)
                        {
                            uint64_t global_num_timeouts = num_timeouts.load(std::memory_order_relaxed);
                            uint64_t global_num_updates = num_updates.load(std::memory_order_relaxed);
                            uint64_t local_num_timeouts = global_num_timeouts-last_num_timeouts;
                            uint64_t local_num_updates = global_num_updates-last_num_updates;
                            if((double)local_num_timeouts/local_num_updates > 0.8*target_timeout_rate)
                            {
                                num_changes_threshold *= 0.9;
                                num_changes_threshold = std::max(num_changes_threshold, threads*4);
                            }
                            if((double)local_num_timeouts/local_num_updates < 0.8*target_timeout_rate && (double)global_num_timeouts/global_num_updates < target_timeout_rate)
                            {
                                if(num_changes_threshold < 1000) num_changes_threshold += 10; else num_changes_threshold *= 1.01;
                                num_changes_threshold = std::min(num_changes_threshold, 1000000lu);
                            }
                            last_num_timeouts += local_num_timeouts;
                            last_num_updates += local_num_updates;
                        }
                    }
                    #pragma omp barrier
                    safe_begin = std::chrono::high_resolution_clock::now();
                    client_cond.notify_all();
                }
                boost::this_fiber::yield();
            }
            for(auto &f : fibers) f.join();

#ifdef ENABLE_WAL
            free(writing_block);
            free(syncing_block);
            io_destroy(io_ctx);
            close(file_fd);
#endif
        }
#ifdef ENABLE_KVS
        #pragma omp parallel
        {
            uint64_t tid = omp_get_thread_num();
            uint64_t client_begin = clients_per_thread*tid;
            uint64_t num_clients = std::min(clients_per_thread*(tid+1), clients)-client_begin;
            if(client_begin >= clients) num_clients = 0;
            for(uint64_t fid=client_begin;fid<client_begin+num_clients;fid++)
            {
                KVUpdateRequest kvs_request;
                auto lfid = fid-client_begin;
                kvs_request = {(uint32_t)fid, KVUpdateRequest::Type::End, 0, 0, 0, 0, 0};
                kvs_streams[tid].send(&kvs_request, sizeof(kvs_request), lfid);
            }
        }
#endif
        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "exec: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }

#if defined(BFS)
    auto print_func = [&]()
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
    };
#elif defined(WCC)
    auto print_func = [&]()
    {
        auto num_wcces = graph.stream_vertices<uint64_t>(
            [&](uint64_t vid)
            {
                return labels[vid].data == vid;
            },
            graph.get_dense_active_all()
        );
        fprintf(stderr, "number of communities: %lu\n", num_wcces);
    };
#elif defined(SSSP) || defined(SSWP)
    auto print_func = [&]()
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
    };
#endif

    print_func();

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

    print_func();

    return 0;
}
