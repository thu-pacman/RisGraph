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

static std::pair<std::tuple<uint64_t, uint64_t, uint64_t>*, uint64_t> mmap_weighted_binary(std::string path)
{
    int fd = open(path.c_str(), O_RDONLY, 0640);
    if(fd == -1) throw std::runtime_error(std::string("open path ") + path + " error.");
    uint64_t size = lseek(fd, 0, SEEK_END);
    void *ret = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if(ret == MAP_FAILED) throw std::runtime_error("mmap error.");
    madvise(ret, size, MADV_DONTNEED);
    close(fd);
    return {(std::tuple<uint64_t, uint64_t, uint64_t>*)ret, size/sizeof(std::tuple<uint64_t, uint64_t, uint64_t>)};
}

int main(int argc, char** argv)
{
    if(argc <= 5)
    {
        fprintf(stderr, "usage: %s graph timeout_ms percent clients WAL_path\n", argv[0]);
        exit(1);
    }
    std::tuple<uint64_t, uint64_t, uint64_t> *raw_edges;
    uint64_t raw_edges_len;
    std::tie(raw_edges, raw_edges_len) = mmap_weighted_binary(argv[1]);
    const uint32_t timeout_ms = std::stoi(argv[2]);
    const double target_timeout_rate = 1-std::stod(argv[3]);
    const uint64_t clients = std::stoull(argv[4]);
    const std::string wal_path = argv[5];
    const uint64_t num_roots = 16;
    std::vector<uint64_t> roots;

    uint64_t num_vertices = 0;
    {
        auto start = std::chrono::system_clock::now();
        #pragma omp parallel for
        for(uint64_t i=0;i<raw_edges_len;i++)
        {
            auto [src, dst, weight] = raw_edges[i];
            write_max(&num_vertices, src+1);
            write_max(&num_vertices, dst+1);
        }
        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "read: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
        fprintf(stderr, "|E|=%lu\n", raw_edges_len);
    }

    std::mt19937_64 random;
    for(uint64_t i=0;i<num_roots;i++)
    {
        roots.emplace_back(random()%num_vertices);
    }

    Graph<uint64_t> graph(num_vertices*2, raw_edges_len, true, false, true);
    //std::random_shuffle(raw_edges.begin(), raw_edges.end());
    uint64_t imported_edges = raw_edges_len;
    {
        auto start = std::chrono::system_clock::now();
        #pragma omp parallel for
        for(uint64_t i=0;i<imported_edges;i++)
        {
            auto [src, dst, weight] = raw_edges[i];
            graph.add_edge({src, dst, weight}, false);
        }
        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "add: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }

    using label_type = std::pair<uint32_t, uint32_t>;
    auto labels = graph.alloc_vertex_tree_array<label_type>();

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
    auto update_func = [](uint64_t src, uint64_t dst, label_type src_data, label_type dst_data, decltype(graph)::adjedge_type adjedge) -> std::pair<bool, label_type>
    {
        return std::make_pair((src_data.first+1 < dst_data.first) || (src_data.first+1 == dst_data.first && src_data.second+adjedge.data > dst_data.second), std::make_pair(src_data.first+1, src_data.second+adjedge.data));
    };
    auto active_result_func = [](uint64_t old_result, uint64_t src, uint64_t dst, label_type src_data, label_type old_dst_data, label_type new_dst_data) -> uint64_t
    {
        return old_result+1;
    };
    auto equal_func = [](uint64_t src, uint64_t dst, label_type src_data, label_type dst_data, decltype(graph)::adjedge_type adjedge) -> bool
    {
        return src_data.first+1 == dst_data.first && src_data.second+adjedge.data == dst_data.second;
    };
    auto init_label_func = [&](uint64_t vid) -> std::pair<label_type, bool>
    {
        for(uint64_t i=0;i<num_roots;i++)
        {
            if(vid == roots[i]) return {std::make_pair(0, 0), true};
        }
        return {std::make_pair(MAXL, 0), false};
    };

    {
        auto start = std::chrono::system_clock::now();

        graph.build_tree<uint64_t, label_type>(
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

    std::mutex history_mutex;
    auto history_labels = graph.alloc_history_array<label_type>();
    std::vector<uint64_t> history_vids;
    std::map<uint64_t, uint64_t> history_vid_idxes;
    std::atomic_uint64_t version_id = 0, history_vids_len = 0;
    history_vid_idxes[version_id] = history_vids_len;

    graph.stream_vertices<uint64_t>(
        [&](uint64_t vid)
        {
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
    };

    std::atomic_uint64_t num_timeouts = 0, num_updates = 0, end_threads = 0;
    std::atomic_bool loop_end = false;
    uint64_t last_num_timeouts = 0, last_num_updates = 0, num_loops = 0;

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
                                decltype(graph)::edge_type edge = {request.src, request.dst, request.data};
                                requests[lfid]->push(Update{Update{Update::Type::Add, edge, request.request_time, nullptr}});
                                break;
                            }
                            case ClientUpdateRequest::Type::Del:
                            {
                                decltype(graph)::edge_type edge = {request.src, request.dst, request.data};
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
                    bool is_end = false;
                    while(!is_end)
                    {
                        update = requests[lfid]->value_pop();
                        uint64_t cur_wal_block_version = 0;
                        if(update.type != Update::Type::End)
                        {
                            WAL log = {update.type == Update::Type::Add ? WAL::Type::Add : WAL::Type::Del, 
                                       (uint32_t)fid, update.edge.src, update.edge.dst, update.edge.data};
                            while(!writing_block->append(log)) boost::this_fiber::yield();
                            cur_wal_block_version = writing_block->version;
                        }
                        switch(update.type)
                        {
                            case Update::Type::Add:
                            {
                                local_batch_size += 1;
                                if(graph.need_update_tree_add(update_func, labels, update.edge, false))
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
                                    graph.add_edge(update.edge, false);
                                }
                                break;
                            }
                            case Update::Type::Del:
                            {
                             
                                local_batch_size += 1;
                                if(graph.need_update_tree_del(labels, update.edge, false))
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
                                    graph.del_edge(update.edge, false);
                                }
                             
                                break;
                            }
                            case Update::Type::End:
                            {
                                is_end = true;
                                if(++end_clients == num_clients) end_threads++;
                                break;
                            }
                        }
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
                                if(graph.add_edge(e, false) == 0)
                                {
                                    graph.update_tree_add<uint64_t, label_type>(
                                        continue_reduce_func,
                                        update_func,
                                        active_result_func,
                                        labels, e, false
                                    );
                                }
                                i_add++;
                            }
                            else
                            {
                                *del_edge[i_del].version_id = ++version_id;
                                const auto &e = del_edge[i_del].edge;
                                if(graph.del_edge(e, false) == 1)
                                {
                                    graph.update_tree_del<uint64_t, label_type>(
                                        init_label_func,
                                        continue_reduce_func,
                                        update_func,
                                        active_result_func,
                                        equal_func,
                                        labels, e, false
                                    );
                                }
                                i_del++;
                            }

                            history_vid_idxes[version_id] = history_vids_len;

                            graph.stream_vertices_hybrid<uint64_t>(
                                [&](uint64_t vid)
                                {
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

            free(writing_block);
            free(syncing_block);
            io_destroy(io_ctx);
            close(file_fd);
        }
        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "exec: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }

    return 0;
}
