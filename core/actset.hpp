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
#include <vector>
#include <mutex>
#include "type.hpp"
#include "bitmap.hpp"
#include "atomic.hpp"
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

class ActiveSet
{
public:
    ActiveSet(uint64_t _size) 
        : size(_size), dense_threshold(_size*dense_rate),
        dense_active(_size),
        dense(false), rw_lock(), mutex(), converted()
    {
        for(uint64_t i=0;i<buckets;i++)
        {
            local_sparse_active[i]().clear();
            local_sparse_active_length[i]() = 0;
        }
    }
    ActiveSet(const ActiveSet &a) = delete;
    ActiveSet(ActiveSet &&a)
        : size(std::move(a.size)), dense_threshold(std::move(a.dense_threshold)),
        dense_active(std::move(a.dense_active)),
        dense(a.dense.load()), rw_lock(), mutex(), converted(std::move(a.converted))
    {
        for(uint64_t i=0;i<buckets;i++)
        {
            local_sparse_active[i] = std::move(a.local_sparse_active[i]);
            local_sparse_active_length[i]() = a.local_sparse_active_length[i]().load();
        }
    }
    ActiveSet &operator = (ActiveSet &&a)
    {
        size = std::move(a.size);
        dense_threshold = std::move(a.dense_threshold);
        dense_active = std::move(a.dense_active);
        for(uint64_t i=0;i<buckets;i++)
        {
            local_sparse_active[i] = std::move(a.local_sparse_active[i]);
            local_sparse_active_length[i]() = a.local_sparse_active_length[i]().load();
        }
        dense = a.dense.load();
        converted = std::move(a.converted);
        return *this;
    }

    void clear(bool force_dense = false)
    {
        for(uint64_t i=0;i<buckets;i++) local_sparse_active_length[i]().store(0, std::memory_order_relaxed);
        if(force_dense) dense_active.clear();
        dense = force_dense;
        converted = true;
    }

    void fill()
    {
        dense_active.fill();
        dense = true;
    }

    bool is_dense() const 
    {
        return dense;
    }

    const Bitmap& get_dense() const
    {
        return dense_active;
    }

    const std::vector<uint64_t>& get_sparse()
    {
        if(!converted) convert();
        return local_sparse_active[0]();
    }

    uint64_t get_sparse_length()
    {
        if(!converted) convert();
        return local_sparse_active_length[0]().load(std::memory_order_relaxed);
    }

    //unsafe
    Bitmap& get_dense_ref()
    {
        return dense_active;
    }

    //unsafe
    std::vector<uint64_t>& get_sparse_ref()
    {
        return local_sparse_active[0]();
    }

    ____force_inline inline void active(uint64_t id)
    {
        converted = false;
        if(dense.load(std::memory_order_relaxed))
        {
            dense_active.set_bit(id);
        }
        else
        {
            const auto hasher = std::hash<std::thread::id>();
            auto tid = hasher(std::this_thread::get_id())%buckets;
            //auto cur_len = sparse_active_length.load(std::memory_order_relaxed);
            //if(cur_len+512 > dense_threshold) rw_lock.ReadLock();
            //rw_lock.ReadLock();
            auto idx = atomic_append(local_sparse_active[tid](), local_sparse_active_length[tid](), id, mutex[tid]());
            //rw_lock.ReadUnlock();
            //if(idx > dense_threshold)
            //{
            //    rw_lock.WriteLock();
            //    if(!dense.load(std::memory_order_acquire))
            //    {
            //        dense_active.clear();
            //        #pragma omp parallel for
            //        for(uint64_t i=0;i<=idx;i++)
            //        {
            //            dense_active.set_bit(sparse_active[i]);
            //        }
            //        dense.store(true, std::memory_order_release);
            //    }
            //    rw_lock.WriteUnlock();
            //    dense_active.set_bit(id);
            //}
        }
    }

    void convert()
    {
        if(!converted) 
            converted = true; 
        else 
            return;

        offsets[0] = local_sparse_active_length[0]().load(std::memory_order_relaxed);
        for(uint64_t i=1;i<buckets;i++)
        {
            offsets[i] = offsets[i-1] + local_sparse_active_length[i]().load(std::memory_order_relaxed);
            local_sparse_active_length[i]().store(0, std::memory_order_relaxed);
        }
        uint64_t len = offsets[buckets-1];
        if(len > local_sparse_active[0]().size()) local_sparse_active[0]().resize(nextPowerOf2(len));
        if(len-offsets[0] > 8*OPENMP_THRESHOLD)
        {
            tbb::enumerable_thread_specific<uint64_t> next_a(1);
            //thread_local uint64_t next = 1;
            tbb::parallel_for(tbb::blocked_range<uint64_t>(0, len-offsets[0], 256),
            [&](const tbb::blocked_range<uint64_t> &range)
            {
                uint64_t next = next_a.local();
                if(next >= buckets || offsets[next-1]-offsets[0] > range.begin()) next = 1;
                for(uint64_t k=range.begin();k!=range.end();k++)
                {
                    auto i = k+offsets[0];
                    while(offsets[next] <= i) next++;
                    auto v_i = local_sparse_active[next]()[i-offsets[next-1]];
                    local_sparse_active[0]()[i] = v_i;
                }
                next_a.local() = next;
            });
        }
        else
        {
            uint64_t cur = offsets[0];
            for(uint64_t i=1;i<buckets;i++)
            {
                uint64_t size = offsets[i]-offsets[i-1];
                for(uint64_t j=0;j<size;j++)
                {
                    local_sparse_active[0]()[cur+j] = local_sparse_active[i]()[j];
                }
                cur += size;
            }

        }
        local_sparse_active_length[0]().store(len, std::memory_order_relaxed);
    }

private:
    uint64_t size, dense_threshold;
    static constexpr double dense_rate = 1.0/8; 
    static constexpr uint64_t buckets = 8;
    Bitmap dense_active;
    aligned_type<std::vector<uint64_t>> local_sparse_active[buckets];
    aligned_type<std::atomic_uint64_t> local_sparse_active_length[buckets];
    uint64_t offsets[buckets];
    std::atomic_bool dense;
    aligned_type<SpinningRWLock> rw_lock;
    aligned_type<std::mutex> mutex[buckets];
    bool converted;

    uint64_t nextPowerOf2(uint64_t n)   
    { 
        n--; 
        n |= n >> 1; 
        n |= n >> 2; 
        n |= n >> 4; 
        n |= n >> 8; 
        n |= n >> 16; 
        n |= n >> 32; 
        n++; 
        return n; 
    }  
};
