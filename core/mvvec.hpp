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
#include <cassert>
#include <vector>
#include <atomic>
#include "atomic.hpp"

template <typename DataType>
class MVVec
{
    struct Node
    {
        uint64_t begin_time;
        uint64_t pre_pointer;
        DataType data;
    };

public:
    MVVec(uint64_t _length)
        :length(_length), pool(), 
            index(length, (uint64_t)-1), deleted(),
            pool_length(), deleted_tail(), deleted_head()
    {
    }

    bool set(const uint64_t &id, const uint64_t &timestamp, const DataType &data, const uint64_t &gc_timestamp = (uint64_t)-1)
    {
        assert(id < length);
        uint64_t cur_idx = index[id];
        bool r = false;
        if(cur_idx == (uint64_t)-1 || (pool[cur_idx].begin_time < timestamp && pool[cur_idx].data != data))
        {
            uint64_t new_idx = create();
            pool[new_idx].begin_time = timestamp;

            do
            {
                compiler_fence();
                cur_idx = index[id];
                pool[new_idx].pre_pointer = cur_idx;
            } while((cur_idx == (uint64_t)-1 || (pool[cur_idx].begin_time < timestamp && pool[cur_idx].data != data))
                    && !(r = cas(&index[id], cur_idx, new_idx)));
            if(!r)
            {
                recycle(new_idx);
            }
            else
            {
                cur_idx = new_idx;
            }
        }

        //overwrite same timestamp
        if(pool[cur_idx].begin_time == timestamp) pool[cur_idx].data = data;

        //gc old timestamp
        if(gc_timestamp != (uint64_t)-1) gc(id, gc_timestamp);
        return r;
    }

    DataType get(const uint64_t &id, const uint64_t &timestamp)
    {
        assert(id < length);
        for(uint64_t cur_idx = index[id]; cur_idx != (uint64_t)-1; cur_idx = pool[cur_idx].pre_pointer)
        {
            if(pool[cur_idx].begin_time <= timestamp) return pool[cur_idx].data;
        }
        assert(false);
        return DataType();
    }

    void gc(const uint64_t &vid, const uint64_t &timestamp)
    {
        uint64_t cur_idx = index[vid];
        uint64_t next_idx = pool[cur_idx].pre_pointer;
        if(pool[cur_idx].begin_time > timestamp)
        {
            while(next_idx != (uint64_t)-1)
            {
                uint64_t tmp = next_idx;
                next_idx = pool[next_idx].pre_pointer;
                if(pool[tmp].begin_time <= timestamp)
                {
                    recycle(tmp);
                    if(pool[cur_idx].begin_time > timestamp)
                    {
                        pool[cur_idx].pre_pointer = (uint64_t)-1;
                    }
                }
                cur_idx = next_idx;

            }
        }
    }
    void gc(const uint64_t &timestamp)
    {
        #pragma omp parallel for
        for(uint64_t i=0; i<length; i++)
        {
            gc(i, timestamp);
        }
    }


private:
    uint64_t length;
    std::vector<Node> pool;
    std::vector<uint64_t> index, deleted;
    std::atomic_uint64_t pool_length, deleted_tail, deleted_head;
    std::mutex pool_mutex, deleted_mutex;

    void recycle(uint64_t idx)
    {
        atomic_append(deleted, deleted_tail, idx, deleted_mutex);
        uint64_t old_head = deleted_head, old_tail = deleted_tail;
        if(old_tail - old_head + 4096 < old_head && old_head > 1000000)
        {
            std::lock_guard<std::mutex> lock(deleted_mutex);
            uint64_t length = 0;

            bool r = false;
            do
            {
                //overlap
                if(old_tail - old_head >= old_head) break;
                while(length < old_tail-old_head)
                {
                    deleted[length] = deleted[length + old_head];
                    length++;
                }
            } while(!(r = deleted_tail.compare_exchange_weak(old_tail, old_tail-old_head)));

            if(r) deleted_head.fetch_sub(old_head);
        }

    }

    uint64_t create()
    {
        //empty padding, avoid solving hazard
        if(deleted_head + 4096 < deleted_tail)
        {
            return deleted[deleted_head.fetch_add(1)];
        }
        return atomic_append(pool, pool_length, {}, pool_mutex);
    }

};
