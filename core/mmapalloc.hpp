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
#include <new>
#include <limits>
#include <memory>
#include <atomic>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <iostream>
#include <tbb/enumerable_thread_specific.h>
#include "atomic.hpp"

class MMapPool
{
public:
    MMapPool(std::string _path, int advise = MADV_RANDOM, uint64_t _capacity = 1ul << 40)
        : capacity(_capacity)
    {
        for(uint64_t i=min_block_size;i<max_block_size;i++)
        {
            std::string path = _path+"."+std::to_string(i);
            fd[i] = open(path.c_str(), O_CREAT | O_RDWR, 0640);
            if(fd[i] == -1) throw std::runtime_error(std::string("open path ") + path + " error.");
            file_size[i] = lseek(fd[i], 0, SEEK_END);
            data[i] = mmap(nullptr, capacity, PROT_READ | PROT_WRITE, MAP_SHARED, fd[i], 0);
            if(data[i] == MAP_FAILED) throw std::runtime_error("mmap error.");
            int ret = madvise(data[i], capacity, advise);
            if(ret != 0) throw std::runtime_error("madvise error.");
            if (file_size[i] == 0) 
            {
                // initialize
                if(ftruncate(fd[i], GIGABYTE) != 0) throw std::runtime_error("ftruncate error.");
                file_size[i] = GIGABYTE;
            }
            used_size[i] = 0;
            futex[i] = Futex();
            global_free_blocks[i].clear();
        }
    }

    ~MMapPool() noexcept
    {
        uint64_t overall = 0;
        for(uint8_t order = min_block_size;order < max_block_size;order++)
        {
            if(!used_size[order].load()) continue;
            overall += used_size[order].load();
            std::cerr << "Order " << (int)order << " : " << used_size[order].load() << std::endl;
            //std::cerr << "Order " << (int)order << ":";
            //std::cerr << "\n\t total :" << used_size[order].load();
            ////std::cerr << "\n\t local :";
            //uint64_t freed = 0;
            //for(auto iter = local_free_blocks[order].begin();iter != local_free_blocks[order].end();iter++)
            //{
            //    freed += iter->size();
            //    //std::cerr << " ,"[iter != local_free_blocks[order].begin()] << iter->size();
            //}
            //freed += global_free_blocks[order].size();
            ////std::cerr << "\n\t global: " << global_free_blocks[order].size();
            //std::cerr << "\n\t used  : " << used_size[order].load()-freed*(1lu<<order);
            //std::cerr << "\n\t free  : " << freed*(1lu<<order) << " " << freed*(8+(1lu<<order));
            //std::cerr << std::endl;
        }
        std::cerr << "Overall : " << overall << std::endl;
    }

    void* alloc(uint64_t size)
    {
        auto order = size2order(size);
        order = std::max(order, min_block_size);
        uintptr_t position = 0;
        if (order < local_gc_block_size)
        {
            position = local_pop(order);
            if(position == 0)
            {
                position = global_pop(order);
            }
        }
        else
        {
            position = global_pop(order);
        }
        if (position == 0)
        {
            if(order < local_gc_block_size)
            {
                uint64_t block_size = 1ul << order;
                uint64_t alloc_size = 512*block_size;
                position = used_size[order].fetch_add(alloc_size);
                if (position + alloc_size >= file_size[order]) 
                {
                    grow_file((file_size[order] + alloc_size) / GIGABYTE * GIGABYTE * 2, order);
                }
                auto &free_blocks = local_free_blocks[order].local();
                for(int i=1;i<512;i++) free_blocks.push_back(position+i*block_size);

            }
            else
            {
                uint64_t block_size = 1ul << order;
                position = used_size[order].fetch_add(block_size);
                if (position + block_size >= file_size[order]) 
                {
                    grow_file((file_size[order] + block_size) / GIGABYTE * GIGABYTE * 2, order);
                }
            }
        }
        void * ptr = (void*)((uintptr_t)data[order] + position);
        return ptr;
    }

    void free(void* ptr, uint64_t size)
    {
        auto order = size2order(size);
        order = std::max(order, min_block_size);
        uintptr_t position = (uintptr_t)ptr-(uintptr_t)data[order];
        if (order < local_gc_block_size)
        {
            if(!local_push(position, order))
            {
                global_push(position, order);
            }
        }
        else
        {
            global_push(position, order);
        }
    }

private:
    static constexpr uint8_t min_block_size = 4;
    static constexpr uint8_t max_block_size = 40;
    static constexpr uint8_t local_gc_block_size = 20;
    static constexpr uint64_t local_gc_max_size = 512*1024;
    static constexpr uint64_t GIGABYTE = 1lu << 30;
    const uint64_t capacity;
    int fd[max_block_size];
    void *data[max_block_size];
    std::atomic_size_t used_size[max_block_size];
    std::atomic_size_t file_size[max_block_size];
    Futex futex[max_block_size];
    tbb::enumerable_thread_specific<std::vector<uintptr_t>> local_free_blocks[max_block_size];
    std::vector<uintptr_t> global_free_blocks[max_block_size];

    uint8_t size2order(uint64_t size)
    {
        uint8_t ans = 0;
        uint64_t pow2 = 1;
        while(pow2 < size)
        {
            ans ++;
            pow2 <<= 1;
        }
        return ans;
    }

    void grow_file(uint64_t expected_size, uint8_t order)
    {
        std::lock_guard<Futex> lock(futex[order]);
        if (file_size[order] < expected_size) 
        {
            if(ftruncate(fd[order], expected_size) != 0) throw std::runtime_error("ftruncate error.");
            std::cerr << "Order:" << (int)order << " Grow from " << file_size[order].load() << " to " << expected_size << std::endl;
            file_size[order] = expected_size;
        }
    }

    uintptr_t global_pop(uint8_t order)
    {
        if (global_free_blocks[order].size() == 0) return 0;
        std::lock_guard<Futex> lock(futex[order]);
        size_t position = 0;
        if (global_free_blocks[order].size() > 0) 
        {
            position = global_free_blocks[order].back();
            global_free_blocks[order].pop_back();
        }
        return position;
    }

    uintptr_t local_pop(uint8_t order)
    {
        auto &free_blocks = local_free_blocks[order].local();
        size_t position = 0;
        if (free_blocks.size() > 0) 
        {
            position = free_blocks.back();
            free_blocks.pop_back();
        }
        return position;
    }

    bool global_push(uintptr_t position, uint8_t order)
    {
        std::lock_guard<Futex> lock(futex[order]);
        global_free_blocks[order].emplace_back(position);
        return true;
    }

    bool local_push(uintptr_t position, uint8_t order)
    {
        auto &free_blocks = local_free_blocks[order].local();
        if (free_blocks.size() < local_gc_max_size) 
        {
            free_blocks.push_back(position);
            return true;
        }
        return false;
    }

};

extern std::shared_ptr<MMapPool> default_mmap_pool;

template <class T>
class MMapAllocator
{
public:
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using void_pointer = void*;
    using const_void_pointer = const void*;
    using value_type = T;
    using size_type = uint64_t;
    using difference_type = int64_t;
    MMapAllocator(std::shared_ptr<MMapPool> _pool = default_mmap_pool)
        :pool(_pool)
    {
    }
    template <class U>
    MMapAllocator(const MMapAllocator<U>& allocator) noexcept
        :pool(allocator.pool)
    {
    }
    T* allocate(std::size_t n) 
    {
        if(!pool) return std_allocator.allocate(n);
        if(n > std::size_t(-1) / sizeof(T)) throw std::bad_alloc();
        if(auto p = static_cast<T*>(pool->alloc(n*sizeof(T)))) return p;
        throw std::bad_alloc();
    }
    void deallocate(T* p, std::size_t n) noexcept
    {
        if(!pool) return std_allocator.deallocate(p, n);
        pool->free(p, n*sizeof(T));
    }

    const std::shared_ptr<MMapPool> pool;

    //for google dense hash
    size_type max_size() const {
      return static_cast<size_type>(-1) / sizeof(value_type);
    }
private:
    std::allocator<T> std_allocator;
};

template <class T1, class T2>
inline static bool operator == (const MMapAllocator<T1> &x, const MMapAllocator<T2> &y) noexcept 
{
    return x.pool == y.pool;
}

template <class T1, class T2>
inline static bool operator != (const MMapAllocator<T1> &x, const MMapAllocator<T2> &y) noexcept 
{
    return x.pool != y.pool;
}


