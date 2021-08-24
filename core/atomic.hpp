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
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <condition_variable>
#include <immintrin.h>
#include <emmintrin.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/futex.h>

#ifndef __x86_64__
#warning "The program is developed for x86-64 architecture only."
#endif
#if !defined(DCACHE1_LINESIZE) || !DCACHE1_LINESIZE
#ifdef DCACHE1_LINESIZE
#undef DCACHE1_LINESIZE
#endif
#define DCACHE1_LINESIZE 64
#endif
#define ____cacheline_aligned    __attribute__((aligned(DCACHE1_LINESIZE)))
#define ____force_inline    __attribute__((always_inline))

#define compiler_fence() asm volatile(""::: "memory")
#define cpu_fence() asm volatile ("mfence" ::: "memory")

template <typename T>
inline typename std::enable_if<sizeof(T) == 1, bool>::type cas(T *ptr, T oldv, T newv)
{
    static_assert(sizeof(char) == 1);
    return __sync_bool_compare_and_swap((char*)ptr, *((char*)&oldv), *((char*)&newv));
}

template <typename T>
inline typename std::enable_if<sizeof(T) == 2, bool>::type cas(T *ptr, T oldv, T newv)
{
    static_assert(sizeof(short) == 2);
    return __sync_bool_compare_and_swap((short*)ptr, *((short*)&oldv), *((short*)&newv));
}

template <typename T>
inline typename std::enable_if<sizeof(T) == 4, bool>::type cas(T *ptr, T oldv, T newv)
{
    static_assert(sizeof(int) == 4);
    return __sync_bool_compare_and_swap((int*)ptr, *((int*)&oldv), *((int*)&newv));
}

template <typename T>
inline typename std::enable_if<sizeof(T) == 8, bool>::type cas(T *ptr, T oldv, T newv)
{
    static_assert(sizeof(long) == 8);
    return __sync_bool_compare_and_swap((long*)ptr, *((long*)&oldv), *((long*)&newv));
}

template <typename T>
inline typename std::enable_if<sizeof(T) == 16, bool>::type cas(T *ptr, T oldv, T newv)
{
    static_assert(sizeof(__int128_t) == 16);
    return __sync_bool_compare_and_swap((__int128_t*)ptr, *((__int128_t*)&oldv), *((__int128_t*)&newv));
}

template <class T>
inline bool write_min(T *a, T b) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, "length not supported by cas.");
    T c;
    bool r = false;
    do
    {
        compiler_fence();
        c = *a;
    }
    while (c > b && !(r = cas(a, c, b)));
    return r;
}

template <class T>
inline bool write_max(T *a, T b) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, "length not supported by cas.");
    T c;
    bool r = false;
    do
    {
        compiler_fence();
        c = *a;
    }
    while (c < b && !(r = cas(a, c, b)));
    return r;
}

template <class T>
inline T write_add(T *a, T b) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, "length not supported by cas.");
    T newV, oldV;
    do
    {
        compiler_fence();
        oldV = *a;
        newV = oldV + b;
    }
    while (!cas(a, oldV, newV));
    return newV;
}

template <class T>
inline T write_sub(T *a, T b) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, "length not supported by cas.");
    T newV, oldV;
    do
    {
        compiler_fence();
        oldV = *a;
        newV = oldV - b;
    }
    while (!cas(a, oldV, newV));
    return newV;
}

class Futex
{
private:
    int futexp;
    int num_using;
    int futex(int *uaddr, int futex_op, int val, const struct timespec *timeout, int *uaddr2, int val3)
    {
        return syscall(SYS_futex, uaddr, futex_op, val, timeout, uaddr2, val3);
    }
public:
    void lock() {
        while(true) {
            write_add(&num_using, 1);
            if (cas(&futexp, 0, 1)) break;
            int ret = futex(&futexp, FUTEX_WAIT, 1, nullptr, nullptr, 0);
            write_sub(&num_using, 1);
            if(ret == -1 && errno != EAGAIN) throw std::runtime_error("Futex wait error.");
        }
    }
   
    void unlock() {
        if (cas(&futexp, 1, 0))
        {
            if(write_sub(&num_using, 1) == 0) return;
            int ret = futex(&futexp, FUTEX_WAKE, 1, nullptr, nullptr, 0);
            if(ret == -1) throw std::runtime_error("Futex wake error.");
        }
    }

};

class SpinLock
{
    std::atomic_flag locked;
public:
    SpinLock()
        :locked(false)
    {
    }
    SpinLock(const SpinLock&)
        :locked(false)
    {
    }
    SpinLock(SpinLock&&)
        :locked(false)
    {
    }
    SpinLock& operator=(const SpinLock&)
    {
        locked.clear();
        return *this;
    }
    SpinLock& operator=(SpinLock&&)
    {
        locked.clear();
        return *this;
    }
    void lock()
    {
        while (locked.test_and_set(std::memory_order_acquire)) _mm_pause();
    }
    void unlock()
    {
        locked.clear(std::memory_order_release);
    }
};

template <typename T>
____force_inline inline uint64_t atomic_append(std::vector<T> &array, std::atomic_uint64_t &length, const T &data, std::mutex &mutex)
{
    uint64_t idx = length.fetch_add(1, std::memory_order_acquire);
    if(idx >= array.size())
    {
        std::lock_guard<std::mutex> lock(mutex);
        if(idx >= array.size())
        {
            if(array.size() < 1024) 
                array.resize(1024);
            else
                array.resize(array.size()*2);
        }
        std::atomic_thread_fence(std::memory_order_release);
    }
    array[idx] = data;
    return idx;
}

____force_inline inline uint64_t atomic_append(std::vector<char> &array, std::atomic_uint64_t &length, const void * data, uint64_t datalen, std::mutex &mutex)
{
    uint64_t idx = length.fetch_add(datalen, std::memory_order_acquire);
    if(idx+datalen >= array.size())
    {
        std::lock_guard<std::mutex> lock(mutex);
        if(idx+datalen >= array.size())
        {
            if(array.size() < 1024) array.resize(1024);
            while(idx+datalen >= array.size()) array.resize(array.size()*2);
        }
        std::atomic_thread_fence(std::memory_order_release);
    }
    memcpy(array.data()+idx, data, datalen);
    return idx;
}


class SpinningRWLock
{
    std::atomic_uint64_t n_readers;
    std::atomic_bool writing;
public:
    SpinningRWLock()
        : n_readers(0), writing(false)
    {
    }

    void ReadLock()
    {
        while(true)
        {
            n_readers++;
            if (!writing) return;
            n_readers--;
            while (writing) _mm_pause();
        }
    }

    void WriteLock()
    {
        while(n_readers) _mm_pause();
        bool f = false;
        while(!writing.compare_exchange_strong(f, true))
        {
            _mm_pause();
            f = false;
        }
        while(n_readers) _mm_pause();
    }

    void ReadUnlock()
    {
        n_readers--;
    }

    void WriteUnlock()
    {
        writing = false;
    }
};

template<typename T>
class alignas(DCACHE1_LINESIZE) aligned_type
{
public:
    T& operator()()
    {
        return data;
    }
private:
    T data;
};
