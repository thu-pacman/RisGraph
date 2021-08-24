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
#include <cstdint>

#define WORD_OFFSET(i) ((i) >> 6)
#define BIT_OFFSET(i) ((i) & 0x3f)
#define BEGIN_OF_WORD(i) ((i) << 6)

class Bitmap
{
public:
    uint64_t size;
    uint64_t * data;
    Bitmap()
        : size(0), data(nullptr)
    { 

    }
    Bitmap(uint64_t size) 
        : size(size)
    {
        data = new uint64_t [WORD_OFFSET(size)+1];
        clear();
    }
    ~Bitmap()
    {
        if (size != 0)
        {
            delete [] data;
        }
    }
    Bitmap(const Bitmap &a) = delete;
    Bitmap(Bitmap &&a)
        :size(a.size), data(a.data)
    {
        a.size = 0;
        a.data = nullptr;
    }
    Bitmap &operator = (Bitmap &&a)
    {
        size = a.size;
        data = a.data;
        a.size = 0;
        a.data = nullptr;
        return *this;
    }
    void clear()
    {
        uint64_t bm_size = WORD_OFFSET(size);
        #pragma omp parallel for
        for (uint64_t i=0;i<=bm_size;i++)
        {
            data[i] = 0;
        }
    }
    void fill()
    {
        uint64_t bm_size = WORD_OFFSET(size);
        #pragma omp parallel for
        for (uint64_t i=0;i<bm_size;i++)
        {
            data[i] = 0xfffffffffffffffful;
        }
        data[bm_size] = 0;
        for (uint64_t i=(bm_size<<6);i<size;i++)
        {
            data[bm_size] |= 1ul << BIT_OFFSET(i);
        }
    }
    uint64_t get_bit(uint64_t i)
    {
        return data[WORD_OFFSET(i)] & (1ul<<BIT_OFFSET(i));
    }
    void set_bit(uint64_t i)
    {
        __sync_fetch_and_or(data+WORD_OFFSET(i), 1ul<<BIT_OFFSET(i));
    }
    void clear_bit(uint64_t i)
    {
        __sync_fetch_and_and(data+WORD_OFFSET(i), UINT64_MAX - (1ul<<BIT_OFFSET(i)));
    }

};

