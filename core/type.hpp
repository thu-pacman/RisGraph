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
#include <functional>
#include <type_traits>

template <typename Data>
struct AdjEdge
{
    uint64_t nbr : 48;
    uint16_t num : 16;
    Data data;
    bool operator < (const AdjEdge<Data> &a)const 
    {
        if(nbr != a.nbr) return nbr < a.nbr;
        return data < a.data;
    }
    bool operator == (const AdjEdge<Data> &a)const 
    {
        return nbr == a.nbr && data == a.data;
    }
    using datatype = Data;
};

template <>
struct AdjEdge<void>
{
    uint64_t nbr : 48;
    uint16_t num : 16;
    const static uint64_t data = 0;
    bool operator < (const AdjEdge<void> &a)const 
    {
        return nbr < a.nbr;
    }
    bool operator == (const AdjEdge<void> &a)const 
    {
        return nbr == a.nbr;
    }
    using datatype = uint64_t;
};

template <typename Data>
struct Edge
{
    uint64_t src, dst;
    Data data;
    operator AdjEdge<Data>() const 
    {
        return AdjEdge<Data>{dst, 0, data};
    }
    using datatype = Data;
};

template <>
struct Edge<void>
{
    uint64_t src, dst;
    const static uint64_t data = 0;
    operator AdjEdge<void>() const 
    {
        return AdjEdge<void>{dst, 0};
    }
    using datatype = uint64_t;
};

static_assert(sizeof(AdjEdge<void>)==8, "Size of AdjEdge<void> is error.");
static_assert(sizeof(Edge<void>)==16, "Size of Edge<void> is error.");

namespace std
{
    template<typename T>
    struct hash<AdjEdge<T>>
    {
        uint64_t operator()(const AdjEdge<T>& e) const 
        {
            return (17lu*std::hash<uint64_t>()(e.nbr) + std::hash<T>()(e.data));
        }

    };
    template<>
    struct hash<AdjEdge<void>>
    {
        uint64_t operator()(const AdjEdge<void>& e) const 
        {
            return std::hash<uint64_t>()(e.nbr);
        }

    };
}

const uint64_t OPENMP_THRESHOLD = 8192;
#define THRESHOLD_OPENMP(para, length, ...) if((length) > OPENMP_THRESHOLD) \
{ \
    _Pragma(para) \
    __VA_ARGS__ \
} \
else  \
{ \
    __VA_ARGS__ \
} (void)0
