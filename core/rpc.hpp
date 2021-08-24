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
#include <chrono>

struct ClientUpdateRequest
{
    uint32_t client_id; 
    enum class Type : uint32_t
    {
        Add,
        Del,
        End
    } type;
    uint64_t src;
    uint64_t dst;
    uint64_t data;
    std::chrono::high_resolution_clock::time_point request_time;
};

struct ClientUpdateResponse
{
    uint32_t client_id; 
    uint64_t version_id;
};

struct ClientTxnUpdateRequest
{
    static constexpr uint32_t max_num_updates = 16;
    uint32_t client_id; 
    uint32_t num_updates;
    enum class Type : uint32_t
    {
        Add,
        Del,
        End
    } types[16];
    uint64_t srcs[16];
    uint64_t dsts[16];
    uint64_t datas[16];
    std::chrono::high_resolution_clock::time_point request_time;
};

struct ClientTxnUpdateResponse
{
    uint32_t client_id; 
    uint64_t version_ids[16];
};

struct KVUpdateRequest
{
    uint32_t client_id; 
    enum class Type : uint32_t
    {
        Add,
        Del,
        His,
        Ping,
        Version,
        End
    } type;
    uint64_t src;
    uint64_t dst;
    uint64_t data;
    uint64_t version_id;
    uint64_t label_data;
};

struct KVUpdateResponse
{
    uint32_t client_id; 
    uint64_t version_id;
};
