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

#include <chrono>
#include <random>
#include <mutex>
#include <omp.h>
#include <absl/container/btree_map.h>
const uint64_t shards = 128;
const uint64_t num = 100000000;
std::mutex mutex[shards];
absl::btree_map<std::tuple<uint64_t, uint64_t, uint64_t>, bool> tree[shards];
int main()
{
    auto bench = [&]()
    {
        auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel
        {
            auto rand = std::mt19937_64(omp_get_thread_num());
            for(uint64_t i=0;i<num;i++)
            {
                auto a = rand(), b = rand(), c = rand();
                std::lock_guard<std::mutex> lock(mutex[a%shards]);
                tree[a%shards].emplace(std::make_tuple(a, b, c), true);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        fprintf(stderr, "%lf s\n", 1e-9*(end-start).count());
        fprintf(stderr, "%lf op/s\n", 1e9*omp_get_max_threads()*num/(end-start).count());
    };
    bench();
    bench();
}
