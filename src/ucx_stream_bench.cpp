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
#include <omp.h>
#include "core/ucx_stream.hpp"

const uint64_t num = 10000000;
const uint64_t size = 32;
const uint64_t clients = 48*16;
int main(int argc, char** argv)
{
    const uint64_t fibers_per_thread = clients/omp_get_max_threads();
    auto streams = UCXStream::make_ucx_stream(argc > 1 ? argv[1] : "", 2333, omp_get_max_threads(), fibers_per_thread);

    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        uint64_t tid = omp_get_thread_num();
        uint64_t ths = omp_get_num_threads();
        uint64_t fbs = ths*fibers_per_thread;
        std::vector<boost::fibers::fiber> f;
        for(uint64_t i=0;i<fibers_per_thread;i++) f.emplace_back([&, lfid=i]()
        {
            uint64_t fid = tid*fibers_per_thread+lfid;
            char send_buffer[size], recv_buffer[size];
            for(uint64_t i=0;i<num;i++)
            {
                //fprintf(stderr, "%lu\n", i);
                if(argc > 1)
                {
                    *(uint64_t*)send_buffer = i*fbs+fid;
                    streams[tid].send(send_buffer, size, lfid);
                    streams[tid].recv(recv_buffer, size, lfid);
                    if(*(uint64_t*)recv_buffer != i*fbs+fid) throw std::runtime_error("data is wrong");
                }
                else
                {
                    streams[tid].recv(recv_buffer, size, lfid);
                    if(*(uint64_t*)recv_buffer != i*fbs+fid) throw std::runtime_error("data is wrong");
                    *(uint64_t*)send_buffer = i*fbs+fid;
                    streams[tid].send(send_buffer, size, lfid);
                }
            }
        });
        for(uint64_t i=0;i<fibers_per_thread;i++) f[i].join();
    }
    auto end = std::chrono::high_resolution_clock::now();
    printf("%lf s\n", 1e-9*(end-start).count());
    printf("%lf op/s\n", num*fibers_per_thread*omp_get_max_threads()*1e9/(end-start).count());
    return 0;

}
