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
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <cstdint>
#include <string>
#include <system_error>

static std::pair<std::pair<uint64_t, uint64_t>*, uint64_t> mmap_binary(std::string path)
{
    int fd = open(path.c_str(), O_RDONLY, 0640);
    if(fd == -1) throw std::runtime_error(std::string("open path ") + path + " error.");
    uint64_t size = lseek(fd, 0, SEEK_END);
    void *ret = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if(ret == MAP_FAILED) throw std::runtime_error("mmap error.");
    madvise(ret, size, MADV_DONTNEED);
    close(fd);
    return {(std::pair<uint64_t, uint64_t>*)ret, size/sizeof(std::pair<uint64_t, uint64_t>)};
}

static void* mmap_alloc(size_t capacity)
{
    void *buffer = mmap(nullptr, capacity, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if(buffer == MAP_FAILED) throw std::runtime_error("mmap error.");
    return buffer;
}

static void mmap_free(void* buffer, size_t capacity)
{
    int ret = munmap(buffer, capacity);
    if(ret != 0) throw std::runtime_error("munmap error.");
}
