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

#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
int main()
{
    std::vector<std::pair<uint64_t, uint64_t>> edges;
    uint64_t x, y;
    while(scanf("%lu%lu", &x, &y) != EOF)
    {
        edges.emplace_back(x, y);
    }
    std::random_shuffle(edges.begin(), edges.end());
    for(auto p : edges)
    {
        fwrite(&p.first, sizeof(p.first), 1, stdout);
        fwrite(&p.second, sizeof(p.second), 1, stdout);
    }
}
