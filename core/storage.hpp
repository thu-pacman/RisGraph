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
#include <cstdio>
#include <vector>
#include <exception>
#include <functional>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <sparsehash/dense_hash_map>
#include <sparsehash/sparse_hash_map>
#include <unordered_map>
#include <art/radix_map.h>
#include <absl/container/btree_map.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/node_hash_map.h>
#include "type.hpp"
#include "atomic.hpp"
#include "mmapalloc.hpp"

namespace absl
{
    template <typename Key, typename Value, typename Compare = std::less<Key>,
              typename Alloc = std::allocator<std::pair<const Key, Value>>>
    class my_btree_map
        : public container_internal::btree_map_container<
              container_internal::btree<container_internal::map_params<
                  Key, Value, Compare, Alloc, /*TargetNodeSize=*/4096,
                  /*Multi=*/false>>> 
    {
        using Base = typename my_btree_map::btree_map_container;
    
    public:
        my_btree_map() {}
        using Base::Base;
        using Base::begin;
        using Base::cbegin;
        using Base::end;
        using Base::cend;
        using Base::empty;
        using Base::max_size;
        using Base::size;
        using Base::clear;
        using Base::erase;
        using Base::insert;
        using Base::emplace;
        using Base::emplace_hint;
        using Base::try_emplace;
        using Base::extract;
        using Base::merge;
        using Base::swap;
        using Base::at;
        using Base::contains;
        using Base::count;
        using Base::equal_range;
        using Base::find;
        using Base::operator[];
        using Base::get_allocator;
        using Base::key_comp;
        using Base::value_comp;
    };
    template <typename K, typename V, typename C, typename A>
    void swap(my_btree_map<K, V, C, A> &x, my_btree_map<K, V, C, A> &y)
    {
        return x.swap(y);
    }
}

namespace storage
{
    namespace data
    {
        template <typename Data>
        class Vector : public std::vector<Data>
        {
        };

        template <typename Data>
        class MMapVector : public std::vector<Data, MMapAllocator<Data>>
        {
        public:
            MMapVector()
                :std::vector<Data, MMapAllocator<Data>>(MMapAllocator<Data>(default_mmap_pool))
            {
            }
            MMapVector(const MMapVector&) = default;
            MMapVector(MMapVector&&) = default;
            MMapVector& operator=(const MMapVector&) = default;
            MMapVector& operator=(MMapVector&&) = default;
            ~MMapVector() = default;
        };
    }
    
    namespace index
    {
        template <typename Key, typename Value>
        class DenseHashMap : public google::dense_hash_map<Key, Value>
        {
        public:
            DenseHashMap(uint64_t hint_size = 0)
                :google::dense_hash_map<Key, Value>(hint_size)
            {
            }
            void set_empty_key(Key _empty_key)
            {
                google::dense_hash_map<Key, Value>::set_empty_key(_empty_key);
            }
            DenseHashMap(const DenseHashMap&) = default;
            DenseHashMap(DenseHashMap&&) = default;
            DenseHashMap& operator=(const DenseHashMap&) = default;
            DenseHashMap& operator=(DenseHashMap&&) = default;
            ~DenseHashMap() = default;

        };

        template <typename Key, typename Value>
        class MMapDenseHashMap : public google::dense_hash_map<Key, Value, std::hash<Key>, std::equal_to<Key>, MMapAllocator<std::pair<const Key, Value>>>
        {
        public:
            MMapDenseHashMap(uint64_t hint_size = 0)
                :google::dense_hash_map<Key, Value, std::hash<Key>, std::equal_to<Key>, MMapAllocator<std::pair<const Key, Value>>>(hint_size, 
                        std::hash<Key>(), std::equal_to<Key>(), MMapAllocator<std::pair<const Key, Value>>(default_mmap_pool))
            {
            }
            void set_empty_key(Key _empty_key)
            {
                google::dense_hash_map<Key, Value, std::hash<Key>, std::equal_to<Key>, MMapAllocator<std::pair<const Key, Value>>>::set_empty_key(_empty_key);
            }
            MMapDenseHashMap(const MMapDenseHashMap&) = default;
            MMapDenseHashMap(MMapDenseHashMap&&) = default;
            MMapDenseHashMap& operator=(const MMapDenseHashMap&) = default;
            MMapDenseHashMap& operator=(MMapDenseHashMap&&) = default;
            ~MMapDenseHashMap() = default;

        };

        template <typename Key, typename Value>
        class SparseHashMap : public google::sparse_hash_map<Key, Value>
        {
        public:
            SparseHashMap(uint64_t hint_size = 0)
                :google::sparse_hash_map<Key, Value>(hint_size)
            {
            }
            void set_empty_key(Key _empty_key)
            {
            }
            SparseHashMap(const SparseHashMap&) = default;
            SparseHashMap(SparseHashMap&&) = default;
            SparseHashMap& operator=(const SparseHashMap&) = default;
            SparseHashMap& operator=(SparseHashMap&&) = default;
            ~SparseHashMap() = default;

        };

        template <typename Key, typename Value>
        class StdHashMap : public std::unordered_map<Key, Value> {
        public:
            StdHashMap(uint64_t hint_size = 0)
                :std::unordered_map<Key, Value>(hint_size) {
            }
            void set_empty_key(Key _empty_key)
            {
            }
            StdHashMap(const StdHashMap&) = default;
            StdHashMap(StdHashMap&&) = default;
            StdHashMap& operator=(const StdHashMap&) = default;
            StdHashMap& operator=(StdHashMap&&) = default;
            ~StdHashMap() = default;
        };

        struct serialize {
            uint64_t operator()(AdjEdge<void> t) const noexcept {
                uint64_t nbr = t.nbr;
                return nbr;
            }
            template<typename T>
            auto operator()(AdjEdge<T> t) const noexcept {
                uint64_t nbr = t.nbr;
                return art::key_transform<std::pair<uint64_t, T>>()(std::pair<uint64_t, T>{nbr, t.data});
            }
        };


        template <typename Key, typename Value>
        class RadixMap : public art::radix_map<Key, Value, serialize> {
        public:
            RadixMap(uint64_t hint_size = 0)
                :art::radix_map<Key, Value, serialize>() {
            }
            void set_empty_key(Key _empty_key)
            {
            }
            RadixMap(const RadixMap&) = default;
            RadixMap(RadixMap&&) = default;
            RadixMap& operator=(const RadixMap&) = default;
            RadixMap& operator=(RadixMap&&) = default;
            ~RadixMap() = default;
        };

        template <typename Key, typename Value>
        class BtreeMap : public absl::btree_map<Key, Value> {
        public:
            BtreeMap(uint64_t hint_size = 0)
                :absl::btree_map<Key, Value>() {
            }
            void set_empty_key(Key _empty_key)
            {
            }
            BtreeMap(const BtreeMap&) = default;
            BtreeMap(BtreeMap&&) = default;
            BtreeMap& operator=(const BtreeMap&) = default;
            BtreeMap& operator=(BtreeMap&&) = default;
            ~BtreeMap() = default;
        };

        template <typename Key, typename Value>
        class MMapBtreeMap : public absl::my_btree_map<Key, Value, std::less<Key>, MMapAllocator<std::pair<const Key, Value>>> {
        public:
            MMapBtreeMap(uint64_t hint_size = 0)
                :absl::my_btree_map<Key, Value, std::less<Key>, MMapAllocator<std::pair<const Key, Value>>>() {
            }
            void set_empty_key(Key _empty_key)
            {
            }
            MMapBtreeMap(const MMapBtreeMap&) = default;
            MMapBtreeMap(MMapBtreeMap&&) = default;
            MMapBtreeMap& operator=(const MMapBtreeMap&) = default;
            MMapBtreeMap& operator=(MMapBtreeMap&&) = default;
            ~MMapBtreeMap() = default;
        };
        
        template <typename Key, typename Value>
        class FlatHashMap : public absl::flat_hash_map<Key, Value, std::hash<Key>> {
        public:
            FlatHashMap(uint64_t hint_size = 0)
                :absl::flat_hash_map<Key, Value, std::hash<Key>>() {
            }
            void set_empty_key(Key _empty_key)
            {
            }
            FlatHashMap(const FlatHashMap&) = default;
            FlatHashMap(FlatHashMap&&) = default;
            FlatHashMap& operator=(const FlatHashMap&) = default;
            FlatHashMap& operator=(FlatHashMap&&) = default;
            ~FlatHashMap() = default;
        };

        template <typename Key, typename Value>
        class NodeHashMap : public absl::node_hash_map<Key, Value, std::hash<Key>> {
        public:
            NodeHashMap(uint64_t hint_size = 0)
                :absl::node_hash_map<Key, Value, std::hash<Key>>(hint_size) {
            }
            void set_empty_key(Key _empty_key)
            {
            }
            NodeHashMap(const NodeHashMap&) = default;
            NodeHashMap(NodeHashMap&&) = default;
            NodeHashMap& operator=(const NodeHashMap&) = default;
            NodeHashMap& operator=(NodeHashMap&&) = default;
            ~NodeHashMap() = default;
        };

    }
}

template <typename AdjEdgeType = AdjEdge<void>,
         template<typename> typename DataType = storage::data::Vector,
         template<typename, typename> typename IndexType = storage::index::DenseHashMap>
class IndexedEdgeStorage
{
public:
    using adjedge_type = AdjEdgeType;
    using adjlist_type = DataType<adjedge_type>;
    using adjmap_type = IndexType<adjedge_type, uint64_t>;
    using adjlist_iter_type = typename DataType<adjedge_type>::iterator;
    using lock_type = SpinLock;
    //adjlist_type must have size() [pos].
    static constexpr bool edge_random_accessable = true;

    struct AdjStruct
    {
        adjlist_type adjlist;
        lock_type futex;
        std::shared_ptr<adjmap_type> adjlist_map;
        uint64_t degree;
    };

    IndexedEdgeStorage(uint64_t _vertices = 0, uint64_t _need_index_threshold = 512)
        : vertices(_vertices), 
        empty_key(),
        need_index_threshold(_need_index_threshold),
        array_of_adjstruct(vertices)
    {
        empty_key.nbr = 281474976710655; // -1 in uint64_t:48
    }
    IndexedEdgeStorage(const IndexedEdgeStorage &) = default;
    IndexedEdgeStorage(IndexedEdgeStorage &&) = default;
    IndexedEdgeStorage& operator=(const IndexedEdgeStorage &) = default;
    IndexedEdgeStorage& operator=(IndexedEdgeStorage && a) = default;
    ~IndexedEdgeStorage() = default;

    void resize(uint64_t _vertices)
    {
        vertices = _vertices;
        array_of_adjstruct.resize(_vertices);
    }

    lock_type& get_lock(uint64_t vid)
    {
        return array_of_adjstruct[vid].futex;
    }

    uint64_t get_degree(uint64_t vid) const
    {
        return array_of_adjstruct[vid].degree;
    }

    std::pair<adjlist_iter_type, adjlist_iter_type> get_adjlist_iter(uint64_t vid)
    {
        return {array_of_adjstruct[vid].adjlist.begin(), array_of_adjstruct[vid].adjlist.end()};
    }

    adjlist_type& get_adjlist(uint64_t vid)
    {
        if(!edge_random_accessable) throw std::runtime_error("Storage is not edge_random_accessable.");
        return array_of_adjstruct[vid].adjlist;
    }

    uint64_t update_edge(adjedge_type ae, uint64_t vid, int update)
    {
        auto &adj_struct = array_of_adjstruct[vid];
        std::lock_guard<lock_type> lock(adj_struct.futex);

        auto &degree = adj_struct.degree;
        auto &adjlist = adj_struct.adjlist;
        auto &adjlist_map = adj_struct.adjlist_map;
        auto has_map = adjlist_map != nullptr;

        ae.num = 0;

        uint64_t current_size = 0;
        if(adjlist.size() >= need_index_threshold && !has_map)
        {
            adjlist_map = std::make_shared<adjmap_type>(adjlist.size());
            adjlist_map->set_empty_key(empty_key);
            for(uint64_t i=0;i<adjlist.size();i++)
            {
                adjlist_map->insert(std::make_pair(adjlist[i], i));
            }
            has_map = true;
        }

        if(has_map)
        {
            auto iter = adjlist_map->find(ae);
            if(iter == adjlist_map->end())
            {
                adjlist.emplace_back(ae);
                iter = adjlist_map->insert(std::make_pair(ae, adjlist.size()-1)).first;
            }
            current_size = adjlist[iter->second].num;
            if((int)adjlist[iter->second].num + update >= 0) adjlist[iter->second].num += update;
        }
        else
        {
            uint64_t iter;
            for(iter=0;iter<adjlist.size();iter++)
            {
                if(adjlist[iter] == ae) break;
            }
            if(iter == adjlist.size())
            {
                adjlist.emplace_back(ae);
                iter = adjlist.size()-1;
            }
            current_size = adjlist[iter].num;
            if((int)adjlist[iter].num + update >= 0) adjlist[iter].num += update;
        }

        degree += (update > 0 || current_size>=(uint64_t)-update)*update;
        return current_size;
    }

    uint64_t get_edge_num(uint64_t vid, adjedge_type ae)
    {
        auto &degree = array_of_adjstruct[vid].degree;
        auto &adjlist = array_of_adjstruct[vid].adjlist;
        auto &adjlist_map = array_of_adjstruct[vid].adjlist_map;
        auto has_map = adjlist_map != nullptr;

        if(adjlist.size() >= need_index_threshold && has_map)
        {
            auto iter = adjlist_map->find(ae);
            if(iter == adjlist_map->end()) return 0;
            return adjlist[iter->second].num;
        }
        else
        {
            uint64_t iter;
            for(iter=0;iter<adjlist.size();iter++)
            {
                if(adjlist[iter] == ae) break;
            }
            if(iter == adjlist.size()) return 0;
            return adjlist[iter].num;
        }
    }

private:
    uint64_t vertices;
    adjedge_type empty_key;
    uint64_t need_index_threshold;
    std::vector<AdjStruct> array_of_adjstruct;
};

template <typename AdjEdgeType = AdjEdge<void>,
         template<typename> typename DataType = storage::data::Vector,
         template<typename, typename> typename IndexType = storage::index::DenseHashMap>
class IndexOnlyStorage
{
public:
    using adjedge_type = AdjEdgeType;
    using adjlist_type = DataType<adjedge_type>;
    using adjmap_type = IndexType<adjedge_type, uint16_t>;
    using lock_type = SpinLock;
    class adjlist_iter_type
    {
    public:
        adjlist_iter_type(typename adjmap_type::iterator _map_iter)
            :map_iter(_map_iter), list_iter(), is_map(true)
        {
        }
        adjlist_iter_type(typename adjlist_type::iterator _list_iter)
            :map_iter(), list_iter(_list_iter), is_map(false)
        {
        }
        adjedge_type operator*()
        {
            if(is_map)
            {
                adjedge_type a = map_iter->first;
                a.num = map_iter->second;
                return a;
            }
            else
            {
                return *list_iter;
            }
        }
        adjlist_iter_type operator++()
        {
            if(is_map) map_iter++; else list_iter++;
            return *this;
        }
        adjlist_iter_type operator++(int)
        {
            auto tmp = *this;
            if(is_map) map_iter++; else list_iter++;
            return tmp;
        }
        bool operator==(const adjlist_iter_type &a) const
        {
            if(is_map != a.is_map) return false;
            if(is_map)
                return map_iter == a.map_iter;
            else
                return list_iter == a.list_iter;
        }
        bool operator!=(const adjlist_iter_type &a) const
        {
            return !(*this == a);
        }
    private:
        typename adjmap_type::iterator map_iter;
        typename adjlist_type::iterator list_iter;
        bool is_map;
    };
    static constexpr bool edge_random_accessable = false;

    struct AdjStruct
    {
        adjlist_type adjlist;
        lock_type futex;
        std::shared_ptr<adjmap_type> adjlist_map;
        uint64_t degree;
    };

    IndexOnlyStorage(uint64_t _vertices = 0, uint64_t _need_index_threshold = 512)
        : vertices(_vertices), 
        empty_key(),
        need_index_threshold(_need_index_threshold),
        array_of_adjstruct(vertices)
    {
        empty_key.nbr = 281474976710655; // -1 in uint64_t:48
    }
    IndexOnlyStorage(const IndexOnlyStorage &) = default;
    IndexOnlyStorage(IndexOnlyStorage &&) = default;
    IndexOnlyStorage& operator=(const IndexOnlyStorage &) = default;
    IndexOnlyStorage& operator=(IndexOnlyStorage && a) = default;
    ~IndexOnlyStorage() = default;

    void resize(uint64_t _vertices)
    {
        vertices = _vertices;
        array_of_adjstruct.resize(_vertices);
    }

    lock_type& get_lock(uint64_t vid)
    {
        return array_of_adjstruct[vid].futex;
    }

    uint64_t get_degree(uint64_t vid) const
    {
        return array_of_adjstruct[vid].degree;
    }

    std::pair<adjlist_iter_type, adjlist_iter_type> get_adjlist_iter(uint64_t vid)
    {
        if(array_of_adjstruct[vid].adjlist_map != nullptr)
        {
            return {adjlist_iter_type(array_of_adjstruct[vid].adjlist_map->begin()), 
                    adjlist_iter_type(array_of_adjstruct[vid].adjlist_map->end())};
        }
        else
        {
            return {adjlist_iter_type(array_of_adjstruct[vid].adjlist.begin()), 
                    adjlist_iter_type(array_of_adjstruct[vid].adjlist.end())};
        }
    }

    adjlist_type& get_adjlist(uint64_t vid)
    {
        throw std::runtime_error("Storage is not edge_random_accessable.");
        return array_of_adjstruct[vid].adjlist;
    }

    uint64_t update_edge(adjedge_type ae, uint64_t vid, int update)
    {
        auto &adj_struct = array_of_adjstruct[vid];
        std::lock_guard<lock_type> lock(adj_struct.futex);

        auto &degree = adj_struct.degree;
        auto &adjlist = adj_struct.adjlist;
        auto &adjlist_map = adj_struct.adjlist_map;
        auto has_map = adjlist_map != nullptr;

        ae.num = 0;

        uint64_t current_size = 0;
        if(adjlist.size() >= need_index_threshold && !has_map)
        {
            adjlist_map = std::make_shared<adjmap_type>(adjlist.size());
            adjlist_map->set_empty_key(empty_key);
            for(uint64_t i=0;i<adjlist.size();i++)
            {
                adjlist_map->insert(std::make_pair(adjlist[i], (uint64_t)adjlist[i].num));
            }
            adjlist.clear();
            has_map = true;
        }

        if(has_map)
        {
            auto iter = adjlist_map->find(ae);
            if(iter == adjlist_map->end())
            {
                iter = adjlist_map->insert(std::make_pair(ae, 0)).first;
            }
            current_size = iter->second;
            if((int)iter->second + update >= 0) iter->second += update;
        }
        else
        {
            uint64_t iter;
            for(iter=0;iter<adjlist.size();iter++)
            {
                if(adjlist[iter] == ae) break;
            }
            if(iter == adjlist.size())
            {
                adjlist.emplace_back(ae);
                iter = adjlist.size()-1;
            }
            current_size = adjlist[iter].num;
            if((int)adjlist[iter].num + update >= 0) adjlist[iter].num += update;
        }

        degree += (update > 0 || current_size>=(uint64_t)-update)*update;
        return current_size;
    }

    uint64_t get_edge_num(uint64_t vid, adjedge_type ae)
    {
        auto &degree = array_of_adjstruct[vid].degree;
        auto &adjlist = array_of_adjstruct[vid].adjlist;
        auto &adjlist_map = array_of_adjstruct[vid].adjlist_map;
        auto has_map = adjlist_map != nullptr;

        if(has_map)
        {
            auto iter = adjlist_map->find(ae);
            if(iter == adjlist_map->end()) return 0;
            return iter->second;
        }
        else
        {
            uint64_t iter;
            for(iter=0;iter<adjlist.size();iter++)
            {
                if(adjlist[iter] == ae) break;
            }
            if(iter == adjlist.size()) return 0;
            return adjlist[iter].num;
        }
    }

private:
    uint64_t vertices;
    adjedge_type empty_key;
    uint64_t need_index_threshold;
    std::vector<AdjStruct> array_of_adjstruct;
};
