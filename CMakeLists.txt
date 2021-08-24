cmake_minimum_required(VERSION 3.1)
project(RisGraph CXX)

set (CMAKE_CXX_STANDARD 17)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)
link_libraries(Threads::Threads)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIRS})
link_directories(${TBB_LIBRARY_DIRS})
link_libraries(${TBB_LIBRARIES})

find_package(AIO REQUIRED)
include_directories(${AIO_INCLUDE_DIRS})
link_directories(${AIO_LIBRARY_DIRS})
link_libraries(${AIO_LIBRARIES})

set(Boost_USE_STATIC_LIBS        ON)  # only find static libs
set(Boost_USE_DEBUG_LIBS         OFF) # ignore debug libs and
set(Boost_USE_RELEASE_LIBS       ON)  # only find release libs
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME     OFF)
find_package(Boost REQUIRED COMPONENTS fiber)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
link_libraries(${Boost_LIBRARIES})

find_package(PkgConfig)
pkg_check_modules(UCX REQUIRED ucx)
include_directories(${UCX_INCLUDE_DIRS})
link_directories(${UCX_LIBRARY_DIRS})
link_libraries(${UCX_LIBRARIES})

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no march native support.")
endif()

try_compile(ARCH_HAS_RTM "${CMAKE_BINARY_DIR}/temp" "${CMAKE_SOURCE_DIR}/cmake/rtm_test.cpp")
if(ARCH_HAS_RTM)
    message(STATUS "Your CPU supports rtm.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DARCH_HAS_RTM")
else()
    message(STATUS "Your CPU doesn't support rtm, fallback to locks.")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-Ofast -g -DNDEBUG")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(PROJECT_DEPS_DIR ${PROJECT_SOURCE_DIR}/deps)
include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_DEPS_DIR})

add_subdirectory(${PROJECT_DEPS_DIR}/abseil-cpp)
link_libraries(absl::flat_hash_map absl::node_hash_map)

option(ENABLE_MMAP "Enable default mmap pool" OFF)
if (ENABLE_MMAP)
    add_definitions("-DENABLE_MMAP")
endif()
add_library(mmapalloc STATIC core/mmapalloc.cpp)
link_libraries(mmapalloc)

option(ENABLE_KVS "Enable Key Value Store" OFF)
if (ENABLE_KVS)
    add_definitions("-DENABLE_KVS")
endif()

option(ENABLE_WAL "Enable Write-Ahead Log" OFF)
if (ENABLE_WAL)
    add_definitions("-DENABLE_WAL")
endif()

add_executable(wcc src/wcc.cpp)
add_executable(wcc_inc src/wcc_inc.cpp)
add_executable(wcc_inc_batch src/wcc_inc_batch.cpp)
add_executable(wcc_inc_clf src/wcc_inc_clf.cpp)
add_executable(wcc_inc_rt src/wcc_inc_rt.cpp)

add_executable(bfs src/bfs.cpp)
add_executable(bfs_inc src/bfs_inc.cpp)
add_executable(bfs_inc_batch src/bfs_inc_batch.cpp)
add_executable(bfs_inc_clf src/bfs_inc_clf.cpp)
add_executable(bfs_inc_rt src/bfs_inc_rt.cpp)

add_executable(sssp src/sssp.cpp)
add_executable(sssp_inc src/sssp_inc.cpp)
add_executable(sssp_inc_batch src/sssp_inc_batch.cpp)
add_executable(sssp_inc_clf src/sssp_inc_clf.cpp)
add_executable(sssp_inc_rt src/sssp_inc_rt.cpp)

add_executable(sswp src/sswp.cpp)
add_executable(sswp_inc src/sswp_inc.cpp)
add_executable(sswp_inc_clf src/sswp_inc_clf.cpp)
add_executable(sswp_inc_rt src/sswp_inc_rt.cpp)

add_executable(pagerank src/pagerank.cpp)
add_executable(pagerank_inc src/pagerank_inc.cpp)

add_executable(bfs_inc_rpc src/all_inc_rpc.cpp)
add_executable(wcc_inc_rpc src/all_inc_rpc.cpp)
add_executable(sssp_inc_rpc src/all_inc_rpc.cpp)
add_executable(sswp_inc_rpc src/all_inc_rpc.cpp)
target_compile_definitions(bfs_inc_rpc PRIVATE -DBFS)
target_compile_definitions(wcc_inc_rpc PRIVATE -DWCC)
target_compile_definitions(sssp_inc_rpc PRIVATE -DSSSP)
target_compile_definitions(sswp_inc_rpc PRIVATE -DSSWP)

add_executable(rpc_client src/rpc_client.cpp)
add_executable(rpc_kvs src/rpc_kvs.cpp)

add_executable(bfs_inc_rpc_txn src/all_inc_rpc_txn.cpp)
add_executable(wcc_inc_rpc_txn src/all_inc_rpc_txn.cpp)
add_executable(sssp_inc_rpc_txn src/all_inc_rpc_txn.cpp)
add_executable(sswp_inc_rpc_txn src/all_inc_rpc_txn.cpp)
target_compile_definitions(bfs_inc_rpc_txn PRIVATE -DBFS)
target_compile_definitions(wcc_inc_rpc_txn PRIVATE -DWCC)
target_compile_definitions(sssp_inc_rpc_txn PRIVATE -DSSSP)
target_compile_definitions(sswp_inc_rpc_txn PRIVATE -DSSWP)

add_executable(rpc_txn_client src/rpc_txn_client.cpp)

add_executable(btree_bench src/btree_bench.cpp)
add_executable(ucx_stream_bench src/ucx_stream_bench.cpp)

add_executable(bfs_inc_rate src/all_inc.cpp)
add_executable(wcc_inc_rate src/all_inc.cpp)
add_executable(sssp_inc_rate src/all_inc.cpp)
add_executable(sswp_inc_rate src/all_inc.cpp)
target_compile_definitions(bfs_inc_rate PRIVATE -DBFS)
target_compile_definitions(wcc_inc_rate PRIVATE -DWCC)
target_compile_definitions(sssp_inc_rate PRIVATE -DSSSP)
target_compile_definitions(sswp_inc_rate PRIVATE -DSSWP)

add_executable(snbq14_inc_rpc src/snbq14_inc_rpc.cpp)
add_executable(rpc_client_snbq14 src/rpc_client_snbq14.cpp)
