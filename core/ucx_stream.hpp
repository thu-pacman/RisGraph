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
#include <string>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>
#include <ucp/api/ucp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <netdb.h>
#include <immintrin.h>
#include <boost/fiber/all.hpp>

class UCXStream
{
public:
    UCXStream()
    {
        ucp_params_t ucp_params;
        ucp_worker_params_t worker_params;
        ucp_config_t *config;
        ucs_status_t status;
        size_t addr_len;

        memset(&ucp_params, 0, sizeof(ucp_params));
        memset(&worker_params, 0, sizeof(worker_params));

        status = ucp_config_read(nullptr, nullptr, &config);
        if(status != UCS_OK) throw std::runtime_error("ucp_config_read Error");

        ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES |
                                UCP_PARAM_FIELD_REQUEST_SIZE |
                                UCP_PARAM_FIELD_REQUEST_INIT;
        ucp_params.features = UCP_FEATURE_STREAM;
        ucp_params.request_size = sizeof(request_context);
        ucp_params.request_init = &request_init;

        status = ucp_init(&ucp_params, config, &ucp_context);
        if(status != UCS_OK) throw std::runtime_error("ucp_init Error");
        ucp_config_release(config);

        worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
        worker_params.thread_mode = UCS_THREAD_MODE_MULTI;

        status = ucp_worker_create(ucp_context, &worker_params, &ucp_worker);
        if(status != UCS_OK) throw std::runtime_error("ucp_worker_create Error");

        status = ucp_worker_get_address(ucp_worker, &local_ucx_addr, &addr_len);
        if(status != UCS_OK) throw std::runtime_error("ucp_worker_get_address Error");
        local_addr = std::string((char*)local_ucx_addr, addr_len);
    }

    ~UCXStream()
    {
        disconnect();
        ucp_worker_release_address(ucp_worker, local_ucx_addr);
        ucp_worker_destroy(ucp_worker);
        ucp_cleanup(ucp_context);
    }

    std::string addr() const
    {
        return local_addr;
    }

    void connect(std::string peer_addr, size_t num_eps = 1)
    {
        ucp_ep_params_t ep_params;
        ucs_status_t status;
        ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                               UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
        ep_params.address = (const ucp_address_t *)peer_addr.data();
        ep_params.err_mode = UCP_ERR_HANDLING_MODE_NONE;

        for(size_t i=0;i<num_eps;i++)
        {
            ucp_eps.emplace_back();
            status = ucp_ep_create(ucp_worker, &ep_params, &ucp_eps.back());
        }
        if(status != UCS_OK) throw std::runtime_error("ucp_ep_create Error");
    }

    void disconnect()
    {
        for(auto &ucp_ep : ucp_eps)
        {
            ucs_status_ptr_t request = ucp_ep_flush_nb(ucp_ep, 0, send_handler);

            if (UCS_PTR_IS_ERR(request))
            {
                throw std::runtime_error("ucp_stream_send_nb Error");
            }
            else if (UCS_PTR_IS_PTR(request))
            {
                wait(ucp_worker, request);
                ucp_request_free(request);
            }

            request = ucp_ep_close_nb(ucp_ep, UCP_EP_CLOSE_MODE_FLUSH);

            if (UCS_PTR_IS_ERR(request))
            {
                throw std::runtime_error("ucp_ep_close_nb Error");
            }
            else if (UCS_PTR_IS_PTR(request))
            {
                ucs_status_t status;
                do
                {
                   ucp_worker_progress(ucp_worker);
                   status = ucp_request_check_status(request);
                } while (status == UCS_INPROGRESS);
                ucp_request_free(request);
            }
        }

        ucp_eps.clear();
    }

    void send(const void* data, size_t length, size_t ep_idx = 0)
    {
        auto &ucp_ep = ucp_eps.at(ep_idx);

        ucs_status_ptr_t request = ucp_stream_send_nb(ucp_ep, data, length, ucp_dt_make_contig(1), send_handler, 0);

        if (UCS_PTR_IS_ERR(request))
        {
            throw std::runtime_error("ucp_stream_send_nb Error");
        }
        else if (UCS_PTR_IS_PTR(request))
        {
            wait(ucp_worker, request);
            ucp_request_free(request);
        }
    }

    void recv(void* data, size_t length, size_t ep_idx = 0)
    {
        auto &ucp_ep = ucp_eps.at(ep_idx);

        size_t recv_length;
        ucs_status_ptr_t request = ucp_stream_recv_nb(ucp_ep, data, length, ucp_dt_make_contig(1), recv_handler, &recv_length, UCP_STREAM_RECV_FLAG_WAITALL);

        if (UCS_PTR_IS_ERR(request))
        {
            throw std::runtime_error("ucp_stream_recv_nb Error");
        }
        else if (UCS_PTR_IS_PTR(request))
        {
            wait(ucp_worker, request);
            ucp_request_free(request);
        }
    }

    static std::vector<UCXStream> make_ucx_stream(std::string server_name, uint16_t server_port, size_t num_streams, size_t num_eps = 1)
    {
        std::vector<UCXStream> streams(num_streams);
        if(!server_name.empty())
        {
            int sock = client_connect(server_name.c_str(), server_port);
            if(sock < 0) throw std::runtime_error("client_connect Error");

            size_t dst_num_streams;
            {
                auto ret = ::recv(sock, &dst_num_streams, sizeof(dst_num_streams), MSG_WAITALL);
                if(ret != sizeof(dst_num_streams)) throw std::runtime_error("recv error");
            }
            size_t dst_num_eps;
            {
                auto ret = ::recv(sock, &dst_num_eps, sizeof(dst_num_eps), MSG_WAITALL);
                if(ret != sizeof(dst_num_eps)) throw std::runtime_error("recv error");
            }
            if(dst_num_eps*dst_num_streams != num_streams*num_eps || (dst_num_eps%num_eps && num_eps%dst_num_eps)) throw std::runtime_error("num_clients error");

            {
                auto ret = ::send(sock, &num_streams, sizeof(num_streams), 0);
                if(ret != sizeof(num_streams)) throw std::runtime_error("esend error");
            }
            {
                auto ret = ::send(sock, &num_eps, sizeof(num_eps), 0);
                if(ret != sizeof(num_eps)) throw std::runtime_error("esend error");
            }

            size_t cur_ep_id = 0;
            size_t cur_stream_id = 0;
            for(size_t i=0;i<dst_num_streams;i++)
            {
                size_t size;
                {
                    auto ret = ::recv(sock, &size, sizeof(size), MSG_WAITALL);
                    if(ret != sizeof(size)) throw std::runtime_error("recv error");
                }
                char* addr = (char*)malloc(size);
                {
                    auto ret = ::recv(sock, addr, size, MSG_WAITALL);
                    if((size_t)ret != size) throw std::runtime_error("recv error");
                }
                if(dst_num_eps >= num_eps)
                {
                    for(size_t j=0;j<dst_num_eps/num_eps;j++)
                    {
                        streams[cur_stream_id++].connect(std::string(addr, size), num_eps);
                    }
                }
                else
                {
                    streams[cur_stream_id].connect(std::string(addr, size), dst_num_eps);
                    cur_ep_id += dst_num_eps;
                    if(cur_ep_id == num_eps)
                    {
                        cur_stream_id++;
                        cur_ep_id = 0;
                    }
                }
                free(addr);
            }
            for(size_t i=0;i<num_streams;i++)
            {
                {
                    size_t size = streams[i].addr().size();
                    auto ret = ::send(sock, &size, sizeof(size), 0);
                    if(ret != sizeof(size)) throw std::runtime_error("send error");
                }
                {
                    auto ret = ::send(sock, streams[i].addr().data(), streams[i].addr().size(), 0);
                    if((size_t)ret != streams[i].addr().size()) throw std::runtime_error("send error");
                }

            }
            close(sock);
        } 
        else 
        {
            int sock = server_connect(server_port);
            if(sock < 0) throw std::runtime_error("server_connect Error");

            {
                auto ret = ::send(sock, &num_streams, sizeof(num_streams), 0);
                if(ret != sizeof(num_streams)) throw std::runtime_error("esend error");
            }
            {
                auto ret = ::send(sock, &num_eps, sizeof(num_eps), 0);
                if(ret != sizeof(num_eps)) throw std::runtime_error("esend error");
            }

            size_t dst_num_streams;
            {
                auto ret = ::recv(sock, &dst_num_streams, sizeof(dst_num_streams), MSG_WAITALL);
                if(ret != sizeof(dst_num_streams)) throw std::runtime_error("recv error");
            }
            size_t dst_num_eps;
            {
                auto ret = ::recv(sock, &dst_num_eps, sizeof(dst_num_eps), MSG_WAITALL);
                if(ret != sizeof(dst_num_eps)) throw std::runtime_error("recv error");
            }
            if(dst_num_eps*dst_num_streams != num_streams*num_eps || (dst_num_eps%num_eps && num_eps%dst_num_eps)) throw std::runtime_error("num_clients error");

            for(size_t i=0;i<num_streams;i++)
            {
                {
                    size_t size = streams[i].addr().size();
                    auto ret = ::send(sock, &size, sizeof(size), 0);
                    if(ret != sizeof(size)) throw std::runtime_error("send error");
                }
                {
                    auto ret = ::send(sock, streams[i].addr().data(), streams[i].addr().size(), 0);
                    if((size_t)ret != streams[i].addr().size()) throw std::runtime_error("send error");
                }

            }

            size_t cur_ep_id = 0;
            size_t cur_stream_id = 0;
            for(size_t i=0;i<dst_num_streams;i++)
            {
                size_t size;
                {
                    auto ret = ::recv(sock, &size, sizeof(size), MSG_WAITALL);
                    if(ret != sizeof(size)) throw std::runtime_error("recv error");
                }
                char* addr = (char*)malloc(size);
                {
                    auto ret = ::recv(sock, addr, size, MSG_WAITALL);
                    if((size_t)ret != size) throw std::runtime_error("recv error");
                }
                if(dst_num_eps >= num_eps)
                {
                    for(size_t j=0;j<dst_num_eps/num_eps;j++)
                    {
                        streams[cur_stream_id++].connect(std::string(addr, size), num_eps);
                    }
                }
                else
                {
                    streams[cur_stream_id].connect(std::string(addr, size), dst_num_eps);
                    cur_ep_id += dst_num_eps;
                    if(cur_ep_id == num_eps)
                    {
                        cur_stream_id++;
                        cur_ep_id = 0;
                    }
                }
                free(addr);
            }
            close(sock);
        }

        fprintf(stderr, "Made %lu UCXStreams...\n", num_streams);

        return streams;
    }

private:
    ucp_context_h ucp_context;
    ucp_worker_h ucp_worker;
    std::vector<ucp_ep_h> ucp_eps;
    ucp_address_t *local_ucx_addr;
    std::string local_addr;
    bool is_connected;

    struct request_context
    {
        int completed;
    };

    static void request_init(void *request)
    {
        ((request_context *)request)->completed = 0;
    }

    static void send_handler(void *request, ucs_status_t status)
    {
        ((request_context *)request)->completed = 1;
    }

    static void recv_handler(void *request, ucs_status_t status, size_t length)
    {
        ((request_context *)request)->completed = 1;
    }

    static void wait(ucp_worker_h ucp_worker, void *request)
    {
        size_t spin = 0;
        while (((request_context *)request)->completed == 0)
        {
            spin++;
            if(ucp_worker_progress(ucp_worker)) continue;
            //ucs_status_t status = ucp_worker_wait(ucp_worker);
            //if(status != UCS_OK) throw std::runtime_error("ucp_worker_wait Error");
            //if(spin > 1024*1024)
            //    std::this_thread::sleep_for(std::chrono::nanoseconds(100));
            //else if(spin > 1024)
            //    std::this_thread::yield();
            //else
            //    boost::this_fiber::yield();
            if(spin > 1024) std::this_thread::yield();
            boost::this_fiber::yield();
        }
        ((request_context *)request)->completed = 0; // need to init, maybe reuse by ucx :(
    }

    static int client_connect(const char *server, uint16_t server_port)
    {
        sockaddr_in conn_addr;
        hostent *he;
        int connfd;
        int ret;

        connfd = socket(AF_INET, SOCK_STREAM, 0);
        if(connfd < 0) throw std::runtime_error("open client socket Error");

        he = gethostbyname(server);
        if(he == nullptr || he->h_addr_list == nullptr) throw std::runtime_error("client gethostbyname Error");

        conn_addr.sin_family = he->h_addrtype;
        conn_addr.sin_port = htons(server_port);

        memcpy(&conn_addr.sin_addr, he->h_addr_list[0], he->h_length);
        memset(conn_addr.sin_zero, 0, sizeof(conn_addr.sin_zero));

        ret = ::connect(connfd, (sockaddr*)&conn_addr, sizeof(conn_addr));
        if(ret < 0) throw std::runtime_error("client connect Error");

        return connfd;
    }

    static int server_connect(uint16_t server_port)
    {
        sockaddr_in inaddr;
        int lsock  = -1;
        int dsock  = -1;
        int optval = 1;
        int ret;

        lsock = socket(AF_INET, SOCK_STREAM, 0);
        if(lsock < 0) throw std::runtime_error("open server socket Error");

        optval = 1;
        ret = setsockopt(lsock, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));
        if(ret < 0) throw std::runtime_error("server setsockopt Error");

        inaddr.sin_family = AF_INET;
        inaddr.sin_port = htons(server_port);
        inaddr.sin_addr.s_addr = INADDR_ANY;
        memset(inaddr.sin_zero, 0, sizeof(inaddr.sin_zero));
        ret = bind(lsock, (struct sockaddr*)&inaddr, sizeof(inaddr));
        if(ret < 0) throw std::runtime_error("server bind Error");

        ret = listen(lsock, 0);
        if(ret < 0) throw std::runtime_error("server listen Error");

        fprintf(stderr, "Waiting for connection...\n");

        dsock = accept(lsock, nullptr, nullptr);
        if(ret < 0) throw std::runtime_error("server accept Error");

        close(lsock);
        return dsock;
    }


};
