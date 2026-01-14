#ifndef UDPSENDER_H
#define UDPSENDER_H

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <string>
#include <atomic>
#include <cstdio>
#include <cstring>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/time.h>

#include "vec.h"
#include "quat.h"
#include "array.h"

class UDPsender {
public:
    // ========= UI / 外部可见状态 =========
    // std::atomic<bool> udp_enabled{false};
    bool udp_enabled = false;
    bool udp_initialized = false;

    // ========= 统计信息（可 UI 显示 / Debug） =========
    std::atomic<uint64_t> frames_sent{0};
    std::atomic<uint64_t> frames_dropped{0};

public:
    static uint32_t getsystemtime() {
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        return static_cast<uint32_t>(tv.tv_sec * 1000LL + tv.tv_usec / 1000);
    }

public:
    UDPsender() {
        start_send_thread();
    }

    ~UDPsender() {
        cleanup_udp();
    }

    bool init_udp(const char* ip, int port) {
        cleanup_socket_only();

        udp_sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        if (udp_sockfd < 0) {
            std::printf("UDP socket creation failed\n");
            return false;
        }

        std::memset(&udp_servaddr, 0, sizeof(udp_servaddr));
        udp_servaddr.sin_family = AF_INET;
        udp_servaddr.sin_port = htons(port);
        udp_servaddr.sin_addr.s_addr = inet_addr(ip);

        udp_initialized = true;
        std::printf("UDP initialized: %s:%d\n", ip, port);
        return true;
    }

    void cleanup_udp() {
        request_stop();
        if (udp_thread.joinable()) {
            udp_thread.join();
        }
        cleanup_socket_only();
    }

    // ========= 主线程调用接口（你原来就在用的） =========
    void send_frame_udp(const slice1d<vec3>& global_positions,
                        const slice1d<quat>& global_rotations,
                        uint32_t timestamp_ms = 0)
    {
        if (!udp_enabled || !udp_initialized) {
            return; // UI 语义：没开就不进队列
        }

        if (timestamp_ms == 0) {
            timestamp_ms = getsystemtime();
        }

        std::lock_guard<std::mutex> lock(queue_mutex);

        // 队列过长：直接丢最旧的（实时系统优先新帧）
        if (frame_queue.size() >= MAX_QUEUE_SIZE) {
            frame_queue.pop();
            frames_dropped++;
        }

        frame_queue.emplace(global_positions, global_rotations,
                            timestamp_ms);
        queue_cv.notify_one();
    }

private:
    // ========= 内部数据结构 =========
    struct FrameData {
        std::vector<vec3> global_positions;
        std::vector<quat> global_rotations;
        uint32_t timestamp_ms;

        FrameData(const slice1d<vec3>& p,
                  const slice1d<quat>& r,
                  uint32_t ts)
            : global_positions(p.data, p.data + p.size),
              global_rotations(r.data, r.data + r.size),
              timestamp_ms(ts) {}
    };

private:
    // ========= 队列 / 线程 =========
    static constexpr size_t MAX_QUEUE_SIZE = 3;

    std::queue<FrameData> frame_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::thread udp_thread;
    bool running = true;

    // ========= UDP =========
    int udp_sockfd = -1;
    sockaddr_in udp_servaddr{};

private:
    void cleanup_socket_only() {
        if (udp_initialized) {
            close(udp_sockfd);
            udp_sockfd = -1;
            udp_initialized = false;
        }
    }

    void request_stop() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            running = false;
        }
        queue_cv.notify_all();
    }

    void start_send_thread() {
        udp_thread = std::thread([this]() {
            while (true) {
                FrameData fd = wait_and_pop_frame();
                if (!running && fd.timestamp_ms == 0) {
                    break;
                }

                if (!udp_enabled || !udp_initialized || udp_sockfd < 0) {
                    frames_dropped++;
                    continue;
                }

                send_frame_udp_impl(fd);
                frames_sent++;
            }
        });
    }

    FrameData wait_and_pop_frame() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        queue_cv.wait(lock, [this] {
            return !frame_queue.empty() || !running;
        });

        if (!running && frame_queue.empty()) {
            return FrameData(slice1d<vec3>(0, nullptr),
                             slice1d<quat>(0, nullptr),
                             0);
        }

        FrameData fd = std::move(frame_queue.front());
        frame_queue.pop();
        return fd;
    }

    // ========= 真正的 UDP 发送 =========
    void send_frame_udp_impl(const FrameData& fd) {
        std::string json;
        json.reserve(1024);

        json += "{";
        json += "\"timestamp_ms\":" + std::to_string(fd.timestamp_ms) + ",";

        json += "\"global_positions\":[";
        for (size_t i = 1; i < fd.global_positions.size(); ++i) {
            const vec3& p = fd.global_positions[i];
            json += "[" + std::to_string(p.x) + "," +
                         std::to_string(p.y) + "," +
                         std::to_string(p.z) + "]";
            if (i + 1 != fd.global_positions.size()) json += ",";
        }

        json += "],\"global_rotations\":[";
        for (size_t i = 1; i < fd.global_rotations.size(); ++i) {
            const quat& q = fd.global_rotations[i];
            json += "[" + std::to_string(q.w) + "," +
                         std::to_string(q.x) + "," +
                         std::to_string(q.y) + "," +
                         std::to_string(q.z) + "]";
            if (i + 1 != fd.global_rotations.size()) json += ",";
        }
        
        json += "]}";

        sendto(udp_sockfd,
               json.data(),
               json.size(),
               0,
               (const sockaddr*)&udp_servaddr,
               sizeof(udp_servaddr));
    }
};

// 全局UDP发送器实例的外部声明
extern UDPsender udp_sender;

#endif