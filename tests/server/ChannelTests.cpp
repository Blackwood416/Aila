#ifndef _WIN32
#error These tests exercise the pinned Windows Trantor poller.
#endif

// Winsock must precede headers that might include Windows.h.
#include <winsock2.h>
#include <ws2tcpip.h>

#include <trantor/net/Channel.h>
#include <trantor/net/EventLoop.h>

#include <chrono>
#include <condition_variable>
#include <exception>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

void Expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

[[noreturn]] void SocketError(const char* operation, int error = WSAGetLastError()) {
    throw std::runtime_error(std::string(operation) + " failed: Winsock " + std::to_string(error));
}

class Winsock final {
public:
    Winsock() {
        WSADATA data{};
        const int error = WSAStartup(MAKEWORD(2, 2), &data);
        if (error != 0) SocketError("WSAStartup", error);
        if (LOBYTE(data.wVersion) != 2 || HIBYTE(data.wVersion) != 2) {
            WSACleanup();
            throw std::runtime_error("Winsock 2.2 is required");
        }
    }
    ~Winsock() { WSACleanup(); }
    Winsock(const Winsock&) = delete;
    Winsock& operator=(const Winsock&) = delete;
};

class Socket final {
public:
    explicit Socket(SOCKET value = INVALID_SOCKET) : value_(value) {}
    ~Socket() { if (value_ != INVALID_SOCKET) closesocket(value_); }
    Socket(Socket&& other) noexcept : value_(std::exchange(other.value_, INVALID_SOCKET)) {}
    Socket& operator=(Socket&& other) noexcept {
        if (this != &other) {
            if (value_ != INVALID_SOCKET) closesocket(value_);
            value_ = std::exchange(other.value_, INVALID_SOCKET);
        }
        return *this;
    }
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    SOCKET Get() const { return value_; }
private:
    SOCKET value_;
};

void Nonblocking(SOCKET socket) {
    u_long value = 1;
    if (ioctlsocket(socket, FIONBIO, &value) == SOCKET_ERROR) SocketError("ioctlsocket(FIONBIO)");
}

Socket NewSocket() {
    Socket socket(::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
    if (socket.Get() == INVALID_SOCKET) SocketError("socket");
    Nonblocking(socket.Get());
    return socket;
}

void WaitReady(SOCKET socket, bool writing, Clock::time_point deadline) {
    for (;;) {
        const auto remaining = std::chrono::duration_cast<std::chrono::microseconds>(deadline - Clock::now());
        Expect(remaining.count() > 0, writing ? "Timed out waiting for socket writability" : "Timed out waiting for socket readability");
        timeval timeout{};
        timeout.tv_sec = static_cast<long>(remaining.count() / 1000000);
        timeout.tv_usec = static_cast<long>(remaining.count() % 1000000);
        fd_set ready{};
        fd_set errors{};
        FD_ZERO(&ready);
        FD_ZERO(&errors);
        FD_SET(socket, &ready);
        FD_SET(socket, &errors);
        const int count = ::select(0, writing ? nullptr : &ready, writing ? &ready : nullptr, &errors, &timeout);
        if (count == SOCKET_ERROR) {
            const int error = WSAGetLastError();
            if (error == WSAEINTR) continue;
            SocketError("select", error);
        }
        Expect(count != 0, writing ? "Socket did not become writable before the deadline" : "Socket did not become readable before the deadline");
        if (FD_ISSET(socket, &errors)) {
            int error = 0;
            int size = sizeof(error);
            if (getsockopt(socket, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&error), &size) == SOCKET_ERROR) {
                SocketError("getsockopt(SO_ERROR)");
            }
            SocketError("Socket exception readiness", error);
        }
        if (FD_ISSET(socket, &ready)) return;
    }
}

struct SocketPair {
    Socket sender;
    Socket receiver;
};

SocketPair ConnectLoopback(Clock::time_point deadline) {
    Socket listener = NewSocket();
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (::bind(listener.Get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == SOCKET_ERROR) SocketError("bind(loopback)");
    if (::listen(listener.Get(), 1) == SOCKET_ERROR) SocketError("listen");
    int addressSize = sizeof(address);
    if (::getsockname(listener.Get(), reinterpret_cast<sockaddr*>(&address), &addressSize) == SOCKET_ERROR) SocketError("getsockname");

    Socket sender = NewSocket();
    if (::connect(sender.Get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == SOCKET_ERROR) {
        const int error = WSAGetLastError();
        if (error != WSAEWOULDBLOCK && error != WSAEINPROGRESS && error != WSAEALREADY) SocketError("connect(loopback)", error);
    }
    WaitReady(sender.Get(), true, deadline);
    int connectError = 0;
    int errorSize = sizeof(connectError);
    if (::getsockopt(sender.Get(), SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&connectError), &errorSize) == SOCKET_ERROR) {
        SocketError("getsockopt(connect)");
    }
    if (connectError != 0) SocketError("connect completion", connectError);
    WaitReady(listener.Get(), false, deadline);
    Socket receiver(::accept(listener.Get(), nullptr, nullptr));
    if (receiver.Get() == INVALID_SOCKET) SocketError("accept");
    Nonblocking(receiver.Get());
    return {std::move(sender), std::move(receiver)};
}

int ChannelFd(SOCKET socket) {
    // Trantor's public Windows Channel API represents SOCKET as int.
    Expect(socket <= static_cast<SOCKET>((std::numeric_limits<int>::max)()), "Socket handle cannot be represented by Channel::fd()");
    return static_cast<int>(socket);
}

class RegisteredChannel final {
public:
    RegisteredChannel(trantor::EventLoop& loop, SOCKET socket) : channel(&loop, ChannelFd(socket)) {}
    ~RegisteredChannel() {
        if (!registered_) return;
        // disableAll() on an already disabled EpollPoller channel would add a
        // new zero-interest registration. Remove the existing one directly.
        if (!channel.isNoneEvent()) channel.disableAll();
        channel.remove();
    }
    void EnableReading() {
        channel.enableReading();
        registered_ = true;
    }
    RegisteredChannel(const RegisteredChannel&) = delete;
    RegisteredChannel& operator=(const RegisteredChannel&) = delete;
    trantor::Channel channel;
private:
    bool registered_ = false;
};

// This deadline does not depend on a Channel callback or an EventLoop timer
// firing. quit() wakes the poller through Trantor's public, thread-safe API.
class LoopWatchdog final {
public:
    explicit LoopWatchdog(trantor::EventLoop& loop) : loop_(loop), thread_([this] {
        std::unique_lock<std::mutex> lock(mutex_);
        if (changed_.wait_for(lock, 2s, [&] { return stopped_; })) return;
        expired_ = true;
        lock.unlock();
        loop_.quit();
    }) {}
    ~LoopWatchdog() { Stop(); }
    bool Stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        changed_.notify_all();
        if (thread_.joinable()) thread_.join();
        return expired_;
    }
    LoopWatchdog(const LoopWatchdog&) = delete;
    LoopWatchdog& operator=(const LoopWatchdog&) = delete;
private:
    trantor::EventLoop& loop_;
    std::mutex mutex_;
    std::condition_variable changed_;
    bool stopped_ = false;
    bool expired_ = false;
    std::thread thread_;
};

enum class ReadAction { DisableAll, DisableWriting, KeepWriting, ReenableWriting };

struct Observation {
    int readCalls = 0;
    int writeCalls = 0;
    int readRevents = 0;
    int writeRevents = 0;
    bool combinedReadiness = false;
    bool disableObserved = false;
    bool subscribedAfterRead = false;
    bool finishedBatch = false;
    bool quitQueued = false;
    std::string order;
    std::string error;
};

std::string Describe(const Observation& observed) {
    std::ostringstream text;
    text << "read_revents=0x" << std::hex << observed.readRevents
         << " write_revents=0x" << observed.writeRevents << std::dec
         << " read=" << observed.readCalls << " write=" << observed.writeCalls
         << " order=" << observed.order;
    return text.str();
}

std::string RunCase(ReadAction action) {
    const auto setupDeadline = Clock::now() + 2s;
    auto sockets = ConnectLoopback(setupDeadline);
    trantor::EventLoop loop;
    RegisteredChannel registration(loop, sockets.receiver.Get());
    auto& channel = registration.channel;
    Observation observed;
    const auto finishBatch = [&] {
        if (observed.quitQueued) return;
        observed.quitQueued = true;
        loop.queueInLoop([&] {
            observed.order += 'q';
            observed.finishedBatch = true;
            loop.quit();
        });
    };
    channel.setReadCallback([&] {
        observed.order += 'r';
        ++observed.readCalls;
        observed.readRevents = channel.revents();
        observed.combinedReadiness = (observed.readRevents & trantor::Channel::kReadEvent) != 0 &&
                                     (observed.readRevents & trantor::Channel::kWriteEvent) != 0;
        char byte = 0;
        const int received = ::recv(sockets.receiver.Get(), &byte, 1, 0);
        if (received != 1 || byte != 'P') observed.error = "Read callback did not consume the preloaded loopback byte";
        switch (action) {
        case ReadAction::DisableAll:
            channel.disableAll();
            observed.disableObserved = channel.isNoneEvent();
            break;
        case ReadAction::DisableWriting:
            channel.disableWriting();
            observed.disableObserved = !channel.isWriting() && channel.isReading();
            break;
        case ReadAction::KeepWriting:
            break;
        case ReadAction::ReenableWriting:
            channel.disableWriting();
            observed.disableObserved = !channel.isWriting() && channel.isReading();
            channel.enableWriting();
            break;
        }
        observed.subscribedAfterRead = channel.isWriting();
        // Do not quit from read itself: let Channel finish dispatching this
        // exact poller batch so a stale write callback remains observable.
        finishBatch();
    });
    channel.setWriteCallback([&] {
        observed.order += 'w';
        ++observed.writeCalls;
        observed.writeRevents = channel.revents();
        finishBatch();
    });
    channel.setCloseCallback([&] {
        observed.error = "Unexpected socket close readiness";
        finishBatch();
    });
    channel.setErrorCallback([&] {
        observed.error = "Unexpected socket error readiness";
        finishBatch();
    });

    registration.EnableReading();
    const char probe = 'P';
    if (::send(sockets.sender.Get(), &probe, 1, 0) != 1) SocketError("send(probe)");
    WaitReady(sockets.receiver.Get(), false, setupDeadline);
    WaitReady(sockets.receiver.Get(), true, setupDeadline);
    char peek = 0;
    if (::recv(sockets.receiver.Get(), &peek, 1, MSG_PEEK) != 1) SocketError("recv(MSG_PEEK)");
    Expect(peek == probe, "Socket readiness was not backed by the expected unread byte");
    // Enable writes only after input is already buffered. Both subscriptions
    // are active before the first poll; no manufactured revents or private
    // Channel dispatch API is used anywhere in this test.
    channel.enableWriting();
    Expect(channel.isReading() && channel.isWriting(), "Channel did not subscribe to both read and write readiness");

    LoopWatchdog watchdog(loop);
    loop.loop();
    const bool timedOut = watchdog.Stop();
    const std::string evidence = Describe(observed);
    Expect(!timedOut, "Independent event-loop timeout expired; " + evidence);
    Expect(observed.error.empty(), observed.error + "; " + evidence);
    Expect(observed.finishedBatch, "The post-batch callback did not run; " + evidence);
    Expect(observed.readCalls == 1 && observed.combinedReadiness,
           "The test did not observe actual read+write readiness in one poller batch; " + evidence);
    const bool expectWrite = action == ReadAction::KeepWriting || action == ReadAction::ReenableWriting;
    Expect(observed.subscribedAfterRead == expectWrite, "Read callback left the wrong subscription state; " + evidence);
    if (action != ReadAction::KeepWriting) Expect(observed.disableObserved, "Read callback did not disable the intended subscription; " + evidence);
    Expect(observed.writeCalls == (expectWrite ? 1 : 0),
           expectWrite ? "An enabled write callback was lost; " + evidence : "Write callback ran after read disabled it; " + evidence);
    Expect(observed.order == (expectWrite ? "rwq" : "rq"), "Callbacks were not completed in one read-then-write batch; " + evidence);
    if (expectWrite) Expect(observed.writeRevents == observed.readRevents, "Read and write callbacks observed different event batches; " + evidence);
    return evidence;
}

std::string RunOnOwnedThread(ReadAction action) {
    std::string result;
    std::exception_ptr failure;
    // A fresh owner thread also isolates EventLoop's thread-local registration
    // if socket setup fails before loop() has a chance to reset it on exit.
    std::thread thread([&] {
        try { result = RunCase(action); }
        catch (...) { failure = std::current_exception(); }
    });
    thread.join();
    if (failure) std::rethrow_exception(failure);
    return result;
}
} // namespace

int main() {
    try {
        Winsock winsock;
        const std::vector<std::pair<const char*, ReadAction>> tests = {
            {"read disables all subscriptions", ReadAction::DisableAll},
            {"read disables only writing", ReadAction::DisableWriting},
            {"read keeps writing subscribed", ReadAction::KeepWriting},
            {"read disables and reenables writing", ReadAction::ReenableWriting}
        };
        size_t failures = 0;
        for (const auto& [name, action] : tests) {
            try {
                const auto evidence = RunOnOwnedThread(action);
                std::cout << "[PASS] " << name << " (" << evidence << ")\n";
            } catch (const std::exception& error) {
                ++failures;
                std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
            }
        }
        std::cout << "Channel dispatch: " << tests.size() - failures << '/' << tests.size() << " groups passed\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Channel test setup failed: " << error.what() << '\n';
        return 1;
    }
}
