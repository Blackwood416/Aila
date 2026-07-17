#include "proxy/ProxyLogging.hpp"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

namespace aila::proxy::logging {
namespace {

class Dispatcher {
public:
    Dispatcher() : thread_([this] { run(); }) {}

    ~Dispatcher() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            queue_.clear();
            callback_ = nullptr;
            user_data_ = nullptr;
        }
        condition_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    void set_callback(AilaLogCallback callback, void* user_data) noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            callback_ = callback;
            user_data_ = callback ? user_data : nullptr;
            if (!callback) queue_.clear();
        } catch (...) {}
    }

    void set_level(int level) noexcept {
        level_.store(level, std::memory_order_release);
    }

    int level() const noexcept { return level_.load(std::memory_order_acquire); }

    void enqueue(const Source& source, int message_level, std::string_view message) noexcept {
        if (!source || !source->active.load(std::memory_order_acquire) ||
            message_level < level()) {
            return;
        }
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_ || !callback_ || queue_.size() >= kCapacity) return;
            queue_.push_back({source, message_level, std::string(message)});
            condition_.notify_one();
        } catch (...) {}
    }

private:
    struct Entry {
        std::weak_ptr<SourceState> source;
        int level = 0;
        std::string message;
    };

    void run() noexcept {
        for (;;) {
            Entry entry;
            AilaLogCallback callback = nullptr;
            void* user_data = nullptr;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
                if (stopping_) return;
                entry = std::move(queue_.front());
                queue_.pop_front();
                callback = callback_;
                user_data = user_data_;
            }
            const auto source = entry.source.lock();
            if (!source || !source->active.load(std::memory_order_acquire) || !callback ||
                entry.level < level()) {
                continue;
            }
            try { callback(entry.level, entry.message.c_str(), user_data); } catch (...) {}
        }
    }

    static constexpr size_t kCapacity = 256;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<Entry> queue_;
    AilaLogCallback callback_ = nullptr;
    void* user_data_ = nullptr;
    std::atomic_int level_{1};
    bool stopping_ = false;
    std::thread thread_;
};

Dispatcher& dispatcher() {
    static Dispatcher value;
    return value;
}

} // namespace

Source create_source() { return std::make_shared<SourceState>(); }

void deactivate(const Source& source) noexcept {
    if (source) source->active.store(false, std::memory_order_release);
}

void set_callback(AilaLogCallback callback, void* user_data) noexcept {
    dispatcher().set_callback(callback, user_data);
}

void set_level(int value) noexcept { dispatcher().set_level(value); }
int level() noexcept { return dispatcher().level(); }

void enqueue(const Source& source, int message_level, std::string_view message) noexcept {
    dispatcher().enqueue(source, message_level, message);
}

} // namespace aila::proxy::logging
