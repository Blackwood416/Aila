#include "proxy/ProxyLogging.hpp"

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <string>
#include <thread>

namespace aila::proxy::logging {

struct SourceState {
    std::mutex mutex;
    std::condition_variable condition;
    bool active = true;
    size_t callbacks_in_flight = 0;
};

namespace {

thread_local SourceState* current_callback_source = nullptr;

bool acquire_source(const Source& source) noexcept {
    if (!source) return false;
    try {
        std::lock_guard<std::mutex> lock(source->mutex);
        if (!source->active) return false;
        ++source->callbacks_in_flight;
        return true;
    } catch (...) {
        return false;
    }
}

void release_source(const Source& source) noexcept {
    try {
        std::lock_guard<std::mutex> lock(source->mutex);
        if (source->callbacks_in_flight != 0) --source->callbacks_in_flight;
        source->condition.notify_all();
    } catch (...) {}
}

void default_log_output(std::string_view message) noexcept {
    try {
        const HANDLE handle = GetStdHandle(STD_ERROR_HANDLE);
        if (handle == nullptr || handle == INVALID_HANDLE_VALUE) return;
        auto write = [&](const char* data, size_t size) {
            while (size != 0) {
                const DWORD chunk = static_cast<DWORD>((std::min)(
                    size, static_cast<size_t>((std::numeric_limits<DWORD>::max)())));
                DWORD written = 0;
                if (WriteFile(handle, data, chunk, &written, nullptr) == FALSE || written == 0) {
                    return false;
                }
                data += written;
                size -= written;
            }
            return true;
        };
        if (!write(message.data(), message.size())) return;
        constexpr char newline = '\n';
        (void)write(&newline, 1);
    } catch (...) {}
}

class Dispatcher {
public:
    Dispatcher() : thread_([this] { run(); }) {}

    ~Dispatcher() {
        set_callback(nullptr, nullptr);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            queue_.clear();
        }
        condition_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    void set_callback(AilaLogCallback callback, void* user_data) noexcept {
        try {
            std::unique_lock<std::mutex> lock(mutex_);
            callback_ = callback;
            user_data_ = callback ? user_data : nullptr;
            const bool called_from_dispatch = std::this_thread::get_id() == thread_.get_id();
            if (!called_from_dispatch) {
                callback_condition_.wait(lock, [&] { return callbacks_in_flight_ == 0; });
            }
        } catch (...) {}
    }

    void set_level(int level) noexcept {
        level_.store(level, std::memory_order_release);
    }

    int level() const noexcept { return level_.load(std::memory_order_acquire); }

    void enqueue(const Source& source, int message_level, std::string_view message) noexcept {
        if (!source || message_level < level()) return;
        try {
            {
                std::lock_guard<std::mutex> source_lock(source->mutex);
                if (!source->active) return;
            }
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_ || queue_.size() >= kCapacity) return;
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
            }
            const auto source = entry.source.lock();
            if (!source || entry.level < level() || !acquire_source(source)) continue;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                callback = callback_;
                user_data = user_data_;
                if (callback) ++callbacks_in_flight_;
            }

            if (callback) {
                current_callback_source = source.get();
                try { callback(entry.level, entry.message.c_str(), user_data); } catch (...) {}
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (callbacks_in_flight_ != 0) --callbacks_in_flight_;
                    callback_condition_.notify_all();
                }
                current_callback_source = nullptr;
                release_source(source);
            } else {
                // Default output owns no host callback/user_data. Release the
                // source lease first so a redirected or slow stderr cannot
                // delay engine destruction.
                release_source(source);
                default_log_output(entry.message);
            }
        }
    }

    static constexpr size_t kCapacity = 256;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::condition_variable callback_condition_;
    std::deque<Entry> queue_;
    AilaLogCallback callback_ = nullptr;
    void* user_data_ = nullptr;
    size_t callbacks_in_flight_ = 0;
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
    if (!source) return;
    try {
        std::unique_lock<std::mutex> lock(source->mutex);
        source->active = false;
        if (current_callback_source != source.get()) {
            source->condition.wait(lock, [&] { return source->callbacks_in_flight == 0; });
        }
    } catch (...) {}
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
