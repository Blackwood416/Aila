#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace aila::memory {

class AllocationLimitExceeded final : public std::runtime_error {
public:
    explicit AllocationLimitExceeded(const char* kind)
        : std::runtime_error(std::string(kind) + " allocation budget exceeded") {}
};

// Serialized by the owning execution domain. Counts explicitly owned payload,
// including transient replacement allocations; allocator/driver metadata is not
// part of this counter. An unsuccessful reservation never changes the ledger.
class AllocationBudget final {
public:
    void SetLimit(size_t value) {
        if (value && current_ > value) throw AllocationLimitExceeded("Existing");
        limit_ = value;
    }

    void Check(size_t bytes, const char* kind) const {
        if (bytes > std::numeric_limits<size_t>::max() - current_ ||
            (limit_ && bytes > limit_ - current_)) {
            throw AllocationLimitExceeded(kind);
        }
    }

    void Acquire(size_t bytes, const char* kind) {
        Check(bytes, kind);
        current_ += bytes;
        if (current_ > peak_) peak_ = current_;
        ++count_;
    }

    void Release(size_t bytes) noexcept {
        current_ -= bytes;
        --count_;
    }

    size_t Current() const noexcept { return current_; }
    size_t Peak() const noexcept { return peak_; }
    size_t Count() const noexcept { return count_; }
    size_t Limit() const noexcept { return limit_; }

private:
    size_t current_ = 0;
    size_t peak_ = 0;
    size_t count_ = 0;
    size_t limit_ = 0;
};

} // namespace aila::memory
