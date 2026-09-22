#include "inference/Request.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace aila::inference;

void Expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template <typename Fn> void ExpectError(ErrorCode code, Fn&& operation) {
    try { operation(); }
    catch (const Error& error) {
        Expect(error.Code() == code, "Incorrect error classification");
        return;
    }
    throw std::runtime_error("Expected error was not reported");
}

struct Domain { int forwards = 0, batches = 0; int invalid = 0; };

class Backend final : public SessionBackend {
public:
    Backend(Domain& domain, int identity, bool batch = true) : domain(domain), identity(identity), batch(batch) {}
    int ContextLength() const override { return length; }
    int Capacity() const override { return 8; }
    int VocabSize() const override { return 16; }
    size_t CacheBytes() const override { return 0; }
    const void* DecodeBatchKey() const noexcept override { return batch ? &domain : nullptr; }
    void Reset() override { length = 0; }
    int Truncate(int value) override { return length = value; }
    ExecutionResult Forward(const std::vector<int>& tokens, const SamplingInput*, bool capture) override {
        ++domain.forwards;
        length += static_cast<int>(tokens.size());
        return Result(tokens.back(), capture);
    }
    std::vector<ExecutionResult> ForwardDecodeBatch(const std::vector<BackendDecodeInput>& rows) override {
        ++domain.batches;
        std::vector<ExecutionResult> result;
        for (const auto& row : rows) {
            auto& member = *static_cast<Backend*>(row.backend);
            ++member.length;
            result.push_back(member.Result(row.tokenId, row.captureLogits));
        }
        if (domain.invalid == 1) result.pop_back();
        if (domain.invalid == 2) result.back().logits.pop_back();
        if (domain.invalid == 3) --static_cast<Backend*>(rows.back().backend)->length;
        return result;
    }
private:
    ExecutionResult Result(int token, bool capture) const {
        ExecutionResult result;
        if (capture) result.logits.assign(16, static_cast<float>(100 * identity + 10 * length + token));
        return result;
    }
    Domain& domain;
    int identity;
    bool batch;
    int length = 0;
};

std::shared_ptr<SessionState> Session(Domain& domain, int identity, bool batch = true) {
    return std::make_shared<SessionState>(std::make_unique<Backend>(domain, identity, batch));
}

void TestRowsAndSingleton() {
    Domain domain;
    auto a = Session(domain, 1), b = Session(domain, 2), c = Session(domain, 3);
    a->ForwardLogits({1, 2});
    b->ForwardLogits({4});
    auto result = SessionState::ForwardDecodeBatchLogits({b, a, c}, {5, 6, 7});
    Expect(result.size() == 3 && result[0][0] == 225 && result[1][0] == 136 && result[2][0] == 317,
           "Batch row ordering or independent context position was lost");
    Expect(domain.batches == 1 && a->ContextLength() == 3 && b->ContextLength() == 2 && c->ContextLength() == 1,
           "Batch did not consume exactly one token per session");
    result = SessionState::ForwardDecodeBatchLogits({a}, {8});
    Expect(domain.batches == 1 && domain.forwards == 3 && result[0][0] == 148,
           "Singleton did not retain the original Forward path");
    auto unsupported = Session(domain, 4, false);
    Expect(SessionState::ForwardDecodeBatchLogits({unsupported}, {1})[0][0] == 411,
           "Singleton fallback requires no batch capability");
    Expect(!a->Busy() && !b->Busy() && !c->Busy(), "Completed batch leaked session leases");
}

void TestValidationBeforeWork() {
    Domain domain, other;
    auto a = Session(domain, 1), b = Session(domain, 2), c = Session(other, 3), noBatch = Session(domain, 4, false);
    auto unchanged = [&] {
        Expect(!a->Busy() && !b->Busy() && a->ContextLength() == 0 && b->ContextLength() == 0 &&
               domain.forwards == 0 && domain.batches == 0, "Rejected batch executed or retained a lease");
    };
    ExpectError(ErrorCode::InvalidArgument, [&] { SessionState::ForwardDecodeBatchLogits({}, {}); });
    ExpectError(ErrorCode::InvalidArgument, [&] { SessionState::ForwardDecodeBatchLogits({a, b}, {1}); });
    ExpectError(ErrorCode::InvalidArgument, [&] { SessionState::ForwardDecodeBatchLogits({a, nullptr}, {1, 2}); });
    ExpectError(ErrorCode::InvalidArgument, [&] { SessionState::ForwardDecodeBatchLogits({a, a}, {1, 2}); });
    ExpectError(ErrorCode::InvalidArgument, [&] { SessionState::ForwardDecodeBatchLogits({a, c}, {1, 2}); });
    ExpectError(ErrorCode::InvalidArgument, [&] { SessionState::ForwardDecodeBatchLogits({a, b}, {1, 16}); });
    ExpectError(ErrorCode::InvalidArgument, [&] { SessionState::ForwardDecodeBatchLogits({a, b}, {-1, 2}); });
    ExpectError(ErrorCode::InvalidArgument, [&] { SessionState::ForwardDecodeBatchLogits({a, b, c, noBatch, a}, {1, 2, 3, 4, 5}); });
    ExpectError(ErrorCode::UnsupportedModel, [&] { SessionState::ForwardDecodeBatchLogits({noBatch, a}, {1, 2}); });
    unchanged();
    b->ForwardLogits({1, 2, 3, 4, 5, 6, 7, 8});
    ExpectError(ErrorCode::ContextOverflow, [&] { SessionState::ForwardDecodeBatchLogits({a, b}, {1, 2}); });
    Expect(a->ContextLength() == 0 && b->ContextLength() == 8 && domain.batches == 0 && !a->Busy() && !b->Busy(),
           "Full peer capacity rejection changed another row");
}

void TestBusyLeaseRollback() {
    Domain domain;
    auto a = Session(domain, 1), b = Session(domain, 2);
    RequestOptions options;
    options.generation.max_new_tokens = 1;
    auto request = RequestState::Create(b, {3}, options);
    ExpectError(ErrorCode::Busy, [&] { SessionState::ForwardDecodeBatchLogits({a, b}, {1, 2}); });
    Expect(!a->Busy() && b->Busy() && domain.batches == 0, "Busy peer leaked first row lease or released someone else's lease");
    Expect(a->ForwardLogits({2})[0] == 112, "First row unavailable after failed acquisition");
    request->Cancel();
    request->Step();
    Expect(!b->Busy(), "Cancelled peer retained its session");
}

void TestMalformedBackendResults() {
    for (int mode : {1, 2, 3}) {
        Domain domain;
        domain.invalid = mode;
        auto a = Session(domain, 1), b = Session(domain, 2);
        ExpectError(ErrorCode::RuntimeFault, [&] { SessionState::ForwardDecodeBatchLogits({a, b}, {1, 2}); });
        Expect(domain.batches == 1 && !a->Busy() && !b->Busy(), "Malformed result lost error or session leases");
    }
}
}

int main() {
    try {
        TestRowsAndSingleton();
        TestValidationBeforeWork();
        TestBusyLeaseRollback();
        TestMalformedBackendResults();
        std::cout << "Session batch tests passed: rows, B=1 fallback, preflight rejection, Busy rollback, backend postconditions\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Session batch tests failed: " << error.what() << '\n';
        return 1;
    }
}
