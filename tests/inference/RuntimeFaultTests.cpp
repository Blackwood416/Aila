#include "inference/RuntimeFault.hpp"
#include "utils/DeviceException.hpp"
#include "core/AsyncErrorState.hpp"

#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace aila::inference;
using detail::RuntimeFaultRecord;

// These are injected at the same typed boundaries as sycl::exception and
// dnnl::error. The test executable has no device/runtime dependency.
class FakeSyclException final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class FakeDnnlException final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void Expect(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

void ExpectExceptionPreserved(const std::exception_ptr& expected,
                              const std::exception_ptr& actual,
                              const std::string& context) {
    Expect(actual != nullptr, context + ": exception was swallowed");
    // current_exception/rethrow_exception may copy exception objects (MSVC does),
    // so preservation means the same dynamic type and payload, not pointer identity.
    try {
        std::rethrow_exception(expected);
    } catch (const std::exception& expectedError) {
        try {
            std::rethrow_exception(actual);
        } catch (const std::exception& actualError) {
            const std::string diagnosis = context + " [" + expectedError.what() + "]";
            Expect(typeid(actualError) == typeid(expectedError),
                   diagnosis + ": exception dynamic type changed");
            Expect(std::string(actualError.what()) == expectedError.what(),
                   diagnosis + ": exception message changed");
            if (const auto* expectedTyped = dynamic_cast<const Error*>(&expectedError)) {
                const auto* actualTyped = dynamic_cast<const Error*>(&actualError);
                Expect(actualTyped && actualTyped->Code() == expectedTyped->Code(),
                       diagnosis + ": inference error code changed");
            }
            if (const auto* expectedFile =
                    dynamic_cast<const std::filesystem::filesystem_error*>(&expectedError)) {
                const auto& actualFile =
                    dynamic_cast<const std::filesystem::filesystem_error&>(actualError);
                Expect(actualFile.code() == expectedFile->code(),
                       diagnosis + ": filesystem error code changed");
                Expect(actualFile.path1() == expectedFile->path1() &&
                           actualFile.path2() == expectedFile->path2(),
                       diagnosis + ": filesystem error paths changed");
            }
            return;
        } catch (...) {
            throw std::runtime_error(context + ": standard exception became a non-standard exception");
        }
    } catch (int expectedValue) {
        try {
            std::rethrow_exception(actual);
        } catch (int actualValue) {
            Expect(actualValue == expectedValue, context + ": integer exception payload changed");
            return;
        } catch (...) {
            throw std::runtime_error(context + ": integer exception type changed");
        }
    } catch (...) {
        throw std::runtime_error(context + ": unsupported reference exception in test");
    }
}

template <typename Work, typename Drain>
void Initialize(RuntimeFaultRecord& fault, Work&& work, Drain&& drain) {
    detail::RunDeviceLifecycle<FakeSyclException, FakeDnnlException>(
        fault, std::forward<Work>(work), std::forward<Drain>(drain));
}

template <typename Work>
void ExpectError(ErrorCode expected, const std::string& message, Work&& work) {
    try {
        work();
    } catch (const Error& error) {
        Expect(error.Code() == expected && error.what() == message,
               "Lifecycle returned the wrong error code or lost the first diagnosis");
        return;
    }
    throw std::runtime_error("Expected a typed lifecycle error");
}

void TestSynchronousDeviceFailuresSurviveSuccessfulDrain() {
    const std::vector<std::exception_ptr> failures{
        std::make_exception_ptr(FakeSyclException("original SYCL upload failure")),
        std::make_exception_ptr(FakeDnnlException("original oneDNN initialization failure")),
        std::make_exception_ptr(Error(ErrorCode::RuntimeFault, "original typed runtime failure")),
    };
    for (const auto& failure : failures) {
        std::string message;
        try { std::rethrow_exception(failure); }
        catch (const std::exception& error) { message = error.what(); }
        RuntimeFaultRecord fault;
        int drains = 0;
        bool recordedBeforeDrain = false;
        ExpectError(ErrorCode::RuntimeFault, message, [&] {
            Initialize(fault, [&] { std::rethrow_exception(failure); }, [&] {
                ++drains;
                recordedBeforeDrain = fault.Faulted() && fault.Message() == message;
            });
        });
        Expect(recordedBeforeDrain && drains == 1 && fault.Faulted() && fault.Message() == message,
               "Successful drain cleared or failed to record the synchronous device failure");
    }
}

void TestFirstDeviceFailureWinsOverDrainAndLaterReports() {
    RuntimeFaultRecord fault;
    int drains = 0;
    ExpectError(ErrorCode::RuntimeFault, "initial kernel failure", [&] {
        Initialize(fault, [] { throw FakeSyclException("initial kernel failure"); }, [&] {
            ++drains;
            throw FakeDnnlException("secondary drain failure");
        });
    });
    fault.Record("later lifecycle failure");
    fault.DrainNoThrow([] { throw 7; });
    Expect(drains == 1 && std::string(fault.Message()) == "initial kernel failure",
           "A secondary failure replaced the domain's original failure");
    bool executed = false;
    ExpectError(ErrorCode::RuntimeFault, "initial kernel failure", [&] {
        Initialize(fault, [&] { executed = true; }, [&] { ++drains; });
    });
    Expect(!executed && drains == 1, "Faulted lifecycle submitted work or attempted recovery");
}

void TestSuccessfulWorkMustPassItsFinalDrain() {
    RuntimeFaultRecord fault;
    bool initialized = false;
    bool published = false;
    int drains = 0;
    ExpectError(ErrorCode::RuntimeFault, "asynchronous initialization failure", [&] {
        Initialize(fault, [&] { initialized = true; }, [&] {
            ++drains;
            // A queue failure is fatal even if its type is not SYCL-specific.
            throw std::runtime_error("asynchronous initialization failure");
        });
        published = true;
    });
    Expect(initialized && !published && drains == 1 && fault.Faulted(),
           "A failed final drain published initialized resources or retried before recording");

    RuntimeFaultRecord unknown;
    ExpectError(ErrorCode::RuntimeFault, "Unknown device failure", [&] {
        Initialize(unknown, [] {}, [] { throw 9; });
    });
}

void TestHostAndCommandFailuresRemainLocal() {
    RuntimeFaultRecord argumentFault;
    int argumentDrains = 0;
    ExpectError(ErrorCode::InvalidArgument, "invalid dimensions", [&] {
        Initialize(argumentFault, [] { throw std::invalid_argument("invalid dimensions"); },
                   [&] { ++argumentDrains; });
    });
    Expect(!argumentFault.Faulted() && argumentDrains == 1,
           "Ordinary argument failure poisoned the execution domain");

    const std::vector<std::exception_ptr> failures{
        std::make_exception_ptr(std::runtime_error("missing model shard")),
        std::make_exception_ptr(std::filesystem::filesystem_error(
            "model directory", std::filesystem::path("models/missing-shard.safetensors"),
            std::make_error_code(std::errc::no_such_file_or_directory))),
        std::make_exception_ptr(std::bad_alloc()),
        std::make_exception_ptr(19),
        std::make_exception_ptr(Error(ErrorCode::InvalidArgument, "invalid checkpoint")),
        std::make_exception_ptr(Error(ErrorCode::UnsupportedModel, "unsupported backend")),
        std::make_exception_ptr(Error(ErrorCode::ResourceLimit, "session budget")),
        std::make_exception_ptr(Error(ErrorCode::Busy, "session busy")),
        std::make_exception_ptr(Error(ErrorCode::ContextOverflow, "context capacity")),
        std::make_exception_ptr(Error(ErrorCode::SchedulerStopped, "worker stopping")),
    };
    for (const auto& failure : failures) {
        RuntimeFaultRecord fault;
        int drains = 0;
        std::exception_ptr returned;
        try {
            Initialize(fault, [&] { std::rethrow_exception(failure); }, [&] { ++drains; });
        } catch (...) {
            returned = std::current_exception();
        }
        ExpectExceptionPreserved(failure, returned, "Host/file/command failure");
        Expect(!fault.Faulted(), "Host/file/command failure poisoned a healthy domain");
        Expect(drains == 1, "Host/file/command failure did not drain exactly once");
        bool healthyWork = false;
        Initialize(fault, [&] { healthyWork = true; }, [&] { ++drains; });
        Expect(healthyWork && drains == 2, "A healthy domain could not continue after a local failure");
    }
}

void TestHostFailureDoesNotHideADeviceDrainFailure() {
    RuntimeFaultRecord fault;
    ExpectError(ErrorCode::RuntimeFault, "device lost during cleanup", [&] {
        Initialize(fault, [] { throw std::bad_alloc(); }, [] {
            throw FakeSyclException("device lost during cleanup");
        });
    });
    Expect(fault.Faulted() && std::string(fault.Message()) == "device lost during cleanup",
           "Host allocation failure hid a device failure during cleanup");
}

void TestDestructorDrainRecordAndBoundedDiagnosis() {
    RuntimeFaultRecord destructorFault;
    destructorFault.DrainNoThrow([] { throw FakeSyclException("destructor queue failure"); });
    destructorFault.DrainNoThrow([] {});
    ExpectError(ErrorCode::RuntimeFault, "destructor queue failure", [&] {
        destructorFault.CheckUsable();
    });

    const std::string suffix = "\xe4\xb8\xad"; // Three-byte UTF-8 code point.
    const std::string original = std::string(1023, 'x') + suffix;
    RuntimeFaultRecord bounded;
    bounded.Record(original);
    Expect(std::string(bounded.Message()) == std::string(1023, 'x'),
           "Runtime fault record split a UTF-8 sequence at its fixed capacity");
    RuntimeFaultRecord empty;
    empty.Record({});
    empty.Record("must not replace the first empty report");
    ExpectError(ErrorCode::RuntimeFault, "Device execution domain has failed", [&] {
        empty.CheckUsable();
    });
}

void TestWeightUploadRecoveryPreservesDeviceException() {
    const auto uploadFailure = std::make_exception_ptr(FakeSyclException("weight upload device lost"));
    bool recovered = false;
    bool nextTensor = false;
    std::exception_ptr returned;
    try {
        aila::device_failure::RecoverNonDeviceExceptions<FakeSyclException>(
            [&] { std::rethrow_exception(uploadFailure); },
            [&](const std::exception&) { recovered = true; });
        nextTensor = true;
    } catch (...) {
        returned = std::current_exception();
    }
    ExpectExceptionPreserved(uploadFailure, returned, "Weight upload device failure");
    Expect(!recovered, "Best-effort tensor recovery handled a device failure as a local failure");
    Expect(!nextTensor, "Tensor loading continued after a device failure");

    RuntimeFaultRecord fault;
    int drains = 0;
    ExpectError(ErrorCode::RuntimeFault, "weight upload device lost", [&] {
        Initialize(fault, [&] {
            aila::device_failure::RecoverNonDeviceExceptions<FakeSyclException>(
                [&] { std::rethrow_exception(uploadFailure); },
                [&](const std::exception&) { recovered = true; });
        }, [&] { ++drains; });
    });
    Expect(fault.Faulted() && drains == 1 && !recovered,
           "Tensor upload failure failed to reach the outer model lifecycle policy");

    for (const auto& local : {
             std::make_exception_ptr(std::runtime_error("unsupported tensor dtype")),
             std::make_exception_ptr(std::bad_alloc())}) {
        int recoveries = 0;
        aila::device_failure::RecoverNonDeviceExceptions<FakeSyclException>(
            [&] { std::rethrow_exception(local); },
            [&](const std::exception&) { ++recoveries; });
        Expect(recoveries == 1, "The legacy non-device tensor recovery behavior changed");
    }
}

void TestAllocationLimitsRemainLocal() {
    RuntimeFaultRecord fault;
    int drains = 0;
    bool recovered = false;
    ExpectError(ErrorCode::ResourceLimit, "Device allocation budget exceeded", [&] {
        Initialize(fault, [&] {
            aila::device_failure::RecoverNonDeviceExceptions<FakeSyclException,
                aila::memory::AllocationLimitExceeded>([&] {
                    throw aila::memory::AllocationLimitExceeded("Device");
                }, [&](const std::exception&) { recovered = true; });
        }, [&] { ++drains; });
    });
    Expect(!fault.Faulted() && drains == 1 && !recovered,
           "Allocation policy was swallowed by weight recovery or faulted the device");
    bool resumed = false;
    Initialize(fault, [&] { resumed = true; }, [&] { ++drains; });
    Expect(resumed && drains == 2, "Domain could not load after a local allocation limit rejection");
}

void TestAsyncFailureDeliveryAndDrain() {
    aila::AsyncErrorState errors;
    errors.Record(nullptr);
    Expect(!errors.Failed(), "Empty async report faulted the queue");
    errors.Rethrow();
    const auto first = std::make_exception_ptr(FakeSyclException("first async device lost"));
    errors.Record(first);
    std::vector<std::thread> reporters;
    for (int i = 0; i < 8; ++i) {
        reporters.emplace_back([&] {
            for (int report = 0; report < 64; ++report) {
                errors.Record(std::make_exception_ptr(FakeDnnlException("secondary async error")));
            }
        });
    }
    for (auto& reporter : reporters) reporter.join();
    Expect(errors.Failed(), "Async failure was not made visible");
    int drains = 0;
    std::exception_ptr returned;
    try {
        errors.DrainAndRethrow([&] {
            ++drains;
            throw FakeDnnlException("secondary drain failure");
        });
    } catch (...) { returned = std::current_exception(); }
    Expect(drains == 1, "A previously recorded async failure bypassed queue drain");
    ExpectExceptionPreserved(first, returned, "Async first cause across concurrent reports and drain");
    bool submitted = false;
    try {
        errors.Rethrow();
        submitted = true;
    } catch (const FakeSyclException&) {}
    Expect(!submitted, "Known asynchronous failure allowed a later submission");

    aila::AsyncErrorState synchronous;
    returned = nullptr;
    try {
        synchronous.DrainAndRethrow([&] { std::rethrow_exception(first); });
    } catch (...) { returned = std::current_exception(); }
    ExpectExceptionPreserved(first, returned, "Synchronous wait failure preservation");

    RuntimeFaultRecord domain;
    ExpectError(ErrorCode::RuntimeFault, "first async device lost", [&] {
        Initialize(domain, [] {}, [&] { errors.DrainAndRethrow([&] { ++drains; }); });
    });
    Expect(domain.Faulted() && drains >= 2, "Async handler report did not fault the model lifecycle domain");
}

} // namespace

int main() {
    const std::vector<std::pair<const char*, void (*)()>> tests{
        {"synchronous device failures survive a successful drain", TestSynchronousDeviceFailuresSurviveSuccessfulDrain},
        {"first failure survives secondary drain errors and reports", TestFirstDeviceFailureWinsOverDrainAndLaterReports},
        {"initialization cannot publish before final drain", TestSuccessfulWorkMustPassItsFinalDrain},
        {"host, file, parameter and admission failures stay local", TestHostAndCommandFailuresRemainLocal},
        {"host allocation failure cannot hide a device drain error", TestHostFailureDoesNotHideADeviceDrainFailure},
        {"destructor drain and bounded first diagnosis", TestDestructorDrainRecordAndBoundedDiagnosis},
        {"weight loader preserves typed device failures", TestWeightUploadRecoveryPreservesDeviceException},
        {"allocation limits survive loader recovery and remain local", TestAllocationLimitsRemainLocal},
        {"async faults preserve first cause, drain and block new work", TestAsyncFailureDeliveryAndDrain},
    };
    size_t failures = 0;
    for (const auto& test : tests) {
        try {
            test.second();
            std::cout << "PASS: " << test.first << std::endl;
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL: " << test.first << ": " << error.what() << std::endl;
        }
    }
    std::cout << "AilaRuntimeFaultTests: " << tests.size() - failures << " passed, " << failures << " failed\n";
    return failures ? 1 : 0;
}
