#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace aila::runtime {

struct CaseInsensitiveLess {
    bool operator()(const std::wstring& left, const std::wstring& right) const noexcept;
};

using EnvironmentMap = std::map<std::wstring, std::wstring, CaseInsensitiveLess>;

EnvironmentMap parse_environment_block(const std::vector<wchar_t>& block);

EnvironmentMap current_environment();

// True when `name` belongs to a GPU-runtime instrumentation family that the
// isolated worker must not inherit from its host process (XPTI/UR tracing
// layers, Level Zero sysman/metrics toggles, OpenCL ICD overrides). Hosts that
// need one of these forwarded can list it in AILA_WORKER_ENV_PASSTHROUGH.
bool is_scrubbed_runtime_variable(const std::wstring& name) noexcept;

std::filesystem::path system_root_directory();

std::vector<wchar_t> build_isolated_environment(
    const EnvironmentMap& inherited,
    const std::filesystem::path& runtime_directory,
    const std::filesystem::path& system_root);

} // namespace aila::runtime
