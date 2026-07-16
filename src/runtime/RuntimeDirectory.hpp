#pragma once

#include <windows.h>

#include <filesystem>
#include <optional>
#include <string>

namespace aila::runtime {

std::filesystem::path proxy_module_path(HMODULE module);

std::optional<std::wstring> runtime_directory_override();

std::filesystem::path resolve_runtime_directory(
    const std::filesystem::path& proxy_module,
    const std::optional<std::wstring>& configured_value);

std::filesystem::path require_worker_executable(
    const std::filesystem::path& runtime_directory);

} // namespace aila::runtime
