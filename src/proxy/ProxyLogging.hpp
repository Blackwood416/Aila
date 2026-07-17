#pragma once

#include "aila_api.h"

#include <memory>
#include <string_view>

namespace aila::proxy::logging {

struct SourceState;

using Source = std::shared_ptr<SourceState>;

Source create_source();
void deactivate(const Source& source) noexcept;
void set_callback(AilaLogCallback callback, void* user_data) noexcept;
void set_level(int level) noexcept;
int level() noexcept;
void enqueue(const Source& source, int level, std::string_view message) noexcept;

} // namespace aila::proxy::logging
