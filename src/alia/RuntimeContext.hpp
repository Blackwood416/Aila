#pragma once

#include "../core/Context.hpp"

#include <sycl/sycl.hpp>

namespace aila::alia {

class RuntimeContext {
public:
    RuntimeContext();

    Context& foreground() { return foreground_; }
    Context& background() { return background_; }
    const sycl::device& device() const { return device_; }
    const sycl::context& sycl_context() const { return sycl_context_; }

private:
    static sycl::queue make_in_order_queue(const sycl::context& context,
                                           const sycl::device& device);

    sycl::device device_;
    sycl::context sycl_context_;
    Context foreground_;
    Context background_;
};

}  // namespace aila::alia

