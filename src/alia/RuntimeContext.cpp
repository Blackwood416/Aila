#include "RuntimeContext.hpp"

namespace aila::alia {

sycl::queue RuntimeContext::make_in_order_queue(const sycl::context& context,
                                                const sycl::device& device) {
    return sycl::queue{context, device, sycl::property::queue::in_order()};
}

RuntimeContext::RuntimeContext()
    : device_(sycl::default_selector_v),
      sycl_context_(device_),
      foreground_(make_in_order_queue(sycl_context_, device_)),
      background_(make_in_order_queue(sycl_context_, device_)) {}

}  // namespace aila::alia

