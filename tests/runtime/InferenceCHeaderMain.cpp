#include "aila_inference.h"
#include <Windows.h>
#include <iostream>

extern "C" int aila_inference_c_header_checks(void);

int main() {
    const int result = aila_inference_c_header_checks();
    if (result) {
        std::cerr << "FAIL C translation-unit ABI check " << result << '\n';
        return 1;
    }
    const auto module = GetModuleHandleW(L"AilaInference.dll");
    if (!module || !GetProcAddress(module, "aila_inference_request_read") ||
        !GetProcAddress(module, "aila_inference_client_start")) {
        std::cerr << "FAIL undecorated C exports\n";
        return 1;
    }
    std::cout << "PASS C layouts, defaults, UTF-8 errors and undecorated exports\n";
    return 0;
}
