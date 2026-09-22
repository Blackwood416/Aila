#pragma pack(push, 1)
#include "aila_inference.h"
struct CallerPacking { char byte; double value; };
_Static_assert(sizeof(struct CallerPacking) == 9, "Public header must restore caller packing");
#pragma pack(pop)

#include <stddef.h>
#include <string.h>

/* These assertions run in a C translation unit, independent of C++ types. */
_Static_assert(sizeof(void*) == 8, "This Windows ABI fixture targets x64");
_Static_assert(sizeof(AilaInferenceClientHandle) == 8, "Handle width");
_Static_assert(sizeof(AilaInferenceError) == 1044, "Error layout");
_Static_assert(sizeof(AilaInferenceClientOptions) == 104, "Client options layout");
_Static_assert(sizeof(AilaInferenceModelOptions) == 88, "Model options layout");
_Static_assert(sizeof(AilaInferenceRequestOptions) == 152, "Request options layout");
_Static_assert(sizeof(AilaInferenceRequestPolicy) == 32, "Request policy layout");
_Static_assert(offsetof(AilaInferenceRequestPolicy, prefill_logits_policy) == 16, "Policy extension uses a reserved slot");
_Static_assert(sizeof(AilaInferenceProgress) == 1072, "Progress layout");
_Static_assert(sizeof(AilaInferenceEvent) == 1120, "Event layout");
_Static_assert(sizeof(AilaInferenceClientInfo) == 1160, "Client info layout");
_Static_assert(sizeof(AilaInferenceRuntimeMemoryInfo) == 120, "Runtime memory layout");
_Static_assert(offsetof(AilaInferenceRuntimeMemoryInfo, device_bytes) == 16, "Memory counter alignment");
_Static_assert(offsetof(AilaInferenceRuntimeMemoryInfo, epoch) == 80, "Memory epoch offset");
_Static_assert(sizeof(AilaInferenceLogEvent) == 1096, "Log event layout");
_Static_assert(offsetof(AilaInferenceLogEvent, event_ordinal) == 8, "Log ordinal alignment");
_Static_assert(offsetof(AilaInferenceLogEvent, message) == 65, "Log message offset");
_Static_assert(sizeof(AilaInferenceModelInfo) == 120, "Model info layout");
_Static_assert(sizeof(AilaInferenceSessionInfo) == 88, "Session info layout");
_Static_assert(sizeof(AilaInferenceRequestInfo) == 1160, "Request info layout");
_Static_assert(offsetof(AilaInferenceRequestOptions, stop_token_ids) == 80, "Pointer alignment");
_Static_assert(offsetof(AilaInferenceEvent, progress) == 48, "Embedded progress offset");

int aila_inference_c_header_checks(void) {
    struct ExtendedClient {
        AilaInferenceClientOptions options;
        uint64_t tail;
    } client;
    AilaInferenceRequestOptions request;
    AilaInferenceModelOptions model;
    AilaInferenceError error = {sizeof(error), AILA_INFERENCE_ABI_VERSION};
    AilaInferenceSessionInfo session = {sizeof(session), AILA_INFERENCE_ABI_VERSION};
    AilaInferenceLogEvent log = {sizeof(log), AILA_INFERENCE_ABI_VERSION};
    AilaInferenceRuntimeMemoryInfo memory = {sizeof(memory), AILA_INFERENCE_ABI_VERSION};
    AilaInferenceClientHandle handle = 123;
    memset(&client, 0xa5, sizeof(client));
    if (aila_inference_abi_version() != AILA_INFERENCE_ABI_VERSION) return 1;
    if (!aila_inference_build_id() || !aila_inference_build_id()[0]) return 2;
    if (aila_inference_default_client_options(&client.options, sizeof(client)) != AILA_INFERENCE_OK) return 3;
    if (client.options.struct_size != sizeof(client) || client.tail != UINT64_C(0xa5a5a5a5a5a5a5a5)) return 4;
    if (aila_inference_default_request_options(&request, sizeof(request)) != AILA_INFERENCE_OK) return 5;
    if (request.thinking_budget_tokens != -1 || request.thinking_open_token_id != -1 ||
        request.thinking_close_token_id != -1 || request.thinking_close_token_count || request.stop_token_ids) return 6;
    if (aila_inference_default_model_options(&model, sizeof(model)) != AILA_INFERENCE_OK) return 7;
    if (aila_inference_default_model_options(&model, sizeof(model) - 1) != AILA_INFERENCE_INVALID_ARGUMENT) return 8;
    if (model.struct_size != sizeof(model) || model.max_sequence_length != 4096) return 9;
    if (aila_inference_client_start(NULL, NULL, &client.options, &handle, &error) != AILA_INFERENCE_INVALID_ARGUMENT) return 10;
    if (handle || error.code != AILA_INFERENCE_INVALID_ARGUMENT || !error.message[0]) return 11;
    if (aila_inference_client_release(0, &error) != AILA_INFERENCE_OK || error.code || error.message[0]) return 12;
    if (aila_inference_session_reset(0, 0, &session, &error) != AILA_INFERENCE_INVALID_HANDLE) return 13;
    if (aila_inference_session_truncate(0, 0, 0, &session, &error) != AILA_INFERENCE_INVALID_HANDLE) return 14;
    if (aila_inference_request_prefill(0, NULL, 0, 1, 0, &handle, &error) != AILA_INFERENCE_INVALID_HANDLE) return 15;
    if (handle) return 16;
    if (aila_inference_request_submit_with_policy(0, NULL, 0, NULL, NULL, 0, &handle, &error) != AILA_INFERENCE_INVALID_HANDLE) return 17;
    if (aila_inference_client_read_log(0, 0, &log, &error) != AILA_INFERENCE_INVALID_HANDLE) return 18;
    if (aila_inference_client_set_log_level(0, 0, 0, &error) != AILA_INFERENCE_INVALID_HANDLE) return 19;
    if (aila_inference_client_set_log_level(0, 4, 0, &error) != AILA_INFERENCE_INVALID_ARGUMENT) return 20;
    if (aila_inference_client_query_memory(0, 0, &memory, &error) != AILA_INFERENCE_INVALID_HANDLE) return 21;
    --memory.struct_size;
    if (aila_inference_client_query_memory(0, 0, &memory, &error) != AILA_INFERENCE_INVALID_ARGUMENT) return 22;
    handle = 123;
    if (aila_inference_client_start_with_batching("C:/aila-invalid-config", NULL, &client.options,
            0, &handle, &error) != AILA_INFERENCE_INVALID_ARGUMENT || handle) return 23;
    client.options.max_active_requests = 1;
    handle = 123;
    if (aila_inference_client_start_with_batching("C:/aila-invalid-config", NULL, &client.options,
            2, &handle, &error) != AILA_INFERENCE_INVALID_ARGUMENT || handle) return 24;
    if (client.tail != UINT64_C(0xa5a5a5a5a5a5a5a5)) return 25;
    client.options.reserved[0] = 1;
    handle = 123;
    if (aila_inference_client_start_with_batching("C:/aila-invalid-config", NULL, &client.options,
            1, &handle, &error) != AILA_INFERENCE_INVALID_ARGUMENT || handle) return 26;
    return 0;
}
