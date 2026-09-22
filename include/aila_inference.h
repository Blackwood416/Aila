/* Aila asynchronous token C ABI. Implemented by the host-only AilaInference DLL.
 * Strings are UTF-8. All input pointers are borrowed only until the call returns.
 * All output storage belongs to the caller; no C++ objects cross this boundary.
 * Initialize structs with their sizeof and AILA_INFERENCE_ABI_VERSION. Options
 * also have default initializers. Reserved input fields must be zero. Version 1
 * accepts a larger struct_size, ignores its tail and writes only the known prefix.
 * The header supports C99 and C++17 with the default 8-byte ABI packing.
 */
#ifndef AILA_INFERENCE_H
#define AILA_INFERENCE_H

#include <stdint.h>

#ifdef __cplusplus
#define AILA_INFERENCE_EXTERN extern "C"
#define AILA_INFERENCE_NOEXCEPT noexcept
#else
#define AILA_INFERENCE_EXTERN extern
#define AILA_INFERENCE_NOEXCEPT
#endif
#if defined(_WIN32)
#define AILA_INFERENCE_CALL __cdecl
#if defined(AILA_INFERENCE_BUILDING_DLL)
#define AILA_INFERENCE_API AILA_INFERENCE_EXTERN __declspec(dllexport)
#else
#define AILA_INFERENCE_API AILA_INFERENCE_EXTERN __declspec(dllimport)
#endif
#else
#define AILA_INFERENCE_CALL
#define AILA_INFERENCE_API AILA_INFERENCE_EXTERN __attribute__((visibility("default")))
#endif

#define AILA_INFERENCE_ABI_VERSION 1u
#define AILA_INFERENCE_MESSAGE_BYTES 1025u
#define AILA_INFERENCE_EPOCH_BYTES 33u
#define AILA_INFERENCE_MAX_HANDLES 4096u

/* Zero is never a live handle. Handles are process/DLL-instance local, typed,
 * generation checked and never rebound to another Worker epoch. */
typedef uint64_t AilaInferenceClientHandle;
typedef uint64_t AilaInferenceModelHandle;
typedef uint64_t AilaInferenceSessionHandle;
typedef uint64_t AilaInferenceRequestHandle;

/* Function return codes; TIMEOUT and END_OF_STREAM are normal read outcomes. */
enum AilaInferenceResult {
    AILA_INFERENCE_OK = 0,
    AILA_INFERENCE_TIMEOUT = 1,
    AILA_INFERENCE_END_OF_STREAM = 2,
    AILA_INFERENCE_INVALID_ARGUMENT = 3,
    AILA_INFERENCE_INVALID_HANDLE = 4,
    AILA_INFERENCE_BUSY = 5,
    AILA_INFERENCE_RESOURCE_LIMIT = 6,
    AILA_INFERENCE_CONTEXT_OVERFLOW = 7,
    AILA_INFERENCE_UNSUPPORTED = 8,
    AILA_INFERENCE_CLOSED = 9,
    AILA_INFERENCE_PROTOCOL_ERROR = 10,
    AILA_INFERENCE_RUNTIME_FAULT = 11,
    AILA_INFERENCE_REQUEST_FAILED = 12,
    AILA_INFERENCE_INTERNAL_ERROR = 13,
    AILA_INFERENCE_OUT_OF_MEMORY = 14,
    AILA_INFERENCE_STALE_EPOCH = 15
};

/* Stable remote error numbers, separate from function return codes. */
enum AilaInferenceRemoteError {
    AILA_INFERENCE_REMOTE_NONE = 0,
    AILA_INFERENCE_REMOTE_INVALID_ARGUMENT = 1,
    AILA_INFERENCE_REMOTE_BUSY = 2,
    AILA_INFERENCE_REMOTE_CONTEXT_OVERFLOW = 3,
    AILA_INFERENCE_REMOTE_UNSUPPORTED_MODEL = 4,
    AILA_INFERENCE_REMOTE_RESOURCE_LIMIT = 5,
    AILA_INFERENCE_REMOTE_RUNTIME_FAULT = 6,
    AILA_INFERENCE_REMOTE_SCHEDULER_STOPPED = 7,
    AILA_INFERENCE_REMOTE_UNSUPPORTED_PROTOCOL = 8,
    AILA_INFERENCE_REMOTE_UNSUPPORTED_CAPABILITY = 9,
    AILA_INFERENCE_REMOTE_STALE_EPOCH = 10,
    AILA_INFERENCE_REMOTE_INVALID_HANDLE = 11,
    AILA_INFERENCE_REMOTE_PROTOCOL_ERROR = 12,
    AILA_INFERENCE_REMOTE_REQUEST_FAILED = 13,
    AILA_INFERENCE_REMOTE_TRANSPORT_ERROR = 14
};

enum AilaInferenceBackend {
    AILA_INFERENCE_QWEN3_NF4 = 1,
    AILA_INFERENCE_QWEN35_NF4 = 2,
    AILA_INFERENCE_QWEN35_BF16 = 3
};
enum AilaInferenceRequestStatus {
    AILA_INFERENCE_QUEUED = 0,
    AILA_INFERENCE_PREFILL = 1,
    AILA_INFERENCE_DECODE = 2,
    AILA_INFERENCE_FINALIZING = 3,
    AILA_INFERENCE_COMPLETED = 4,
    AILA_INFERENCE_CANCELLED = 5,
    AILA_INFERENCE_FAILED = 6
};
enum AilaInferenceEndReason {
    AILA_INFERENCE_END_NONE = 0,
    AILA_INFERENCE_END_LENGTH = 1,
    AILA_INFERENCE_END_STOP_TOKEN = 2,
    AILA_INFERENCE_END_CANCELLED = 3,
    AILA_INFERENCE_END_OUTPUT_BACKPRESSURE = 4,
    AILA_INFERENCE_END_SCHEDULER_STOPPED = 5,
    AILA_INFERENCE_END_RUNTIME_FAULT = 6,
    AILA_INFERENCE_END_FAILED = 7
};
enum AilaInferenceDomainStatus {
    AILA_INFERENCE_DOMAIN_STARTING = 0,
    AILA_INFERENCE_DOMAIN_READY = 1,
    AILA_INFERENCE_DOMAIN_STOPPING = 2,
    AILA_INFERENCE_DOMAIN_FAULTED = 3,
    AILA_INFERENCE_DOMAIN_CLOSED = 4
};
enum AilaInferenceMemoryState {
    AILA_INFERENCE_MEMORY_UNINITIALIZED = 0,
    AILA_INFERENCE_MEMORY_BUSY = 1,
    AILA_INFERENCE_MEMORY_READY = 2
};
enum AilaInferenceEventKind {
    AILA_INFERENCE_EVENT_NONE = 0,
    AILA_INFERENCE_EVENT_TOKEN = 1,
    AILA_INFERENCE_EVENT_TERMINAL = 2
};
enum AilaInferenceThinkingPolicy {
    AILA_INFERENCE_THINKING_LIMIT_EACH_REOPEN = 0,
    AILA_INFERENCE_THINKING_CLOSE_ONCE = 1
};
enum AilaInferenceSamplingPolicy {
    AILA_INFERENCE_SAMPLING_AUTO = 0,
    AILA_INFERENCE_SAMPLING_HOST = 1
};

enum AilaInferencePrefillLogitsPolicy {
    AILA_INFERENCE_PREFILL_LOGITS_LAST = 0,
    /* Preserve the old Hybrid ABI's first-row sample for short continuation
       chunks. New inference callers should keep LAST. */
    AILA_INFERENCE_PREFILL_LOGITS_LEGACY = 1
};

/* Optional extended_generation_v1. Defaults remain 1024; prompt + generation
   must still fit context and the negotiated outstanding-token reservation. */
#define AILA_INFERENCE_EXTENDED_MAX_GENERATED_TOKENS 32768u

#pragma pack(push, 8)
typedef struct AilaInferenceError {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t code;
    uint32_t remote_error;
    char message[AILA_INFERENCE_MESSAGE_BYTES];
} AilaInferenceError;

typedef struct AilaInferenceClientOptions {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t handshake_timeout_ms;
    uint32_t command_timeout_ms;
    uint32_t shutdown_timeout_ms;
    uint32_t max_active_requests;
    uint32_t max_outstanding_requests;
    uint32_t max_prompt_tokens;
    uint32_t max_generated_tokens;
    uint32_t max_sessions;
    uint32_t max_retained_request_handles;
    uint32_t max_buffered_tokens_per_request;
    uint32_t token_window;
    uint32_t max_prefill_chunk_size;
    uint64_t max_outstanding_tokens;
    uint64_t max_session_cache_bytes;
    uint32_t reserved[8];
} AilaInferenceClientOptions;

typedef struct AilaInferenceModelOptions {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t max_sequence_length;
    uint32_t max_sessions;
    uint64_t max_session_cache_bytes;
    uint32_t allocation_limits_enabled; /* 0/1; when 0 all three limits must be 0 */
    uint32_t reserved0;
    uint64_t max_device_bytes;          /* explicit USM; 0 = no byte ceiling */
    uint64_t max_pinned_host_bytes;     /* explicit USM; 0 = no byte ceiling */
    uint64_t max_session_host_bytes;    /* mirrors/checkpoints, not process RSS */
    uint32_t reserved[8];
} AilaInferenceModelOptions;

typedef struct AilaInferenceRequestOptions {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t max_new_tokens;
    uint32_t prefill_chunk_size;
    uint32_t do_sample;                /* 0/1 */
    uint32_t use_fixed_seed;           /* 0/1 */
    uint64_t sampling_seed;
    double temperature;
    double top_p;
    double repetition_penalty;
    double presence_penalty;
    double frequency_penalty;
    uint32_t top_k;
    uint32_t stop_token_count;
    const int32_t* stop_token_ids;
    int32_t thinking_budget_tokens;    /* -1 disabled; >=0 content token budget */
    uint32_t prompt_opens_think;        /* 0/1 */
    int32_t thinking_open_token_id;
    int32_t thinking_close_token_id;
    const int32_t* thinking_close_token_ids; /* starts with close marker, 1..16 */
    uint32_t thinking_close_token_count;
    uint32_t reserved[8];
} AilaInferenceRequestOptions;

typedef struct AilaInferenceRequestPolicy {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t thinking_policy;
    uint32_t sampling_policy;
    uint32_t prefill_logits_policy;
    uint32_t reserved[3];
} AilaInferenceRequestPolicy;

typedef struct AilaInferenceProgress {
    uint32_t status;
    uint32_t end_reason;
    uint32_t remote_error;
    uint32_t reserved0;
    uint64_t prompt_tokens_processed;
    uint64_t generated_tokens;
    uint64_t event_count;
    char message[AILA_INFERENCE_MESSAGE_BYTES];
} AilaInferenceProgress;

typedef struct AilaInferenceEvent {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t kind;
    uint32_t local_failure;            /* 1 = host failure, not a wire terminal */
    uint64_t sequence;                 /* local failure terminal has sequence 0 */
    int32_t token_id;                  /* -1 unless kind == TOKEN */
    uint32_t delivery_abandoned;
    uint64_t delivered_tokens;
    uint64_t discarded_published_tokens;
    AilaInferenceProgress progress;
} AilaInferenceEvent;

typedef struct AilaInferenceClientInfo {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t domain_status;
    uint32_t reserved0;
    uint64_t outstanding_requests;
    uint64_t retained_request_handles;
    uint64_t pending_commands;
    uint64_t queued_commands;
    uint64_t reserved_tokens;
    uint64_t ingress_bytes;
    uint64_t transport_bytes;
    uint64_t peak_ingress_bytes;
    uint64_t peak_transport_bytes;
    uint64_t dropped_logs;
    char epoch[AILA_INFERENCE_EPOCH_BYTES];
    char message[AILA_INFERENCE_MESSAGE_BYTES];
} AilaInferenceClientInfo;

/* Optional runtime_memory_v1 snapshot. Byte/count/limit fields are meaningful
 * only for MEMORY_READY. They describe explicit USM, not RSS, WDDM residency,
 * Session host mirrors or library/driver-private allocation. Lifetime peaks
 * survive model unload and may exceed a later, lower allocation limit.
 */
typedef struct AilaInferenceRuntimeMemoryInfo {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t state;
    uint32_t reserved0;
    uint64_t device_bytes;
    uint64_t peak_device_bytes;
    uint64_t pinned_host_bytes;
    uint64_t peak_pinned_host_bytes;
    uint64_t device_allocations;
    uint64_t pinned_host_allocations;
    uint64_t device_limit_bytes;
    uint64_t pinned_host_limit_bytes;
    char epoch[AILA_INFERENCE_EPOCH_BYTES];
} AilaInferenceRuntimeMemoryInfo;

/* Worker-wide logs have no Request identity. event_ordinal shares the wire
 * connection's order with token/terminal/fault events and may contain gaps.
 * message_bytes excludes the trailing NUL; epoch identifies this Worker run.
 * dropped_logs counts host queue overflow/release, not Worker-side drops.
 */
typedef struct AilaInferenceLogEvent {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t event_ordinal;
    uint32_t level;
    uint32_t message_bytes;
    uint64_t dropped_logs;
    char epoch[AILA_INFERENCE_EPOCH_BYTES];
    char message[AILA_INFERENCE_MESSAGE_BYTES];
} AilaInferenceLogEvent;

typedef struct AilaInferenceModelInfo {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t model_id;
    uint32_t vocab_size;
    uint32_t max_sequence_length;
    uint32_t backend;
    uint32_t has_memory_info;
    uint64_t session_reservation_bytes;
    uint64_t resident_device_bytes;
    uint64_t session_host_reservation_bytes;
    uint64_t max_device_bytes;
    uint64_t max_pinned_host_bytes;
    uint64_t max_session_host_bytes;
    char epoch[AILA_INFERENCE_EPOCH_BYTES];
} AilaInferenceModelInfo;

typedef struct AilaInferenceSessionInfo {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t session_id;
    uint64_t model_id;
    uint32_t capacity;
    uint32_t busy;
    uint64_t cache_bytes;
    uint32_t has_context_length;
    uint32_t context_length;
    char epoch[AILA_INFERENCE_EPOCH_BYTES];
} AilaInferenceSessionInfo;

typedef struct AilaInferenceRequestInfo {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t inference_id;
    uint64_t buffered_tokens;
    uint64_t last_sent_sequence;
    uint64_t last_acknowledged_sequence;
    uint32_t completion_ready;
    uint32_t reserved0;
    char epoch[AILA_INFERENCE_EPOCH_BYTES];
    AilaInferenceProgress progress;
} AilaInferenceRequestInfo;
#pragma pack(pop)

AILA_INFERENCE_API uint32_t AILA_INFERENCE_CALL aila_inference_abi_version(void) AILA_INFERENCE_NOEXCEPT;
AILA_INFERENCE_API const char* AILA_INFERENCE_CALL aila_inference_build_id(void) AILA_INFERENCE_NOEXCEPT;
/* Initializers take the caller's sizeof and preserve any unknown tail bytes. */
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_default_client_options(
    AilaInferenceClientOptions* options, uint32_t struct_size) AILA_INFERENCE_NOEXCEPT;
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_default_model_options(
    AilaInferenceModelOptions* options, uint32_t struct_size) AILA_INFERENCE_NOEXCEPT;
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_default_request_options(
    AilaInferenceRequestOptions* options, uint32_t struct_size) AILA_INFERENCE_NOEXCEPT;

/* Error output is optional. If supplied, its header must also be initialized.
 * Each call clears/fills its own error; there is no process/thread last-error.
 * Valid out_handle storage is zeroed before a potentially failing operation.
 * Options may be NULL for defaults. The Worker build ID must match this DLL.
 * Worker path may be NULL to select AilaWorker.exe directly in runtime_directory.
 * Launch, load, create, submit-acceptance, query and shutdown may block. Submit
 * returns after acceptance, without waiting for generation to complete.
 * RPC timeout 0 uses the configured timeout. Read timeout 0 is a nonblocking poll.
 */
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_client_start(
    const char* runtime_directory, const char* worker_executable,
    const AilaInferenceClientOptions* options, AilaInferenceClientHandle* out_handle,
    AilaInferenceError* error) AILA_INFERENCE_NOEXCEPT;
/* Additive continuous_batching_v1 configuration. The bound must be 1..4 and
 * <= options.max_active_requests (defaults apply for NULL options). One uses
 * the original scheduling path and requires no new Worker capability. Larger
 * values require explicit capability acknowledgement during startup. They do
 * not increase admission, Session, cache, token or output limits. The bound is
 * fixed for the Client's Worker lifetime; compatible decode requests may batch.
 */
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_client_start_with_batching(
    const char* runtime_directory, const char* worker_executable,
    const AilaInferenceClientOptions* options, uint32_t max_decode_batch_size,
    AilaInferenceClientHandle* out_handle, AilaInferenceError* error) AILA_INFERENCE_NOEXCEPT;
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_client_snapshot(
    AilaInferenceClientHandle handle, AilaInferenceClientInfo* out_info,
    AilaInferenceError* error) AILA_INFERENCE_NOEXCEPT;
/* One explicit RPC; 0 timeout uses the configured RPC timeout. Returns OK for
 * all three memory states. BUSY means no immediate execution-lock snapshot;
 * UNINITIALIZED does not create a device. Neither means zero device usage.
 * Output header is checked before any RPC. Requires runtime_memory_v1 only
 * when invoked; starting a Client does not require this optional capability.
 */
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_client_query_memory(
    AilaInferenceClientHandle handle, uint32_t timeout_ms, AilaInferenceRuntimeMemoryInfo* out_info,
    AilaInferenceError* error) AILA_INFERENCE_NOEXCEPT;
/* One log reader per Client; overlap returns BUSY. Zero timeout polls.
 * Logs use a separate bounded host queue (64 KiB by default); new logs are
 * dropped on overflow without blocking inference or consuming token credit.
 * Headers are validated before consumption. Queued logs can be drained after
 * shutdown/fault, then END_OF_STREAM is returned; consult client_snapshot for
 * connection failure. Client release discards logs and wakes a pending read,
 * while previously created child handles remain usable.
 */
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_client_read_log(
    AilaInferenceClientHandle handle, uint32_t timeout_ms, AilaInferenceLogEvent* out_event,
    AilaInferenceError* error) AILA_INFERENCE_NOEXCEPT;
/* Optional worker_log_control_v1. Applies to the entire shared Worker, including
 * all Models/Sessions. 0=debug, 1=info, 2=warning, 3=error. A timeout does not
 * undo a submitted level change; no callback runs on the I/O/GPU threads.
 */
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_client_set_log_level(
    AilaInferenceClientHandle handle, uint32_t level, uint32_t timeout_ms,
    AilaInferenceError* error) AILA_INFERENCE_NOEXCEPT;
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_client_shutdown(
    AilaInferenceClientHandle handle, uint32_t timeout_ms,
    AilaInferenceError* error) AILA_INFERENCE_NOEXCEPT;
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_client_release(
    AilaInferenceClientHandle handle, AilaInferenceError* error) AILA_INFERENCE_NOEXCEPT;
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_model_load(
    AilaInferenceClientHandle client, uint32_t backend, const char* model_directory,
    const AilaInferenceModelOptions* options, uint32_t timeout_ms,
    AilaInferenceModelHandle* out_handle, AilaInferenceError* error) AILA_INFERENCE_NOEXCEPT;
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_model_info(
    AilaInferenceModelHandle handle, AilaInferenceModelInfo* out_info,
    AilaInferenceError* error) AILA_INFERENCE_NOEXCEPT;
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_model_release(
    AilaInferenceModelHandle handle, AilaInferenceError* error) AILA_INFERENCE_NOEXCEPT;
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_session_create(
    AilaInferenceModelHandle model, uint32_t timeout_ms,
    AilaInferenceSessionHandle* out_handle, AilaInferenceError* error) AILA_INFERENCE_NOEXCEPT;
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_session_query(
    AilaInferenceSessionHandle handle, uint32_t timeout_ms, AilaInferenceSessionInfo* out_info,
    AilaInferenceError* error) AILA_INFERENCE_NOEXCEPT;
/* Optional session_context_v1 capability. Mutations reject busy Sessions and
 * reserve the Session until completion. Reset returns context_length == 0.
 * Truncate returns the actual restored checkpoint, which may precede length;
 * callers must replay the missing prefix. length must not exceed the current
 * context length. An RPC timeout does not cancel a submitted mutation: query
 * until idle before deciding whether another operation is needed.
 * Output/error headers are checked before submitting any mutation.
 */
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_session_reset(
    AilaInferenceSessionHandle handle, uint32_t timeout_ms, AilaInferenceSessionInfo* out_info,
    AilaInferenceError* error) AILA_INFERENCE_NOEXCEPT;
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_session_truncate(
    AilaInferenceSessionHandle handle, uint32_t length, uint32_t timeout_ms,
    AilaInferenceSessionInfo* out_info, AilaInferenceError* error) AILA_INFERENCE_NOEXCEPT;
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_session_release(
    AilaInferenceSessionHandle handle, AilaInferenceError* error) AILA_INFERENCE_NOEXCEPT;
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_request_submit(
    AilaInferenceSessionHandle session, const int32_t* prompt_token_ids, uint32_t prompt_token_count,
    const AilaInferenceRequestOptions* options, uint32_t timeout_ms,
    AilaInferenceRequestHandle* out_handle, AilaInferenceError* error) AILA_INFERENCE_NOEXCEPT;
/* Same contract as submit, with explicit optional policies. CLOSE_ONCE
 * requires thinking_budget_close_once_v1 when a budget is enabled: the first
 * forced close exhausts the policy, even if the model opens reasoning again.
 * HOST requires host_sampling_v1 and preserves the host sampler's numerical
 * behavior. NULL policy uses LIMIT_EACH_REOPEN/AUTO, as does existing submit.
 * This does not alter budget 0;
 * callers implementing a no-think template pass budget -1 for generation.
 */
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_request_submit_with_policy(
    AilaInferenceSessionHandle session, const int32_t* prompt_token_ids, uint32_t prompt_token_count,
    const AilaInferenceRequestOptions* options, const AilaInferenceRequestPolicy* policy, uint32_t timeout_ms,
    AilaInferenceRequestHandle* out_handle, AilaInferenceError* error) AILA_INFERENCE_NOEXCEPT;
/* Optional prefill_only_v1 capability. Append a nonempty prefix without
 * sampling. Uses the same scheduler admission, busy check and Request lifetime
 * as submit. chunk_size must be positive and within the negotiated limit.
 * Success emits one terminal event (COMPLETED/END_LENGTH, zero generated
 * tokens). Cancellation may leave a completed prefix of prompt chunks in the
 * Session; query its actual length before replaying the remainder.
 */
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_request_prefill(
    AilaInferenceSessionHandle session, const int32_t* prompt_token_ids, uint32_t prompt_token_count,
    uint32_t chunk_size, uint32_t timeout_ms, AilaInferenceRequestHandle* out_handle,
    AilaInferenceError* error) AILA_INFERENCE_NOEXCEPT;
/* One reader per Request. An overlapping Read returns BUSY. Output headers are
 * checked before consuming anything. Success returns exactly one token/terminal;
 * after that terminal, Read returns END_OF_STREAM. TIMEOUT does not cancel.
 * No caller callback runs on the connection's I/O threads. Independent Requests
 * may be read concurrently. All other operations can overlap with Read.
 */
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_request_read(
    AilaInferenceRequestHandle handle, uint32_t timeout_ms, AilaInferenceEvent* out_event,
    AilaInferenceError* error) AILA_INFERENCE_NOEXCEPT;
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_request_cancel(
    AilaInferenceRequestHandle handle, uint32_t timeout_ms, uint32_t* out_cancel_requested,
    AilaInferenceError* error) AILA_INFERENCE_NOEXCEPT;
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_request_query(
    AilaInferenceRequestHandle handle, uint32_t timeout_ms, AilaInferenceRequestInfo* out_info,
    AilaInferenceError* error) AILA_INFERENCE_NOEXCEPT;
/* Release is nonblocking; zero is a no-op. Other stale/wrong-type handles fail.
 * Request release immediately asks the private connection to cancel and drain,
 * even while a reader owns its wrapper. Parent release leaves children usable.
 * Calls already in flight may complete or report a released/closed handle.
 * Shutdown explicitly stops the entire connection and is distinct from release.
 * Before unloading the DLL, finish all API calls, shut down clients, and release
 * every handle. Releasing the last owner also initiates asynchronous shutdown.
 */
AILA_INFERENCE_API int32_t AILA_INFERENCE_CALL aila_inference_request_release(
    AilaInferenceRequestHandle handle, AilaInferenceError* error) AILA_INFERENCE_NOEXCEPT;

#endif /* AILA_INFERENCE_H */
