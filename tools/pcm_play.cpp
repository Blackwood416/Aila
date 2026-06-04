// Streaming PCM player for Windows (waveOut double-buffering).
// Reads 24kHz mono f32 PCM from stdin pipe, plays as data arrives.
// Compile: cl /O2 /Fe:pcm_play.exe tools/pcm_play.cpp /link winmm.lib
// Usage: aila --stream-tts ... 2>/dev/null | pcm_play

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <mmsystem.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <io.h>
#include <fcntl.h>

#pragma comment(lib, "winmm.lib")

static const int SAMPLE_RATE  = 24000;
static const size_t CHUNK_F32 = SAMPLE_RATE / 5;  // 200ms of f32 samples
static const size_t MIN_BUFFER_F32 = SAMPLE_RATE / 2; // start playing after 500ms

int main() {
    _setmode(_fileno(stdin), _O_BINARY);
    HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);

    // Audio format
    WAVEFORMATEX wf = {};
    wf.wFormatTag = WAVE_FORMAT_PCM;
    wf.nChannels = 1;
    wf.nSamplesPerSec = SAMPLE_RATE;
    wf.wBitsPerSample = 16;
    wf.nBlockAlign = 2;
    wf.nAvgBytesPerSec = SAMPLE_RATE * 2;

    HWAVEOUT hwo = nullptr;
    if (waveOutOpen(&hwo, WAVE_MAPPER, &wf, 0, 0, CALLBACK_NULL) != MMSYSERR_NOERROR) {
        fprintf(stderr, "pcm_play: failed to open audio device\n");
        return 1;
    }

    // Accumulation buffer: f32 PCM from stdin, consumed by waveOut as s16
    std::vector<float> f32_buf;
    f32_buf.reserve(SAMPLE_RATE * 10);
    // Separate s16 buffers per waveOut header (prevents pointer invalidation)
    std::vector<int16_t> s16_a, s16_b;

    size_t total_read = 0;
    size_t total_played = 0;
    bool pipe_closed = false;
    bool started = false;

    WAVEHDR hdr_a = {}, hdr_b = {};
    bool b_active = false;

    float tmp[512]; // small read buffer
    DWORD avail = 0;

    fprintf(stderr, "pcm_play: buffering...\r");
    fflush(stderr);

    while (!pipe_closed || total_played < total_read) {
        // --- Read more data from pipe ---
        if (!pipe_closed) {
            if (PeekNamedPipe(hStdin, NULL, 0, NULL, &avail, NULL) && avail >= sizeof(float)) {
                DWORD to_read = std::min<DWORD>(avail, sizeof(tmp));
                DWORD bytes_read = 0;
                if (ReadFile(hStdin, tmp, to_read, &bytes_read, NULL) && bytes_read > 0) {
                    size_t n = bytes_read / sizeof(float);
                    f32_buf.insert(f32_buf.end(), tmp, tmp + n);
                    total_read += n;
                }
            } else if (avail == 0) {
                Sleep(10); // pipe empty, wait for more data
            } else {
                pipe_closed = true;
            }
        }

        // --- Start playback once enough buffered ---
        if (!started && total_read >= MIN_BUFFER_F32) {
            started = true;
            fprintf(stderr, "pcm_play: playing (%.1fs buffered)\n",
                    (double)total_read / SAMPLE_RATE);
            size_t sz = std::min(CHUNK_F32, total_read - total_played);
            s16_a.resize(sz);
            for (size_t i = 0; i < sz; ++i) {
                float s = f32_buf[i];
                if (s < -1.0f) s = -1.0f; else if (s > 1.0f) s = 1.0f;
                s16_a[i] = static_cast<int16_t>(s * 32767.0f);
            }
            hdr_a.lpData = reinterpret_cast<LPSTR>(s16_a.data());
            hdr_a.dwBufferLength = static_cast<DWORD>(sz * 2);
            waveOutPrepareHeader(hwo, &hdr_a, sizeof(WAVEHDR));
            waveOutWrite(hwo, &hdr_a, sizeof(WAVEHDR));
            total_played += sz;
        }

        if (started) {
            // Reclaim completed buffer A, refill
            if ((hdr_a.dwFlags & WHDR_DONE) && total_played < total_read) {
                waveOutUnprepareHeader(hwo, &hdr_a, sizeof(WAVEHDR));
                size_t sz = std::min(CHUNK_F32, total_read - total_played);
                s16_a.resize(sz);
                for (size_t i = 0; i < sz; ++i) {
                    float s = f32_buf[total_played + i];
                    if (s < -1.0f) s = -1.0f; else if (s > 1.0f) s = 1.0f;
                    s16_a[i] = static_cast<int16_t>(s * 32767.0f);
                }
                hdr_a = {};
                hdr_a.lpData = reinterpret_cast<LPSTR>(s16_a.data());
                hdr_a.dwBufferLength = static_cast<DWORD>(sz * 2);
                waveOutPrepareHeader(hwo, &hdr_a, sizeof(WAVEHDR));
                waveOutWrite(hwo, &hdr_a, sizeof(WAVEHDR));
                total_played += sz;
            }

            // Reclaim completed buffer B, refill
            if (b_active && (hdr_b.dwFlags & WHDR_DONE) && total_played < total_read) {
                waveOutUnprepareHeader(hwo, &hdr_b, sizeof(WAVEHDR));
                size_t sz = std::min(CHUNK_F32, total_read - total_played);
                s16_b.resize(sz);
                for (size_t i = 0; i < sz; ++i) {
                    float s = f32_buf[total_played + i];
                    if (s < -1.0f) s = -1.0f; else if (s > 1.0f) s = 1.0f;
                    s16_b[i] = static_cast<int16_t>(s * 32767.0f);
                }
                hdr_b = {};
                hdr_b.lpData = reinterpret_cast<LPSTR>(s16_b.data());
                hdr_b.dwBufferLength = static_cast<DWORD>(sz * 2);
                waveOutPrepareHeader(hwo, &hdr_b, sizeof(WAVEHDR));
                waveOutWrite(hwo, &hdr_b, sizeof(WAVEHDR));
                total_played += sz;
            }

            // Start second buffer for overlap
            if (!b_active && total_played > CHUNK_F32 && total_read - total_played >= CHUNK_F32 / 2) {
                size_t sz = std::min(CHUNK_F32, total_read - total_played);
                s16_b.resize(sz);
                for (size_t i = 0; i < sz; ++i) {
                    float s = f32_buf[total_played + i];
                    if (s < -1.0f) s = -1.0f; else if (s > 1.0f) s = 1.0f;
                    s16_b[i] = static_cast<int16_t>(s * 32767.0f);
                }
                hdr_b = {};
                hdr_b.lpData = reinterpret_cast<LPSTR>(s16_b.data());
                hdr_b.dwBufferLength = static_cast<DWORD>(sz * 2);
                waveOutPrepareHeader(hwo, &hdr_b, sizeof(WAVEHDR));
                waveOutWrite(hwo, &hdr_b, sizeof(WAVEHDR));
                b_active = true;
                total_played += sz;
            }

            // Progress
            fprintf(stderr, "pcm_play: %.1f / %.1fs\r",
                    (double)total_played / SAMPLE_RATE,
                    total_read > 0 ? (double)total_read / SAMPLE_RATE : 0.0);
            fflush(stderr);
        }

        Sleep(10);
    }

    // Wait for final buffers
    while (!(hdr_a.dwFlags & WHDR_DONE)) Sleep(5);
    waveOutUnprepareHeader(hwo, &hdr_a, sizeof(WAVEHDR));
    if (b_active) {
        while (!(hdr_b.dwFlags & WHDR_DONE)) Sleep(5);
        waveOutUnprepareHeader(hwo, &hdr_b, sizeof(WAVEHDR));
    }

    waveOutClose(hwo);
    fprintf(stderr, "\npcm_play: done (%.1fs)\n", (double)total_read / SAMPLE_RATE);
    return 0;
}
