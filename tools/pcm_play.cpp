// Real-time PCM float player for Windows (waveOut double-buffering).
// Reads 24kHz mono f32 PCM from stdin, plays with ~300ms pre-buffer latency.
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

static const int SAMPLE_RATE = 24000;
static const int PREBUFFER_SAMPLES = SAMPLE_RATE * 3 / 10; // 300ms pre-buffer

int main() {
    _setmode(_fileno(stdin), _O_BINARY);

    // Read all PCM from stdin (TTS utterances are typically 1-5 seconds)
    std::vector<float> all_pcm;
    float buf[480]; // read ~20ms at a time
    while (true) {
        size_t n = fread(buf, sizeof(float), 480, stdin);
        if (n == 0) break;
        all_pcm.insert(all_pcm.end(), buf, buf + n);
    }

    size_t total = all_pcm.size();
    if (total == 0) {
        fprintf(stderr, "pcm_play: no input\n");
        return 1;
    }

    // Convert f32 → s16
    std::vector<int16_t> pcm(total);
    for (size_t i = 0; i < total; ++i) {
        float s = all_pcm[i];
        if (s < -1.0f) s = -1.0f; else if (s > 1.0f) s = 1.0f;
        pcm[i] = static_cast<int16_t>(s * 32767.0f);
    }
    all_pcm.clear(); // free f32 memory
    all_pcm.shrink_to_fit();

    WAVEFORMATEX wf = {};
    wf.wFormatTag = WAVE_FORMAT_PCM;
    wf.nChannels = 1;
    wf.nSamplesPerSec = SAMPLE_RATE;
    wf.wBitsPerSample = 16;
    wf.nBlockAlign = 2;
    wf.nAvgBytesPerSec = SAMPLE_RATE * 2;
    wf.cbSize = 0;

    HWAVEOUT hwo = nullptr;
    if (waveOutOpen(&hwo, WAVE_MAPPER, &wf, 0, 0, CALLBACK_NULL) != MMSYSERR_NOERROR) {
        fprintf(stderr, "pcm_play: failed to open audio device\n");
        return 1;
    }

    fprintf(stderr, "pcm_play: %.1fs, playing...\n", (double)total / SAMPLE_RATE);

    // Feed in ~200ms chunks, keep 2 buffers queued for smooth playback
    const size_t CHUNK = SAMPLE_RATE / 5; // 200ms
    size_t offset = 0;

    // Pre-queue first buffer
    size_t first_chunk = std::min(CHUNK + PREBUFFER_SAMPLES, total);
    WAVEHDR hdr1 = {};
    hdr1.lpData = reinterpret_cast<LPSTR>(pcm.data());
    hdr1.dwBufferLength = static_cast<DWORD>(first_chunk * 2);
    waveOutPrepareHeader(hwo, &hdr1, sizeof(WAVEHDR));
    waveOutWrite(hwo, &hdr1, sizeof(WAVEHDR));
    offset = first_chunk;

    WAVEHDR hdr2 = {};
    bool hdr2_active = false;

    while (offset < total || hdr2_active) {
        // Check if hdr1 is done, queue next chunk
        if ((hdr1.dwFlags & WHDR_DONE) && offset < total) {
            waveOutUnprepareHeader(hwo, &hdr1, sizeof(WAVEHDR));
            size_t sz = std::min(CHUNK, total - offset);
            hdr1 = {};
            hdr1.lpData = reinterpret_cast<LPSTR>(pcm.data() + offset);
            hdr1.dwBufferLength = static_cast<DWORD>(sz * 2);
            waveOutPrepareHeader(hwo, &hdr1, sizeof(WAVEHDR));
            waveOutWrite(hwo, &hdr1, sizeof(WAVEHDR));
            offset += sz;
        }

        // Same for hdr2
        if (hdr2_active && (hdr2.dwFlags & WHDR_DONE) && offset < total) {
            waveOutUnprepareHeader(hwo, &hdr2, sizeof(WAVEHDR));
            size_t sz = std::min(CHUNK, total - offset);
            hdr2 = {};
            hdr2.lpData = reinterpret_cast<LPSTR>(pcm.data() + offset);
            hdr2.dwBufferLength = static_cast<DWORD>(sz * 2);
            waveOutPrepareHeader(hwo, &hdr2, sizeof(WAVEHDR));
            waveOutWrite(hwo, &hdr2, sizeof(WAVEHDR));
            offset += sz;
        } else if (!hdr2_active && offset > 0 && offset < total) {
            // Start hdr2
            size_t sz = std::min(CHUNK, total - offset);
            hdr2 = {};
            hdr2.lpData = reinterpret_cast<LPSTR>(pcm.data() + offset);
            hdr2.dwBufferLength = static_cast<DWORD>(sz * 2);
            waveOutPrepareHeader(hwo, &hdr2, sizeof(WAVEHDR));
            waveOutWrite(hwo, &hdr2, sizeof(WAVEHDR));
            hdr2_active = true;
            offset += sz;
        }

        Sleep(10);
    }

    // Wait for final buffers
    while (!(hdr1.dwFlags & WHDR_DONE)) Sleep(10);
    waveOutUnprepareHeader(hwo, &hdr1, sizeof(WAVEHDR));
    if (hdr2_active) {
        while (!(hdr2.dwFlags & WHDR_DONE)) Sleep(10);
        waveOutUnprepareHeader(hwo, &hdr2, sizeof(WAVEHDR));
    }

    waveOutClose(hwo);
    fprintf(stderr, "pcm_play: done\n");
    return 0;
}
