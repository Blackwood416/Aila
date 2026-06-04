// Streaming PCM float player for Windows (waveOut double-buffering).
// Reads 24kHz mono f32 PCM from stdin, plays in real-time as data arrives.
// Compile: cl /O2 /Fe:pcm_play.exe tools/pcm_play.cpp /link winmm.lib
// Usage: aila --stream-tts ... 2>/dev/null | pcm_play [-b prebuffer_ms]

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <mmsystem.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <io.h>
#include <fcntl.h>

#pragma comment(lib, "winmm.lib")

static const int SAMPLE_RATE = 24000;

int main(int argc, char** argv) {
    _setmode(_fileno(stdin), _O_BINARY);

    int prebuffer_ms = 400;
    if (argc > 1 && strcmp(argv[1], "-b") == 0 && argc > 2) {
        prebuffer_ms = atoi(argv[2]);
    }

    // Audio format: 24kHz mono 16-bit PCM
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

    // Buffer queue: reader thread fills, playback loop drains
    static const int CHUNK_MS = 120;
    static const size_t CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS / 1000; // 2880
    static const int MAX_QUEUED = 6; // ~720ms of audio queued max

    struct Chunk {
        std::vector<int16_t> data;
        bool ready = false;
    };
    std::deque<Chunk> chunks;

    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> input_done{false};
    size_t total_read = 0;
    size_t total_played = 0;

    // Pre-allocate chunks
    for (int i = 0; i < MAX_QUEUED * 2; ++i) {
        Chunk c;
        c.data.resize(CHUNK_SAMPLES);
        chunks.push_back(std::move(c));
    }

    // Reader thread: reads f32 from stdin, converts to s16, fills chunks
    std::thread reader([&]() {
        std::vector<float> fbuf(CHUNK_SAMPLES);
        size_t chunk_idx = 0;

        while (true) {
            size_t n = fread(fbuf.data(), sizeof(float), CHUNK_SAMPLES, stdin);
            if (n == 0) break;

            {
                std::unique_lock<std::mutex> lock(mtx);
                // Find next free chunk (cycling through pool)
                int tries = 0;
                while (chunks[chunk_idx % chunks.size()].ready && tries < 100) {
                    cv.wait_for(lock, std::chrono::milliseconds(20));
                    tries++;
                }
                auto& c = chunks[chunk_idx % chunks.size()];
                c.data.resize(n);
                for (size_t i = 0; i < n; ++i) {
                    float s = fbuf[i];
                    if (s < -1.0f) s = -1.0f; else if (s > 1.0f) s = 1.0f;
                    c.data[i] = static_cast<int16_t>(s * 32767.0f);
                }
                c.ready = true;
                total_read += n;
                chunk_idx++;
            }
            cv.notify_one();
        }

        input_done = true;
        cv.notify_one();
    });

    // Wait for pre-buffer to fill
    fprintf(stderr, "pcm_play: buffering %dms...\r", prebuffer_ms);
    fflush(stderr);
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait_for(lock, std::chrono::milliseconds(prebuffer_ms),
            [&]() { return total_read >= SAMPLE_RATE * prebuffer_ms / 1000 || input_done; });
    }

    if (total_read == 0) {
        fprintf(stderr, "pcm_play: no input\n");
        input_done = true;
        cv.notify_all();
        reader.join();
        waveOutClose(hwo);
        return 1;
    }

    fprintf(stderr, "pcm_play: playing (%.1fs buffered)   \n",
            (double)total_read / SAMPLE_RATE);

    // Playback loop: dequeue ready chunks, send to waveOut, recycle completed
    size_t play_idx = 0;
    std::vector<WAVEHDR> active_hdrs;
    std::vector<size_t> active_indices;
    bool playback_done = false;

    while (!playback_done) {
        // Queue new chunks for playback
        {
            std::unique_lock<std::mutex> lock(mtx);
            while (play_idx < chunks.size() && chunks[play_idx].ready
                   && active_hdrs.size() < 3) {
                auto& c = chunks[play_idx];

                WAVEHDR hdr = {};
                hdr.lpData = reinterpret_cast<LPSTR>(c.data.data());
                hdr.dwBufferLength = static_cast<DWORD>(c.data.size() * 2);
                waveOutPrepareHeader(hwo, &hdr, sizeof(WAVEHDR));
                waveOutWrite(hwo, &hdr, sizeof(WAVEHDR));

                active_hdrs.push_back(hdr);
                active_indices.push_back(play_idx);
                play_idx++;
            }
        }

        // Reclaim completed headers
        for (size_t i = 0; i < active_hdrs.size(); ) {
            if (active_hdrs[i].dwFlags & WHDR_DONE) {
                waveOutUnprepareHeader(hwo, &active_hdrs[i], sizeof(WAVEHDR));
                total_played += active_hdrs[i].dwBufferLength / 2;

                // Mark chunk as free
                size_t idx = active_indices[i];
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    chunks[idx].ready = false;
                }
                cv.notify_one();

                active_hdrs.erase(active_hdrs.begin() + i);
                active_indices.erase(active_indices.begin() + i);
            } else {
                ++i;
            }
        }

        // Check if done
        if (input_done && active_hdrs.empty()) {
            bool all_consumed = true;
            std::lock_guard<std::mutex> lock(mtx);
            for (size_t i = 0; i < chunks.size(); ++i) {
                if (chunks[i].ready) all_consumed = false;
            }
            if (all_consumed) playback_done = true;
        }

        // Progress indicator
        if (total_read > 0 && (total_played % (SAMPLE_RATE * 2) < CHUNK_SAMPLES)) {
            fprintf(stderr, "pcm_play: %.1f / %.1fs\r",
                    (double)total_played / SAMPLE_RATE,
                    (double)total_read / SAMPLE_RATE);
            fflush(stderr);
        }

        if (!playback_done) Sleep(10);
    }

    waveOutClose(hwo);
    reader.join();

    fprintf(stderr, "\npcm_play: done (%.1fs)\n", (double)total_played / SAMPLE_RATE);
    return 0;
}
