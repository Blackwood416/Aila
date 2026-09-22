#include "engine/Engine.hpp"

int InferenceEngine::getSpeakerId(const std::string& name) const {
        if (name.empty()) return 0;
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(),
            [](unsigned char c) { return std::tolower(c); });
        auto it = model_spec_.qwen3.spk_id.find(lower);
        if (it != model_spec_.qwen3.spk_id.end()) {
            return it->second;
        }
        return 0;
    }

int InferenceEngine::getLanguageId(const std::string& lang) const {
        if (lang.empty()) return 0;
        std::string lower = lang;
        std::transform(lower.begin(), lower.end(), lower.begin(),
            [](unsigned char c) { return std::tolower(c); });
        auto it = model_spec_.qwen3.codec_language_id.find(lower);
        if (it != model_spec_.qwen3.codec_language_id.end()) {
            return it->second;
        }
        return 0;
    }

std::string InferenceEngine::refCacheKey(const std::string& audio_path, int dim) {
        return audio_path + ":dim" + std::to_string(dim);
    }

std::string InferenceEngine::refDiskCachePath(const std::string& audio_path, int dim) const {
        std::string suffix = ".dim" + std::to_string(dim) + ".ref.bin";
        if (!ref_cache_dir_.empty()) {
            size_t sep = audio_path.find_last_of("/\\");
            std::string base = (sep != std::string::npos) ? audio_path.substr(sep + 1) : audio_path;
            return ref_cache_dir_ + "/" + base + suffix;
        }
        return audio_path + suffix;
    }

bool InferenceEngine::lookupRefCache(const std::string& audio_path,
                            std::vector<float>& embedding) {
        return lookupRefCache(audio_path, 1024, embedding)
            || lookupRefCache(audio_path, 2048, embedding);
    }

bool InferenceEngine::lookupRefCache(const std::string& audio_path, int dim_hint,
                            std::vector<float>& embedding) {
        if (dim_hint <= 0) return false;
        std::string key = refCacheKey(audio_path, dim_hint);
        auto mem_it = ref_cache_.find(key);
        if (mem_it != ref_cache_.end()) {
            embedding = mem_it->second;
            return true;
        }
        std::string disk_path = refDiskCachePath(audio_path, dim_hint);
        std::ifstream in(disk_path, std::ios::binary);
        if (in.is_open()) {
            in.seekg(0, std::ios::end);
            size_t sz = in.tellg();
            in.seekg(0, std::ios::beg);
            if (sz >= 4 && (sz - 4) % sizeof(float) == 0) {
                int32_t hdr = 0;
                in.read(reinterpret_cast<char*>(&hdr), 4);
                int dim = static_cast<int>(hdr);
                if (dim == dim_hint && static_cast<size_t>(dim) * sizeof(float) == sz - 4) {
                    embedding.resize(dim);
                    in.read(reinterpret_cast<char*>(embedding.data()), dim * sizeof(float));
                    float norm_sq = 0.0f;
                    for (float v : embedding) norm_sq += v * v;
                    if (norm_sq > 1.0f) {
                        ref_cache_[key] = embedding;
                        return true;
                    }
                    embedding.clear();
                }
            }
        }
        return false;
    }

void InferenceEngine::cacheRefEmbedding(const std::string& audio_path,
                               const std::vector<float>& embedding) {
        if (embedding.empty()) return;
        int dim = static_cast<int>(embedding.size());
        std::string key = refCacheKey(audio_path, dim);
        ref_cache_[key] = embedding;
        std::string disk_path = refDiskCachePath(audio_path, dim);
        std::error_code ec;
        std::filesystem::create_directories(
            std::filesystem::path(disk_path).parent_path(), ec);
        std::ofstream out(disk_path, std::ios::binary);
        if (out.is_open()) {
            int32_t hdr = dim;
            out.write(reinterpret_cast<const char*>(&hdr), 4);
            out.write(reinterpret_cast<const char*>(embedding.data()),
                     embedding.size() * sizeof(float));
        }
    }

std::string InferenceEngine::canonicalAudioHash(const std::string& audio_path) const {
        std::error_code ec;
        auto p = std::filesystem::canonical(audio_path, ec);
        std::string path_str = ec ? audio_path : p.string();
        std::hash<std::string> hasher;
        size_t h = hasher(path_str);
        std::stringstream ss;
        ss << std::hex << h;
        return ss.str();
    }

bool InferenceEngine::getFileMeta(const std::string& path, uint64_t& out_size, int64_t& out_mtime) const {
        std::error_code ec;
        auto fs = std::filesystem::file_size(path, ec);
        if (ec) return false;
        out_size = fs;
        auto lwt = std::filesystem::last_write_time(path, ec);
        if (ec) return false;
        out_mtime = lwt.time_since_epoch().count();
        return true;
    }

bool InferenceEngine::getAudioFileMeta(const std::string& path, uint64_t& out_size, int64_t& out_mtime) const {
        return getFileMeta(path, out_size, out_mtime);
    }

std::string InferenceEngine::refCodesDiskCachePath(const std::string& audio_path) const {
        std::string suffix = ".qwen3tts12hz.codes.bin";
        if (!ref_cache_dir_.empty()) {
            std::string hash_str = canonicalAudioHash(audio_path);
            size_t sep = audio_path.find_last_of("/\\");
            std::string base = (sep != std::string::npos) ? audio_path.substr(sep + 1) : audio_path;
            return ref_cache_dir_ + "/" + base + "_" + hash_str + suffix;
        }
        return audio_path + suffix;
    }

bool InferenceEngine::lookupRefCodesCache(const std::string& audio_path, TTSReferenceCodes& outCodes) {
        uint64_t curr_audio_size = 0;
        int64_t curr_audio_mtime = 0;
        bool has_audio_meta = getFileMeta(audio_path, curr_audio_size, curr_audio_mtime);

        uint64_t curr_tok_size = 0;
        int64_t curr_tok_mtime = 0;
        std::string tok_path = model_dir_ + "/speech_tokenizer/model.safetensors";
        bool has_tok_meta = getFileMeta(tok_path, curr_tok_size, curr_tok_mtime);

        // 1. Check in-memory cache with full metadata validation
        auto mem_it = ref_codes_cache_.find(audio_path);
        if (mem_it != ref_codes_cache_.end()) {
            const auto& item = mem_it->second;
            bool valid = true;
            if (has_audio_meta && (item.source_size != curr_audio_size || item.source_mtime != curr_audio_mtime)) {
                valid = false;
            }
            if (has_tok_meta && (item.tokenizer_size != curr_tok_size || item.tokenizer_mtime != curr_tok_mtime)) {
                valid = false;
            }
            if (valid) {
                outCodes = item.codes;
                return true;
            } else {
                ref_codes_cache_.erase(mem_it);
            }
        }

        // 2. Check disk cache
        std::string disk_path = refCodesDiskCachePath(audio_path);
        std::ifstream in(disk_path, std::ios::binary);
        if (in.is_open()) {
            char magic[4];
            in.read(magic, 4);
            if (std::memcmp(magic, "MIMI", 4) == 0) {
                int32_t version = 0;
                in.read(reinterpret_cast<char*>(&version), 4);
                if (version == 2) {
                    uint64_t cached_audio_size = 0;
                    int64_t cached_audio_mtime = 0;
                    uint64_t cached_tok_size = 0;
                    int64_t cached_tok_mtime = 0;
                    int32_t frames = 0;
                    int32_t codebooks = 0;
                    in.read(reinterpret_cast<char*>(&cached_audio_size), 8);
                    in.read(reinterpret_cast<char*>(&cached_audio_mtime), 8);
                    in.read(reinterpret_cast<char*>(&cached_tok_size), 8);
                    in.read(reinterpret_cast<char*>(&cached_tok_mtime), 8);
                    in.read(reinterpret_cast<char*>(&frames), 4);
                    in.read(reinterpret_cast<char*>(&codebooks), 4);

                    if (has_audio_meta && (cached_audio_size != curr_audio_size || cached_audio_mtime != curr_audio_mtime)) {
                        return false; // Source audio changed on disk
                    }
                    if (has_tok_meta && (cached_tok_size != curr_tok_size || cached_tok_mtime != curr_tok_mtime)) {
                        return false; // Speech tokenizer changed/mismatched
                    }

                    if (frames > 0 && codebooks == 16) {
                        outCodes.frames = frames;
                        outCodes.codebooks = codebooks;
                        outCodes.codes.resize(static_cast<size_t>(frames) * 16);
                        in.read(reinterpret_cast<char*>(outCodes.codes.data()),
                                outCodes.codes.size() * sizeof(int32_t));
                        if (in.gcount() == static_cast<std::streamsize>(outCodes.codes.size() * sizeof(int32_t))) {
                            CachedRefCodes item;
                            item.codes = outCodes;
                            item.source_size = curr_audio_size;
                            item.source_mtime = curr_audio_mtime;
                            item.tokenizer_size = curr_tok_size;
                            item.tokenizer_mtime = curr_tok_mtime;
                            ref_codes_cache_[audio_path] = std::move(item);
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }

void InferenceEngine::cacheRefCodes(const std::string& audio_path, const TTSReferenceCodes& codes) {
        if (codes.frames <= 0 || codes.codebooks != 16) return;

        uint64_t curr_audio_size = 0;
        int64_t curr_audio_mtime = 0;
        getFileMeta(audio_path, curr_audio_size, curr_audio_mtime);

        uint64_t curr_tok_size = 0;
        int64_t curr_tok_mtime = 0;
        std::string tok_path = model_dir_ + "/speech_tokenizer/model.safetensors";
        getFileMeta(tok_path, curr_tok_size, curr_tok_mtime);

        CachedRefCodes item;
        item.codes = codes;
        item.source_size = curr_audio_size;
        item.source_mtime = curr_audio_mtime;
        item.tokenizer_size = curr_tok_size;
        item.tokenizer_mtime = curr_tok_mtime;
        ref_codes_cache_[audio_path] = item;

        std::string disk_path = refCodesDiskCachePath(audio_path);
        std::error_code ec;
        std::filesystem::create_directories(
            std::filesystem::path(disk_path).parent_path(), ec);

        std::string tmp_path = disk_path + ".tmp." + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        {
            std::ofstream out(tmp_path, std::ios::binary);
            if (out.is_open()) {
                const char magic[4] = {'M', 'I', 'M', 'I'};
                int32_t version = 2;
                int32_t frames = codes.frames;
                int32_t codebooks = 16;
                out.write(magic, 4);
                out.write(reinterpret_cast<const char*>(&version), 4);
                out.write(reinterpret_cast<const char*>(&curr_audio_size), 8);
                out.write(reinterpret_cast<const char*>(&curr_audio_mtime), 8);
                out.write(reinterpret_cast<const char*>(&curr_tok_size), 8);
                out.write(reinterpret_cast<const char*>(&curr_tok_mtime), 8);
                out.write(reinterpret_cast<const char*>(&frames), 4);
                out.write(reinterpret_cast<const char*>(&codebooks), 4);
                out.write(reinterpret_cast<const char*>(codes.codes.data()),
                          codes.codes.size() * sizeof(int32_t));
                out.flush();
            } else {
                return;
            }
        }
        std::filesystem::rename(tmp_path, disk_path, ec);
    }
