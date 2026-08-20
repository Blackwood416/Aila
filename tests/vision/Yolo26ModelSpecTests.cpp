#include "utils/ModelSpec.hpp"

#include <filesystem>
#include <fstream>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

constexpr const char* kHash =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

void expect(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

void write_text(const fs::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary);
    output << text;
    if (!output) throw std::runtime_error("failed to write test fixture");
}

std::string config(const std::string& scale, int nc = 2) {
    std::string text =
        "{\"model_type\":\"yolo26\",\"format_version\":1,\"task\":\"detect\","
        "\"scale\":\"" + scale + "\",\"input_width\":640,\"input_height\":640,"
        "\"num_classes\":" + std::to_string(nc) +
        ",\"class_names\":[\"猫\",\"traffic light\"],\"reg_max\":1,"
        "\"end2end\":true,\"dtype\":\"float16\",\"strides\":[8,16,32],"
        "\"topology_sha256\":\"" + std::string(kHash) + "\"}";
    return text;
}

std::string manifest(const std::string& scale, int nc = 2) {
    return "{\"format\":\"aila-yolo26-conversion\",\"format_version\":1,"
           "\"scale\":\"" + scale + "\",\"num_classes\":" + std::to_string(nc) +
           ",\"topology_sha256\":\"" + std::string(kHash) + "\"}";
}

} // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / "aila-yolo26-modelspec-tests";
    fs::remove_all(root);
    fs::create_directories(root);
    try {
        for (const std::string scale : {"n", "s", "m", "l", "x"}) {
            write_text(root / "config.json", config(scale));
            write_text(root / "manifest.json", manifest(scale));
            ModelSpec spec;
            std::string error;
            expect(aila::modelspec::load_from_dir(root.string(), spec, &error), error);
            expect(spec.family == ModelFamily::Yolo26 && spec.yolo26.scale == scale,
                   "scale was not preserved");
            expect(spec.yolo26.class_names == std::vector<std::string>({u8"猫", "traffic light"}),
                   "UTF-8 class names were not preserved");
        }

        for (const auto& replacement : {
                 std::pair{"\"reg_max\":1", "\"reg_max\":2"},
                 std::pair{"\"end2end\":true", "\"end2end\":false"},
                 std::pair{"\"task\":\"detect\"", "\"task\":\"segment\""},
                 std::pair{"\"strides\":[8,16,32]", "\"strides\":[8,16,64]"},
                 std::pair{"\"dtype\":\"float16\"", "\"dtype\":\"float32\""}}) {
            std::string bad = config("n");
            const size_t position = bad.find(replacement.first);
            expect(position != std::string::npos, "invalid fixture replacement");
            bad.replace(position, std::strlen(replacement.first), replacement.second);
            write_text(root / "config.json", bad);
            write_text(root / "manifest.json", manifest("n"));
            ModelSpec spec;
            std::string error;
            expect(!aila::modelspec::load_from_dir(root.string(), spec, &error),
                   "unsupported YOLO26 config was accepted");
        }

        write_text(root / "config.json", config("n"));
        write_text(root / "manifest.json", manifest("s"));
        ModelSpec spec;
        std::string error;
        expect(!aila::modelspec::load_from_dir(root.string(), spec, &error),
               "mismatched manifest was accepted");
        fs::remove_all(root);
        std::cout << "AilaYolo26ModelSpecTests passed\n";
        return 0;
    } catch (const std::exception& error) {
        fs::remove_all(root);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
