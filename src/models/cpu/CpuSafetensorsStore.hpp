#pragma once

#include "CpuTensorView.hpp"
#include "utils/MemoryMappedFile.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class CpuSafetensorsStore {
public:
    bool load_from_dir(const std::string& model_dir, std::string* error);

    bool has(const std::string& name) const;
    const CpuTensorView& get(const std::string& name) const;
    std::vector<std::string> names() const;

private:
    bool load_file(const std::string& path, std::string* error);
    void clear();

    std::vector<std::unique_ptr<MemoryMappedFile>> mapped_files_;
    std::unordered_map<std::string, CpuTensorView> tensors_;
};
