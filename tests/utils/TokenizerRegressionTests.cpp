// Regression tests for the server-side BPE tokenizer (P1 prefix-reuse fixes):
//   1. '#'-prefixed merge rules must be loaded; only "#version" is a comment.
//   2. A non-newline whitespace run followed by a non-whitespace character
//      leaves its final character to the next pretokenize branch (the
//      reference `\s+(?!\S)` backtracking), so "   **" keeps a ' ' for ' **'.
// Both bugs previously broke server prefix reuse after long generations:
// re-encoding '###' produced three '#' and '   **' produced 'ĠĠĠ'+'**'.
#include "utils/Tokenizer.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <process.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot write " + path.string());
    }
    out << content;
}

// Byte-level pieces: space -> U+0120 (Ġ); '*'/'-'/'#'/'a' map to themselves.
std::filesystem::path make_fixture() {
    const auto dir = std::filesystem::temp_directory_path() /
                     ("aila-tokenizer-regression-" + std::to_string(::_getpid()));
    std::filesystem::create_directories(dir);
    write_file(dir / "vocab.json",
               "{\"#\":0,\"##\":1,\"###\":2,\"*\":3,\"**\":4,\"-\":5,"
               "\"\u0120\":6,\"\u0120\u0120\":7,\"\u0120\u0120\u0120\":8,"
               "\"\u0120-\":9,\"\u0120**\":10,\"a\":11}");
    write_file(dir / "merges.txt",
               "#version: 0.2\n"
               "# #\n"
               "## #\n"
               "* *\n"
               "\u0120 \u0120\n"
               "\u0120\u0120 \u0120\n"
               "\u0120 -\n");
    return dir;
}

int find_id(const Tokenizer& tokenizer, const std::string& piece) {
    for (int id = 0; id < tokenizer.vocab_size(); ++id) {
        if (tokenizer.raw_token(id) == piece) {
            return id;
        }
    }
    return -1;
}

std::string ids_to_string(const std::vector<int>& ids) {
    std::string out;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) out += ",";
        out += std::to_string(ids[i]);
    }
    return out;
}

} // namespace

int main() {
    int failures = 0;
    const auto check = [&](const char* name, bool ok, const std::string& detail) {
        std::printf("%s %s%s\n", ok ? "PASS" : "FAIL", name,
                    detail.empty() ? "" : (" (" + detail + ")").c_str());
        if (!ok) {
            ++failures;
        }
    };

    std::filesystem::path dir;
    try {
        dir = make_fixture();
        Tokenizer tokenizer;
        if (!tokenizer.load(dir.string())) {
            std::printf("FAIL tokenizer load\n");
            std::filesystem::remove_all(dir);
            return 1;
        }

        const int hash = find_id(tokenizer, "#");
        const int hash2 = find_id(tokenizer, "##");
        const int hash3 = find_id(tokenizer, "###");
        const int star2 = find_id(tokenizer, "**");
        const int space = find_id(tokenizer, "\u0120");
        const int space2 = find_id(tokenizer, "\u0120\u0120");
        const int space3 = find_id(tokenizer, "\u0120\u0120\u0120");
        const int space_dash = find_id(tokenizer, "\u0120-");

        check("vocab fixture", hash == 0 && hash2 == 1 && hash3 == 2 && star2 == 4 &&
                                   space == 6 && space2 == 7 && space3 == 8 && space_dash == 9,
              "ids=" + std::to_string(hash) + "/" + std::to_string(hash2) + "/" +
                  std::to_string(hash3) + "/" + std::to_string(star2));

        // '#'-prefixed merge rules ("# #", "## #") must be loaded.
        const auto triple = tokenizer.encode("###");
        check("encode(\"###\") merges to one token",
              triple == std::vector<int>{hash3}, ids_to_string(triple));

        // "   **" -> ["  ", " **"] -> [ĠĠ, Ġ, **].
        const auto spaces_stars = tokenizer.encode("   **");
        check("encode(\"   **\") keeps final space for punctuation",
              spaces_stars == std::vector<int>{space2, space, star2},
              ids_to_string(spaces_stars));

        // "  -" -> [" ", " -"] -> [Ġ, Ġ-].
        const auto spaces_dash = tokenizer.encode("  -");
        check("encode(\"  -\") keeps final space for punctuation",
              spaces_dash == std::vector<int>{space, space_dash},
              ids_to_string(spaces_dash));

        // Two spaces followed by a letter stay one whitespace token + " a".
        const int letter_a = find_id(tokenizer, "a");
        const auto spaces_letter = tokenizer.encode("  a");
        check("encode(\"  a\") leaves final space for the word",
              spaces_letter == std::vector<int>{space, space, letter_a},
              ids_to_string(spaces_letter));

        std::filesystem::remove_all(dir);
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        if (!dir.empty()) {
            std::filesystem::remove_all(dir);
        }
        return 1;
    }

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("Tokenizer regression checks PASSED\n");
    return 0;
}
