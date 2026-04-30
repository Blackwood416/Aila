#include "Ops.hpp"
#include <algorithm>
#include <cmath>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace ops {

void vision_patchify_rgb_u8(Context& ctx, const uint8_t* rgb_device, Tensor& patches,
                            int width, int height, int patch_size,
                            float mean0, float mean1, float mean2,
                            float std0, float std1, float std2) {
    if (!rgb_device || width <= 0 || height <= 0 || patch_size <= 0) return;

    bf16* patch_ptr = static_cast<bf16*>(patches.data());
    const int patch_grid_w = width / patch_size;
    const int patch_grid_h = height / patch_size;
    const int num_patches = patch_grid_w * patch_grid_h;
    const int patch_area = patch_size * patch_size;
    const int patch_dim = 3 * patch_area;
    const float u8_scale = 1.0f / 255.0f;
    const float inv_std0 = 1.0f / std0;
    const float inv_std1 = 1.0f / std1;
    const float inv_std2 = 1.0f / std2;

    if (patch_size == 16) {
        constexpr int exact_patch_size = 16;
        constexpr int exact_patch_area = exact_patch_size * exact_patch_size;
        constexpr int exact_patch_dim = 3 * exact_patch_area;
        constexpr int wg_size = 128;
        static_assert(exact_patch_area == 256);
        static_assert(exact_patch_dim == 768);

        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(num_patches * wg_size, wg_size),
                [=](sycl::nd_item<1> item) {
                    int patch_idx = item.get_group(0);
                    int lid = item.get_local_id(0);
                    int py = patch_idx / patch_grid_w;
                    int px = patch_idx - py * patch_grid_w;
                    int base_y = py * exact_patch_size;
                    int base_x = px * exact_patch_size;
                    int patch_base = patch_idx * exact_patch_dim;

                    for (int flat = lid; flat < exact_patch_dim; flat += wg_size) {
                        int channel = flat >> 8;
                        int rem = flat & 255;
                        int ky = rem >> 4;
                        int kx = rem & 15;
                        int iy = base_y + ky;
                        int ix = base_x + kx;
                        int rgb_offset = (iy * width + ix) * 3 + channel;
                        float value = static_cast<float>(rgb_device[rgb_offset]) * u8_scale;
                        if (channel == 0) value = (value - mean0) * inv_std0;
                        else if (channel == 1) value = (value - mean1) * inv_std1;
                        else value = (value - mean2) * inv_std2;
                        patch_ptr[patch_base + flat] = bf16(value);
                    }
                });
        });
        return;
    }

    ctx.queue().parallel_for(sycl::range<2>(num_patches, patch_dim),
        [=](sycl::id<2> idx) {
            int patch_idx = idx[0];
            int flat = idx[1];
            int channel = flat / patch_area;
            int rem = flat % patch_area;
            int ky = rem / patch_size;
            int kx = rem % patch_size;

            int py = patch_idx / patch_grid_w;
            int px = patch_idx % patch_grid_w;
            int iy = py * patch_size + ky;
            int ix = px * patch_size + kx;

            int rgb_offset = (iy * width + ix) * 3 + channel;
            float value = static_cast<float>(rgb_device[rgb_offset]) * u8_scale;
            if (channel == 0) value = (value - mean0) * inv_std0;
            else if (channel == 1) value = (value - mean1) * inv_std1;
            else value = (value - mean2) * inv_std2;
            patch_ptr[patch_idx * patch_dim + flat] = bf16(value);
        });
}

void vision_add_position_embedding(Context& ctx, Tensor& tokens, Tensor& pos_embed,
                                   int patch_grid_w, int patch_grid_h, int hidden_size) {
    if (patch_grid_w <= 0 || patch_grid_h <= 0 || hidden_size <= 0) return;

    bf16* token_ptr = static_cast<bf16*>(tokens.data());
    int num_patches = patch_grid_w * patch_grid_h;
    int pos_len = static_cast<int>(pos_embed.numel() / hidden_size);
    if (pos_len <= 0) return;

    auto launch_exact_add = [&](auto* pe_ptr) {
        int n = std::min(pos_len, num_patches);
        ctx.queue().parallel_for(sycl::range<2>(n, hidden_size),
            [=](sycl::id<2> idx) {
                int token_idx = idx[0];
                int dim = idx[1];
                int offset = token_idx * hidden_size + dim;
                token_ptr[offset] = bf16(static_cast<float>(token_ptr[offset]) +
                                         static_cast<float>(pe_ptr[offset]));
            });
    };

    if (pos_len == num_patches) {
        if (pos_embed.dtype() == dnnl::memory::data_type::f32) {
            launch_exact_add(static_cast<float*>(pos_embed.data()));
        } else {
            launch_exact_add(static_cast<bf16*>(pos_embed.data()));
        }
        return;
    }

    int base_side = static_cast<int>(std::round(std::sqrt(static_cast<float>(pos_len))));
    if (base_side * base_side != pos_len) {
        if (pos_embed.dtype() == dnnl::memory::data_type::f32) {
            launch_exact_add(static_cast<float*>(pos_embed.data()));
        } else {
            launch_exact_add(static_cast<bf16*>(pos_embed.data()));
        }
        return;
    }

    if (pos_embed.dtype() == dnnl::memory::data_type::f32) {
        float* pe_ptr = static_cast<float*>(pos_embed.data());
        ctx.queue().parallel_for(sycl::range<2>(num_patches, hidden_size),
            [=](sycl::id<2> idx) {
                int token_idx = idx[0];
                int dim = idx[1];
                int y = token_idx / patch_grid_w;
                int x = token_idx % patch_grid_w;

                float fy = 0.0f;
                if (patch_grid_h > 1) {
                    fy = static_cast<float>(y) * static_cast<float>(base_side - 1) /
                         static_cast<float>(patch_grid_h - 1);
                }
                int y0 = sycl::clamp(static_cast<int>(sycl::floor(fy)), 0, base_side - 1);
                int y1 = sycl::min(y0 + 1, base_side - 1);
                float wy1 = fy - static_cast<float>(y0);
                float wy0 = 1.0f - wy1;

                float fx = 0.0f;
                if (patch_grid_w > 1) {
                    fx = static_cast<float>(x) * static_cast<float>(base_side - 1) /
                         static_cast<float>(patch_grid_w - 1);
                }
                int x0 = sycl::clamp(static_cast<int>(sycl::floor(fx)), 0, base_side - 1);
                int x1 = sycl::min(x0 + 1, base_side - 1);
                float wx1 = fx - static_cast<float>(x0);
                float wx0 = 1.0f - wx1;

                int id00 = (y0 * base_side + x0) * hidden_size + dim;
                int id01 = (y0 * base_side + x1) * hidden_size + dim;
                int id10 = (y1 * base_side + x0) * hidden_size + dim;
                int id11 = (y1 * base_side + x1) * hidden_size + dim;
                float value = wy0 * (wx0 * pe_ptr[id00] + wx1 * pe_ptr[id01]) +
                              wy1 * (wx0 * pe_ptr[id10] + wx1 * pe_ptr[id11]);
                int out_offset = token_idx * hidden_size + dim;
                token_ptr[out_offset] = bf16(static_cast<float>(token_ptr[out_offset]) + value);
            });
    } else {
        bf16* pe_ptr = static_cast<bf16*>(pos_embed.data());
        ctx.queue().parallel_for(sycl::range<2>(num_patches, hidden_size),
            [=](sycl::id<2> idx) {
                int token_idx = idx[0];
                int dim = idx[1];
                int y = token_idx / patch_grid_w;
                int x = token_idx % patch_grid_w;

                float fy = 0.0f;
                if (patch_grid_h > 1) {
                    fy = static_cast<float>(y) * static_cast<float>(base_side - 1) /
                         static_cast<float>(patch_grid_h - 1);
                }
                int y0 = sycl::clamp(static_cast<int>(sycl::floor(fy)), 0, base_side - 1);
                int y1 = sycl::min(y0 + 1, base_side - 1);
                float wy1 = fy - static_cast<float>(y0);
                float wy0 = 1.0f - wy1;

                float fx = 0.0f;
                if (patch_grid_w > 1) {
                    fx = static_cast<float>(x) * static_cast<float>(base_side - 1) /
                         static_cast<float>(patch_grid_w - 1);
                }
                int x0 = sycl::clamp(static_cast<int>(sycl::floor(fx)), 0, base_side - 1);
                int x1 = sycl::min(x0 + 1, base_side - 1);
                float wx1 = fx - static_cast<float>(x0);
                float wx0 = 1.0f - wx1;

                int id00 = (y0 * base_side + x0) * hidden_size + dim;
                int id01 = (y0 * base_side + x1) * hidden_size + dim;
                int id10 = (y1 * base_side + x0) * hidden_size + dim;
                int id11 = (y1 * base_side + x1) * hidden_size + dim;
                float value = wy0 * (wx0 * static_cast<float>(pe_ptr[id00]) + wx1 * static_cast<float>(pe_ptr[id01])) +
                              wy1 * (wx0 * static_cast<float>(pe_ptr[id10]) + wx1 * static_cast<float>(pe_ptr[id11]));
                int out_offset = token_idx * hidden_size + dim;
                token_ptr[out_offset] = bf16(static_cast<float>(token_ptr[out_offset]) + value);
            });
    }
}

void vision_reorder_merge_blocks(Context& ctx, Tensor& src, Tensor& dst,
                                 int grid_w, int grid_h, int hidden_size, int merge_size) {
    if (grid_w <= 0 || grid_h <= 0 || hidden_size <= 0 || merge_size <= 0) return;

    bf16* src_ptr = static_cast<bf16*>(src.data());
    bf16* dst_ptr = static_cast<bf16*>(dst.data());
    int num_tokens = grid_w * grid_h;
    int block_size = merge_size * merge_size;
    int blocks_per_row = grid_w / merge_size;

    ctx.queue().parallel_for(sycl::range<2>(num_tokens, hidden_size),
        [=](sycl::id<2> idx) {
            int out_idx = idx[0];
            int dim = idx[1];
            int block_idx = out_idx / block_size;
            int in_block = out_idx % block_size;
            int block_y = block_idx / blocks_per_row;
            int block_x = block_idx % blocks_per_row;
            int dy = in_block / merge_size;
            int dx = in_block % merge_size;
            int sy = block_y * merge_size + dy;
            int sx = block_x * merge_size + dx;
            int src_idx = sy * grid_w + sx;
            dst_ptr[out_idx * hidden_size + dim] = src_ptr[src_idx * hidden_size + dim];
        });
}

void vision_mrope_inplace(Context& ctx, Tensor& x,
                          int num_tokens, int num_heads, int head_dim,
                          const int* pos_y, const int* pos_x) {
    if (num_tokens <= 0 || num_heads <= 0 || head_dim <= 0 || !pos_y || !pos_x) return;

    bf16* x_ptr = static_cast<bf16*>(x.data());
    int n_dims = head_dim / 2;
    if (n_dims <= 0) return;

    const int sections0 = head_dim / 4;
    const int sections1 = head_dim / 4;
    const int sections2 = head_dim / 4;
    const int sections3 = head_dim - sections0 - sections1 - sections2;
    const int sec_w = sections0 + sections1;
    const int sec_e = sec_w + sections2;
    const float theta_scale = std::pow(10000.0f, -2.0f / static_cast<float>(n_dims));
    const int sect_dims = sections0 + sections1 + sections2 + sections3;

    ctx.queue().parallel_for(sycl::range<2>(num_tokens, num_heads * n_dims),
        [=](sycl::id<2> idx) {
            int token_idx = idx[0];
            int flat = idx[1];
            int head = flat / n_dims;
            int pair_idx = flat % n_dims;
            int sector = sect_dims > 0 ? (pair_idx % sect_dims) : 0;

            float angle = 0.0f;
            if (sector < sections0) {
                angle = static_cast<float>(pos_y[token_idx]) *
                        sycl::pow(theta_scale, static_cast<float>(sector));
            } else if (sector < sec_w) {
                angle = static_cast<float>(pos_x[token_idx]) *
                        sycl::pow(theta_scale, static_cast<float>(sector - sections0));
            } else if (sector < sec_e) {
                angle = static_cast<float>(pos_y[token_idx]) *
                        sycl::pow(theta_scale, static_cast<float>(sector - sec_w));
            } else {
                angle = static_cast<float>(pos_x[token_idx]) *
                        sycl::pow(theta_scale, static_cast<float>(sector - sec_e));
            }

            int base = token_idx * num_heads * head_dim + head * head_dim;
            float c = sycl::cos(angle);
            float s = sycl::sin(angle);
            float x0 = static_cast<float>(x_ptr[base + pair_idx]);
            float x1 = static_cast<float>(x_ptr[base + pair_idx + n_dims]);
            x_ptr[base + pair_idx] = bf16(x0 * c - x1 * s);
            x_ptr[base + pair_idx + n_dims] = bf16(x0 * s + x1 * c);
        });
}

} // namespace ops
