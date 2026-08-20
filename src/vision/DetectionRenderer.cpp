#include "DetectionRenderer.hpp"

#ifdef _WIN32
#include <windows.h>
#include <wincodec.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cwchar>
#include <memory>
#include <vector>
#endif

namespace aila::vision {

#ifdef _WIN32
namespace {

template <typename T> struct ComRelease {
    void operator()(T* value) const { if (value) value->Release(); }
};
template <typename T> using ComPtr = std::unique_ptr<T, ComRelease<T>>;

std::wstring wide(const std::string& value) {
    if (value.empty()) return {};
    const int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                          static_cast<int>(value.size()), nullptr, 0);
    if (count <= 0) return {};
    std::wstring output(static_cast<size_t>(count), L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                            static_cast<int>(value.size()), output.data(), count) != count) return {};
    return output;
}

void fail(std::string* error, const char* message) { if (error) *error = message; }

struct ComApartment {
    bool uninitialize = false;
    ComApartment() : uninitialize(SUCCEEDED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) {}
    ~ComApartment() { if (uninitialize) CoUninitialize(); }
};

} // namespace
#endif

bool save_detection_png(const std::string& input_path, const std::string& output_path,
                        const std::vector<Detection>& detections, std::string* error_message) {
#ifdef _WIN32
    ComApartment apartment;
    IWICImagingFactory* raw_factory = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                  IID_PPV_ARGS(&raw_factory));
    if (FAILED(hr)) { fail(error_message, "WIC factory creation failed"); return false; }
    ComPtr<IWICImagingFactory> factory(raw_factory);
    IWICBitmapDecoder* raw_decoder = nullptr;
    const std::wstring input = wide(input_path);
    hr = factory->CreateDecoderFromFilename(input.c_str(), nullptr, GENERIC_READ,
                                            WICDecodeMetadataCacheOnLoad, &raw_decoder);
    ComPtr<IWICBitmapDecoder> decoder(raw_decoder);
    IWICBitmapFrameDecode* raw_frame = nullptr;
    if (SUCCEEDED(hr)) hr = decoder->GetFrame(0, &raw_frame);
    ComPtr<IWICBitmapFrameDecode> frame(raw_frame);
    IWICBitmapSource* source = frame.get();
    IWICBitmapFlipRotator* raw_rotator = nullptr;
    if (SUCCEEDED(hr)) {
        IWICMetadataQueryReader* raw_metadata = nullptr;
        PROPVARIANT value;
        PropVariantInit(&value);
        if (SUCCEEDED(frame->GetMetadataQueryReader(&raw_metadata)) &&
            SUCCEEDED(raw_metadata->GetMetadataByName(L"/app1/ifd/{ushort=274}", &value)) &&
            value.vt == VT_UI2 && value.uiVal >= 2 && value.uiVal <= 8) {
            static constexpr WICBitmapTransformOptions kExifTransforms[] = {
                WICBitmapTransformRotate0, WICBitmapTransformFlipHorizontal,
                WICBitmapTransformRotate180, WICBitmapTransformFlipVertical,
                static_cast<WICBitmapTransformOptions>(WICBitmapTransformRotate90 |
                                                        WICBitmapTransformFlipHorizontal),
                WICBitmapTransformRotate90,
                static_cast<WICBitmapTransformOptions>(WICBitmapTransformRotate270 |
                                                        WICBitmapTransformFlipHorizontal),
                WICBitmapTransformRotate270,
            };
            if (SUCCEEDED(factory->CreateBitmapFlipRotator(&raw_rotator)) &&
                SUCCEEDED(raw_rotator->Initialize(frame.get(), kExifTransforms[value.uiVal - 1]))) {
                source = raw_rotator;
            }
        }
        PropVariantClear(&value);
        if (raw_metadata) raw_metadata->Release();
    }
    ComPtr<IWICBitmapFlipRotator> rotator(raw_rotator);
    IWICFormatConverter* raw_converter = nullptr;
    if (SUCCEEDED(hr)) hr = factory->CreateFormatConverter(&raw_converter);
    ComPtr<IWICFormatConverter> converter(raw_converter);
    if (SUCCEEDED(hr)) hr = converter->Initialize(source, GUID_WICPixelFormat32bppBGRA,
        WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom);
    UINT width = 0, height = 0;
    if (SUCCEEDED(hr)) hr = converter->GetSize(&width, &height);
    if (FAILED(hr) || width == 0 || height == 0 || width > UINT_MAX / 4 ||
        static_cast<uint64_t>(width) * height * 4 > UINT_MAX) {
        fail(error_message, "input image decode failed"); return false;
    }
    const UINT stride = width * 4;
    std::vector<unsigned char> pixels(static_cast<size_t>(stride) * height);
    hr = converter->CopyPixels(nullptr, stride, static_cast<UINT>(pixels.size()), pixels.data());
    if (FAILED(hr)) { fail(error_message, "input image pixel conversion failed"); return false; }

    BITMAPINFO info{};
    info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    info.bmiHeader.biWidth = static_cast<LONG>(width);
    info.bmiHeader.biHeight = -static_cast<LONG>(height);
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;
    void* dib_pixels = nullptr;
    HDC dc = CreateCompatibleDC(nullptr);
    HBITMAP bitmap = CreateDIBSection(dc, &info, DIB_RGB_COLORS, &dib_pixels, nullptr, 0);
    if (!dc || !bitmap || !dib_pixels) {
        if (bitmap) DeleteObject(bitmap); if (dc) DeleteDC(dc);
        fail(error_message, "annotation surface creation failed"); return false;
    }
    std::memcpy(dib_pixels, pixels.data(), pixels.size());
    HGDIOBJ old_bitmap = SelectObject(dc, bitmap);
    SetBkMode(dc, OPAQUE); SetTextColor(dc, RGB(255, 255, 255)); SetBkColor(dc, RGB(0, 96, 255));
    HPEN pen = CreatePen(PS_SOLID, (std::max)(2, static_cast<int>(width / 320)), RGB(0, 96, 255));
    HGDIOBJ old_pen = SelectObject(dc, pen);
    HGDIOBJ old_brush = SelectObject(dc, GetStockObject(HOLLOW_BRUSH));
    for (const Detection& value : detections) {
        const int left = std::clamp(static_cast<int>(std::lround(value.x1)), 0, static_cast<int>(width));
        const int top = std::clamp(static_cast<int>(std::lround(value.y1)), 0, static_cast<int>(height));
        const int right = std::clamp(static_cast<int>(std::lround(value.x2)), 0, static_cast<int>(width));
        const int bottom = std::clamp(static_cast<int>(std::lround(value.y2)), 0, static_cast<int>(height));
        Rectangle(dc, left, top, right, bottom);
        wchar_t score[32]{};
        swprintf_s(score, L" %.2f", value.confidence);
        const std::wstring label = wide(value.class_name) + score;
        TextOutW(dc, left, (std::max)(0, top - 18), label.c_str(), static_cast<int>(label.size()));
    }
    SelectObject(dc, old_brush); SelectObject(dc, old_pen); SelectObject(dc, old_bitmap);
    DeleteObject(pen); DeleteDC(dc);
    std::memcpy(pixels.data(), dib_pixels, pixels.size());
    DeleteObject(bitmap);

    IWICStream* raw_stream = nullptr;
    const std::wstring output = wide(output_path);
    hr = factory->CreateStream(&raw_stream);
    ComPtr<IWICStream> stream(raw_stream);
    if (SUCCEEDED(hr)) hr = stream->InitializeFromFilename(output.c_str(), GENERIC_WRITE);
    IWICBitmapEncoder* raw_encoder = nullptr;
    if (SUCCEEDED(hr)) hr = factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &raw_encoder);
    ComPtr<IWICBitmapEncoder> encoder(raw_encoder);
    if (SUCCEEDED(hr)) hr = encoder->Initialize(stream.get(), WICBitmapEncoderNoCache);
    IWICBitmapFrameEncode* raw_output_frame = nullptr;
    if (SUCCEEDED(hr)) hr = encoder->CreateNewFrame(&raw_output_frame, nullptr);
    ComPtr<IWICBitmapFrameEncode> output_frame(raw_output_frame);
    if (SUCCEEDED(hr)) hr = output_frame->Initialize(nullptr);
    if (SUCCEEDED(hr)) hr = output_frame->SetSize(width, height);
    WICPixelFormatGUID format = GUID_WICPixelFormat32bppBGRA;
    if (SUCCEEDED(hr)) hr = output_frame->SetPixelFormat(&format);
    if (SUCCEEDED(hr)) hr = output_frame->WritePixels(height, stride, static_cast<UINT>(pixels.size()), pixels.data());
    if (SUCCEEDED(hr)) hr = output_frame->Commit();
    if (SUCCEEDED(hr)) hr = encoder->Commit();
    if (FAILED(hr)) { fail(error_message, "annotated PNG encoding failed"); return false; }
    return true;
#else
    (void)input_path; (void)output_path; (void)detections;
    if (error_message) *error_message = "annotated PNG output requires Windows";
    return false;
#endif
}

} // namespace aila::vision
