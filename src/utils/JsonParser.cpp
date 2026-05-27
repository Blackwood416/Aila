#include "JsonParser.hpp"
#include "Base64.hpp"
#include "simdjson.h"
#include <stdexcept>
#include <iostream>
#include <cctype>

namespace aila {
namespace utils {

bool parse_messages_json(const std::string& messages_json,
                         std::vector<Message>& out_messages,
                         GenerationConfig& out_config,
                         std::string* error_message) {
    auto set_error = [&](const std::string& msg) {
        if (error_message) *error_message = msg;
    };

    out_messages.clear();
    try {
        simdjson::dom::parser parser;
        simdjson::dom::element root = parser.parse(messages_json);
        simdjson::dom::array arr;
        if (root.get_array().get(arr) != simdjson::SUCCESS) {
            simdjson::dom::object obj_root;
            if (root.get_object().get(obj_root) != simdjson::SUCCESS) {
                set_error("messages root is neither array nor object");
                return false;
            }

            simdjson::dom::element msgs_elem;
            if (obj_root.at_key("messages").get(msgs_elem) != simdjson::SUCCESS) {
                set_error("messages field missing in object root");
                return false;
            }
            if (msgs_elem.get_array().get(arr) != simdjson::SUCCESS) {
                set_error("messages field must be an array");
                return false;
            }

            double temp_val = 0.0;
            bool has_temp = false;
            simdjson::dom::element temp_elem;
            if (obj_root.at_key("temperature").get(temp_elem) == simdjson::SUCCESS) {
                if (temp_elem.get_double().get(temp_val) == simdjson::SUCCESS) {
                    out_config.temperature = static_cast<float>(temp_val);
                    has_temp = true;
                }
            }

            bool has_do_sample = false;
            bool do_sample_val = false;
            simdjson::dom::element ds_elem;
            if (obj_root.at_key("do_sample").get(ds_elem) == simdjson::SUCCESS) {
                if (ds_elem.get_bool().get(do_sample_val) == simdjson::SUCCESS) {
                    out_config.do_sample = do_sample_val;
                    has_do_sample = true;
                } else {
                    int64_t ds_int = 0;
                    if (ds_elem.get_int64().get(ds_int) == simdjson::SUCCESS) {
                        out_config.do_sample = (ds_int != 0);
                        has_do_sample = true;
                    }
                }
            }

            if (has_temp && temp_val == 0.0 && !has_do_sample) {
                out_config.do_sample = false;
            }

            simdjson::dom::element top_p_elem;
            if (obj_root.at_key("top_p").get(top_p_elem) == simdjson::SUCCESS) {
                double top_p_val = 0.0;
                if (top_p_elem.get_double().get(top_p_val) == simdjson::SUCCESS) {
                    out_config.top_p = static_cast<float>(top_p_val);
                }
            }

            simdjson::dom::element max_tokens_elem;
            bool got_max_tokens = false;
            if (obj_root.at_key("max_tokens").get(max_tokens_elem) == simdjson::SUCCESS) {
                int64_t mt_val = 0;
                if (max_tokens_elem.get_int64().get(mt_val) == simdjson::SUCCESS) {
                    out_config.max_new_tokens = static_cast<int>(mt_val);
                    got_max_tokens = true;
                }
            }
            if (!got_max_tokens) {
                if (obj_root.at_key("max_new_tokens").get(max_tokens_elem) == simdjson::SUCCESS) {
                    int64_t mt_val = 0;
                    if (max_tokens_elem.get_int64().get(mt_val) == simdjson::SUCCESS) {
                        out_config.max_new_tokens = static_cast<int>(mt_val);
                    }
                }
            }

            simdjson::dom::element seed_elem;
            if (obj_root.at_key("seed").get(seed_elem) == simdjson::SUCCESS) {
                int64_t seed_val = 0;
                if (seed_elem.get_int64().get(seed_val) == simdjson::SUCCESS) {
                    out_config.sampling_seed = static_cast<uint64_t>(seed_val);
                    out_config.use_fixed_seed = true;
                }
            }
        }

        for (auto item : arr) {
            simdjson::dom::object obj;
            if (item.get_object().get(obj) != simdjson::SUCCESS) {
                set_error("message item is not an object");
                return false;
            }

            Message msg;
            {
                simdjson::dom::element role_elem;
                if (obj.at_key("role").get(role_elem) != simdjson::SUCCESS) {
                    set_error("message.role missing");
                    return false;
                }
                std::string_view role_sv;
                if (role_elem.get_string().get(role_sv) != simdjson::SUCCESS) {
                    set_error("message.role must be string");
                    return false;
                }
                msg.role = std::string(role_sv);
            }

            simdjson::dom::element content_elem;
            if (obj.at_key("content").get(content_elem) != simdjson::SUCCESS) {
                set_error("message.content missing");
                return false;
            }

            std::string_view content_str;
            if (content_elem.get_string().get(content_str) == simdjson::SUCCESS) {
                msg.content.push_back(ContentPart{ContentType::Text, std::string(content_str), ""});
            } else {
                simdjson::dom::array content_arr;
                if (content_elem.get_array().get(content_arr) != simdjson::SUCCESS) {
                    set_error("message.content must be string or array");
                    return false;
                }

                for (auto part_elem : content_arr) {
                    simdjson::dom::object part_obj;
                    if (part_elem.get_object().get(part_obj) != simdjson::SUCCESS) {
                        set_error("content part must be object");
                        return false;
                    }

                    simdjson::dom::element type_elem;
                    if (part_obj.at_key("type").get(type_elem) != simdjson::SUCCESS) {
                        set_error("content part.type missing");
                        return false;
                    }
                    std::string_view type_sv;
                    if (type_elem.get_string().get(type_sv) != simdjson::SUCCESS) {
                        set_error("content part.type must be string");
                        return false;
                    }

                    std::string type(type_sv);
                    if (type == "text" || type == "input_text") {
                        simdjson::dom::element text_elem;
                        if (part_obj.at_key("text").get(text_elem) != simdjson::SUCCESS) {
                            set_error("text content missing text field");
                            return false;
                        }
                        std::string_view text_sv;
                        if (text_elem.get_string().get(text_sv) != simdjson::SUCCESS) {
                            set_error("text content.text must be string");
                            return false;
                        }
                        msg.content.push_back(ContentPart{ContentType::Text, std::string(text_sv), ""});
                    } else if (type == "image" || type == "image_url" || type == "input_image") {
                        std::string uri;
                        simdjson::dom::element image_url_elem;
                        if (part_obj.at_key("image_url").get(image_url_elem) == simdjson::SUCCESS) {
                            std::string_view image_url_sv;
                            if (image_url_elem.get_string().get(image_url_sv) == simdjson::SUCCESS) {
                                uri = std::string(image_url_sv);
                            } else {
                                simdjson::dom::object image_obj;
                                if (image_url_elem.get_object().get(image_obj) == simdjson::SUCCESS) {
                                    simdjson::dom::element url_elem;
                                    if (image_obj.at_key("url").get(url_elem) == simdjson::SUCCESS) {
                                        std::string_view url_sv;
                                        if (url_elem.get_string().get(url_sv) == simdjson::SUCCESS) {
                                            uri = std::string(url_sv);
                                        }
                                    }
                                }
                            }
                        }
                        if (uri.empty()) {
                            simdjson::dom::element image_elem;
                            if (part_obj.at_key("image").get(image_elem) == simdjson::SUCCESS) {
                                std::string_view image_sv;
                                if (image_elem.get_string().get(image_sv) == simdjson::SUCCESS) {
                                    uri = std::string(image_sv);
                                }
                            }
                        }
                        if (uri.empty()) {
                            set_error("image content missing image/image_url/url field");
                            return false;
                        }

                        ContentPart part;
                        part.type = ContentType::Image;
                        if (is_data_uri(uri)) {
                            if (!parse_data_uri(uri, part.media_format, part.binary_data)) {
                                set_error("Failed to parse image Base64 data URI");
                                return false;
                            }
                        } else {
                            part.uri = uri;
                            size_t dot = uri.find_last_of('.');
                            if (dot != std::string::npos) {
                                part.media_format = uri.substr(dot + 1);
                            }
                        }
                        msg.content.push_back(std::move(part));

                    } else if (type == "audio" || type == "audio_url" || type == "input_audio") {
                        std::string uri;
                        std::string base64_payload;
                        std::string audio_format = "wav";

                        simdjson::dom::element input_audio_elem;
                        if (part_obj.at_key("input_audio").get(input_audio_elem) == simdjson::SUCCESS) {
                            simdjson::dom::object audio_obj;
                            if (input_audio_elem.get_object().get(audio_obj) == simdjson::SUCCESS) {
                                simdjson::dom::element data_elem;
                                if (audio_obj.at_key("data").get(data_elem) == simdjson::SUCCESS) {
                                    std::string_view data_sv;
                                    if (data_elem.get_string().get(data_sv) == simdjson::SUCCESS) {
                                        base64_payload = std::string(data_sv);
                                    }
                                }
                                simdjson::dom::element fmt_elem;
                                if (audio_obj.at_key("format").get(fmt_elem) == simdjson::SUCCESS) {
                                    std::string_view fmt_sv;
                                    if (fmt_elem.get_string().get(fmt_sv) == simdjson::SUCCESS) {
                                        audio_format = std::string(fmt_sv);
                                    }
                                }
                            }
                        }

                        if (base64_payload.empty()) {
                            simdjson::dom::element audio_url_elem;
                            if (part_obj.at_key("audio_url").get(audio_url_elem) == simdjson::SUCCESS) {
                                std::string_view audio_url_sv;
                                if (audio_url_elem.get_string().get(audio_url_sv) == simdjson::SUCCESS) {
                                    uri = std::string(audio_url_sv);
                                } else {
                                    simdjson::dom::object url_obj;
                                    if (audio_url_elem.get_object().get(url_obj) == simdjson::SUCCESS) {
                                        simdjson::dom::element url_elem;
                                        if (url_obj.at_key("url").get(url_elem) == simdjson::SUCCESS) {
                                            std::string_view url_sv;
                                            if (url_elem.get_string().get(url_sv) == simdjson::SUCCESS) {
                                                uri = std::string(url_sv);
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        ContentPart part;
                        part.type = ContentType::Audio;
                        if (!base64_payload.empty()) {
                            part.binary_data = decode_base64(base64_payload);
                            part.media_format = audio_format;
                            if (part.binary_data.empty()) {
                                set_error("Failed to decode input_audio base64 data");
                                return false;
                            }
                        } else if (!uri.empty()) {
                            if (is_data_uri(uri)) {
                                if (!parse_data_uri(uri, part.media_format, part.binary_data)) {
                                    set_error("Failed to parse audio Data URI");
                                    return false;
                                }
                            } else {
                                part.uri = uri;
                                size_t dot = uri.find_last_of('.');
                                if (dot != std::string::npos) {
                                    part.media_format = uri.substr(dot + 1);
                                }
                            }
                        } else {
                            set_error("audio content missing input_audio/audio_url/url field");
                            return false;
                        }
                        msg.content.push_back(std::move(part));

                    } else if (type == "video" || type == "video_url" || type == "input_video") {
                        std::string uri;
                        simdjson::dom::element video_url_elem;
                        if (part_obj.at_key("video_url").get(video_url_elem) == simdjson::SUCCESS) {
                            std::string_view video_url_sv;
                            if (video_url_elem.get_string().get(video_url_sv) == simdjson::SUCCESS) {
                                uri = std::string(video_url_sv);
                            } else {
                                simdjson::dom::object video_obj;
                                if (video_url_elem.get_object().get(video_obj) == simdjson::SUCCESS) {
                                    simdjson::dom::element url_elem;
                                    if (video_obj.at_key("url").get(url_elem) == simdjson::SUCCESS) {
                                        std::string_view url_sv;
                                        if (url_elem.get_string().get(url_sv) == simdjson::SUCCESS) {
                                            uri = std::string(url_sv);
                                        }
                                    }
                                }
                            }
                        }
                        if (uri.empty()) {
                            simdjson::dom::element video_elem;
                            if (part_obj.at_key("video").get(video_elem) == simdjson::SUCCESS) {
                                std::string_view video_sv;
                                if (video_elem.get_string().get(video_sv) == simdjson::SUCCESS) {
                                    uri = std::string(video_sv);
                                }
                            }
                        }
                        if (uri.empty()) {
                            set_error("video content missing video/video_url/url field");
                            return false;
                        }
                        ContentPart part;
                        part.type = ContentType::Video;
                        part.uri = uri;
                        size_t dot = uri.find_last_of('.');
                        if (dot != std::string::npos) {
                            part.media_format = uri.substr(dot + 1);
                        }
                        msg.content.push_back(std::move(part));
                    } else {
                        set_error("unknown content part type: " + type);
                        return false;
                    }
                }
            }

            out_messages.push_back(std::move(msg));
        }
        return true;
    } catch (const std::exception& e) {
        set_error(std::string("JSON parse failed: ") + e.what());
        return false;
    }
}

} // namespace utils
} // namespace aila
