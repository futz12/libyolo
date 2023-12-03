//
// Created by wuyex on 2023/12/3.
//

#include "pch.h"
#include "libyolo.h"
#include "base_algorithm.h"

LIBYOLO_API void *create_detector(const char *cfg_json, const char *param, const unsigned char *bin) {
    configor::value cfg = configor::json::parse(cfg_json);
    std::string name = cfg["name"];
    std::string version = cfg["version"];
    bool use_gpu = cfg["use_gpu"];\
    int use_cpu_thread = cfg["use_cpu_thread"];
    bool use_fp16_packed = cfg["use_fp16_packed"];
    bool use_fp16_storage = cfg["use_fp16_storage"];
    bool use_fp16_arithmetic = cfg["use_fp16_arithmetic"];
    bool light_mode = cfg["light_mode"];

    ncnn::Net *net = new ncnn::Net();
    net->opt.use_vulkan_compute = use_gpu;
    net->opt.use_fp16_packed = use_fp16_packed;
    net->opt.use_fp16_storage = use_fp16_storage;
    net->opt.use_fp16_arithmetic = use_fp16_arithmetic;
    net->opt.num_threads = use_cpu_thread;
    net->opt.lightmode = light_mode;

    if (use_gpu)
        net->set_vulkan_device(cfg["gpu_device_id"].get<int>());

    net->load_param_mem(param);
    net->load_model(bin);

    return net;
}

LIBYOLO_API void destroy_detector(void *detector) {
    ncnn::Net *net = (ncnn::Net *) detector;
    delete net;
}

LIBYOLO_API void *detect(const char *cfg_json, void *detector, const unsigned char *data, const int data_len) {
    configor::value cfg = configor::json::parse(cfg_json);
    int channels = cfg["channels"];
    int width = cfg["width"];
    int height = cfg["height"];
    int max_stride = cfg["MAX_STRIDE"];
    int pixel_type = cfg["pixels_type"];

    cv::_InputArray image_io(data, data_len);
    cv::Mat image = cv::imdecode(image_io, cv::IMREAD_UNCHANGED);

    ncnn::Net *net = (ncnn::Net *) detector;

    int img_w = image.cols;
    int img_h = image.rows;

    float scale = 1.f;


    int w = img_w;
    int h = img_h;
    if (w > h) {
        scale = (float) width / w;
        w = width;
        h = h * scale;
    } else {
        scale = (float) height / h;
        h = height;
        w = w * scale;
    }


    ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, pixel_type, img_w, img_h, w, h);

    int wpad = (w + max_stride - 1) / max_stride * max_stride - w;
    int hpad = (h + max_stride - 1) / max_stride * max_stride - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT,
                           114.f);

    std::shared_ptr<float[]> norm_vals;

    if (cfg["skip_norm"] == false) {
        if (pixel_type & (ncnn::Mat::PIXEL_GRAY << ncnn::Mat::PIXEL_CONVERT_SHIFT)) {
            norm_vals = std::shared_ptr<float[]>(new float[1]{cfg["norm_vals"][0]});
        } else if (pixel_type & (ncnn::Mat::PIXEL_RGB << ncnn::Mat::PIXEL_CONVERT_SHIFT)) {
            norm_vals = std::shared_ptr<float[]>(
                    new float[3]{cfg["norm_vals"][0], cfg["norm_vals"][1], cfg["norm_vals"][2]});
        } else if (pixel_type & (ncnn::Mat::PIXEL_BGR << ncnn::Mat::PIXEL_CONVERT_SHIFT)) {
            norm_vals = std::shared_ptr<float[]>(
                    new float[3]{cfg["norm_vals"][2], cfg["norm_vals"][1], cfg["norm_vals"][0]});
        } else if (pixel_type & (ncnn::Mat::PIXEL_RGBA << ncnn::Mat::PIXEL_CONVERT_SHIFT)) {
            norm_vals = std::shared_ptr<float[]>(
                    new float[4]{cfg["norm_vals"][0], cfg["norm_vals"][1], cfg["norm_vals"][2], cfg["norm_vals"][3]});
        } else if (pixel_type & (ncnn::Mat::PIXEL_BGRA << ncnn::Mat::PIXEL_CONVERT_SHIFT)) {
            norm_vals = std::shared_ptr<float[]>(
                    new float[4]{cfg["norm_vals"][2], cfg["norm_vals"][1], cfg["norm_vals"][0], cfg["norm_vals"][3]});
        } else {
            throw std::runtime_error("Unsupported pixel type");
        }
    } else {
        norm_vals = std::shared_ptr<float[]>(nullptr);
    }

    std::shared_ptr<float[]> mean_vals;

    if (cfg["skip_mean"] == false) {
        if (pixel_type & (ncnn::Mat::PIXEL_GRAY << ncnn::Mat::PIXEL_CONVERT_SHIFT)) {
            mean_vals = std::shared_ptr<float[]>(new float[1]{cfg["mean_vals"][0]});
        } else if (pixel_type & (ncnn::Mat::PIXEL_RGB << ncnn::Mat::PIXEL_CONVERT_SHIFT)) {
            mean_vals = std::shared_ptr<float[]>(
                    new float[3]{cfg["mean_vals"][0], cfg["mean_vals"][1], cfg["mean_vals"][2]});
        } else if (pixel_type & (ncnn::Mat::PIXEL_BGR << ncnn::Mat::PIXEL_CONVERT_SHIFT)) {
            mean_vals = std::shared_ptr<float[]>(
                    new float[3]{cfg["mean_vals"][2], cfg["mean_vals"][1], cfg["mean_vals"][0]});
        } else if (pixel_type & (ncnn::Mat::PIXEL_RGBA << ncnn::Mat::PIXEL_CONVERT_SHIFT)) {
            mean_vals = std::shared_ptr<float[]>(
                    new float[4]{cfg["mean_vals"][0], cfg["mean_vals"][1], cfg["mean_vals"][2], cfg["mean_vals"][3]});
        } else if (pixel_type & (ncnn::Mat::PIXEL_BGRA << ncnn::Mat::PIXEL_CONVERT_SHIFT)) {
            mean_vals = std::shared_ptr<float[]>(
                    new float[4]{cfg["mean_vals"][2], cfg["mean_vals"][1], cfg["mean_vals"][0], cfg["mean_vals"][3]});
        } else {
            throw std::runtime_error("Unsupported pixel type");
        }
    } else {
        mean_vals = std::shared_ptr<float[]>(nullptr);
    }

    in_pad.substract_mean_normalize(mean_vals.get(), norm_vals.get());

    ncnn::Extractor ex = net->create_extractor();
    ex.input(((std::string) (cfg["input_layer"])).c_str(), in_pad);

    std::vector<Object> proposals;

    for (int i = 0; i < cfg["output_layers"].size(); i++) {
        configor::value output_layer = cfg["output_layers"][i];

        ncnn::Mat out;
        ex.extract(((std::string) output_layer["layer"]).c_str(), out);
        ncnn::Mat anchors(6);

        for (int j = 0; j < 6; j++) {
            anchors[j] = output_layer["anchors"][j];
        }
        std::vector<Object> objects_tmp;

        generate_proposals(anchors, output_layer["stride"], in_pad, out, cfg["prob_threshold"], objects_tmp);
        proposals.insert(proposals.end(), objects_tmp.begin(), objects_tmp.end());
    }

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, cfg["nms_threshold"], cfg["agnostic"]);


    int count = picked.size();

    std::vector<Object> objects;

    objects.resize(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float) (img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float) (img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float) (img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float) (img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

// return as json
    configor::value result;
    for (auto &x: objects) {
        configor::value obj;
        obj["rect"]["x"] = x.rect.x;
        obj["rect"]["y"] = x.rect.y;
        obj["rect"]["width"] = x.rect.width;
        obj["rect"]["height"] = x.rect.height;
        obj["label"] = x.label;
        obj["prob"] = x.prob;
        result["objects"].push_back(obj);
    }
    result["count"] = count;
    std::string ret = configor::json::dump(result);
    char *ret_c = new char[ret.length() + 1];
    strcpy(ret_c, ret.c_str());
    //printf("%s\n",ret_c);
    return ret_c;
}

LIBYOLO_API char *get_gpuList() {
    int count = ncnn::get_gpu_count();
    configor::value result;
    for (int i = 0; i < count; i++) {
        configor::value obj;
        const ncnn::GpuInfo &info = ncnn::get_gpu_info(i);
        obj["id"] = i;
        obj["name"] = info.device_name();
        obj["driver_version"] = info.driver_version();
        result["gpus"].push_back(obj);
    }
    result["count"] = count;
    std::string ret = configor::json::dump(result);
    char *ret_c = new char[ret.length() + 1];
    strcpy(ret_c, ret.c_str());
    //printf("%s\n",ret_c);
    return ret_c;
}

LIBYOLO_API void destroy_vulkan() {
    ncnn::destroy_gpu_instance();
}