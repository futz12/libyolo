//
// Created by wuyex on 2023/12/3.
//

#ifndef LIBYOLO_BASE_ALGORITHM_H
#define LIBYOLO_BASE_ALGORITHM_H

#ifndef DLL_EXPORT
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};
void nms_sorted_bboxes(std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold, bool agnostic);
void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects);
#elif

#endif

#endif //LIBYOLO_BASE_ALGORITHM_H