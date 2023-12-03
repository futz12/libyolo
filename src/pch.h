//
// Created by wuyex on 2023/12/3.
//

#ifndef LIBYOLO_PCH_H
#define LIBYOLO_PCH_H

#define NOMINMAX

#include <opencv2/opencv.hpp>
#include <ncnn/net.h>
#include <ncnn/mat.h>
#include <ncnn/gpu.h>
#include <ncnn/cpu.h>
#include <string>
#include <json.hpp>
#include <vector>
#include <algorithm>
#include <memory>

#ifndef DLL_EXPORT
#define LIBYOLO_API extern "C" __declspec(dllexport)
#else
#define LIBYOLO_API extern "C" __declspec(dllimport)
#endif

#endif //LIBYOLO_PCH_H
