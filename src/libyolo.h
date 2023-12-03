//
// Created by wuyex on 2023/12/3.
//

#ifndef LIBYOLO_LIBYOLO_H
#define LIBYOLO_LIBYOLO_H

LIBYOLO_API void* create_detector(const char* cfg_json,const char* param, const unsigned char* bin);
LIBYOLO_API void destroy_detector(void* detector);
LIBYOLO_API void* detect(const char* cfg_json,void* detector, const unsigned char* data,int data_len);
LIBYOLO_API char* get_gpuList();
LIBYOLO_API void destroy_vulkan();

#endif //LIBYOLO_LIBYOLO_H
