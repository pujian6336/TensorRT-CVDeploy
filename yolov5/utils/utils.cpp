#include "utils.h"
#include "logger.h"
#include <fstream>

bool cuda_check(cudaError_t err, const char* call, int iLine, const char* szFile)
{
	if (err != cudaSuccess) {
		sample::gLogInfo
			<< ("CUDA Runtime error %s # %s, code= %s [%d] in file %s:%d", call, cudaGetErrorString(err), cudaGetErrorName(err),
				err, szFile, iLine);
		return false;
	}
	return true;
}

utils::HostTime::HostTime()
{
	t1 = std::chrono::high_resolution_clock::now();
}

float utils::HostTime::getUsedTime()
{
	t2 = std::chrono::high_resolution_clock::now();
	auto time_used =
		std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.f;
	return time_used;  // ms
}
utils::HostTime::~HostTime() {}

utils::CUDATimer::CUDATimer() {
    // 创建CUDA事件
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    // 纪录CUDA事件时间点
    cudaEventRecord(start);
}

float utils::CUDATimer::getUsedTime() {
    // 纪录cuda事件时间点
    cudaEventRecord(end);
    // 等待cuda事件完成
    cudaEventSynchronize(end);
    float total_time;
    // 计算cuda事件的时间间隔
    cudaEventElapsedTime(&total_time, start, end);
    return total_time;
}

utils::CUDATimer::CUDATimer(cudaStream_t stream) {
    // 创建CUDA事件
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    // 纪录CUDA事件时间点
    cudaEventRecord(start, stream);
}

float utils::CUDATimer::getUsedTime(cudaStream_t stream) {
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float total_time;
    cudaEventElapsedTime(&total_time, start, end);
    return total_time;
}

utils::CUDATimer::~CUDATimer() {
    // 销毁cuda事件
    cudaEventDestroy(start);
    cudaEventDestroy(end);
}

void utils::DrawDetection(cv::Mat& img, const std::vector<Detection>& objects, const std::vector<std::string>& classNames)
{
    cv::Scalar color = cv::Scalar(0, 255, 0);
    cv::Point bbox_points[1][4];
    const cv::Point* bbox_point0[1] = { bbox_points[0] };
    int num_points[] = { 4 };
    if (!objects.empty())
    {
        for (auto& box : objects)
        {
            color = utils::Colors::color20[box.class_id % 20];

            cv::rectangle(img, cv::Point(box.bbox.left, box.bbox.top), cv::Point(box.bbox.right, box.bbox.bottom), color, 2, cv::LINE_AA);
            cv::String det_info;
            if (classNames.size() != 0)
            {
                det_info = classNames[box.class_id] + " " + cv::format("%.4f", box.conf);
            }
            else
            {
                det_info = cv::format("%i", box.class_id) + " " + cv::format("%.4f", box.conf);
            }
            // 在方框右上角绘制对应类别的底色
            bbox_points[0][0] = cv::Point(box.bbox.left, box.bbox.top);
            bbox_points[0][1] = cv::Point(box.bbox.left + det_info.size() * 11, box.bbox.top);
            bbox_points[0][2] = cv::Point(box.bbox.left + det_info.size() * 11, box.bbox.top - 15);
            bbox_points[0][3] = cv::Point(box.bbox.left, box.bbox.top - 15);
            cv::fillPoly(img, bbox_point0, num_points, 1, color);
            cv::putText(img, det_info, bbox_points[0][0], cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    }
}

void utils::DrawSegmentation(cv::Mat& img, const std::vector<Detection>& dets, const std::vector<cv::Mat>& masks, const std::vector<std::string>& classNames)
{
    cv::Point bbox_points[1][4];
    const cv::Point* bbox_point0[1] = { bbox_points[0] };
    int num_points[] = { 4 };

    if (!dets.empty())
    {
        for (size_t i = 0; i < dets.size(); i++)
        {
            cv::Scalar color = utils::Colors::color20[dets[i].class_id % 20];

            cv::Mat mask_bgr;

            cv::cvtColor(masks[i], mask_bgr, cv::COLOR_GRAY2BGR);
            mask_bgr.setTo(color, masks[i]);
            cv::addWeighted(mask_bgr, 0.45, img, 1.0, 0., img);

            cv::rectangle(img, cv::Point(dets[i].bbox.left, dets[i].bbox.top), cv::Point(dets[i].bbox.right, dets[i].bbox.bottom), color, 2, cv::LINE_AA);
            cv::String det_info;
            if (classNames.size() != 0)
            {
                det_info = classNames[dets[i].class_id] + " " + cv::format("%.4f", dets[i].conf);
            }
            else
            {
                det_info = cv::format("%i", dets[i].class_id) + " " + cv::format("%.4f", dets[i].conf);
            }

            // 在方框右上角绘制对应类别的底色
            bbox_points[0][0] = cv::Point(dets[i].bbox.left, dets[i].bbox.top);
            bbox_points[0][1] = cv::Point(dets[i].bbox.left + det_info.size() * 11, dets[i].bbox.top);
            bbox_points[0][2] = cv::Point(dets[i].bbox.left + det_info.size() * 11, dets[i].bbox.top - 15);
            bbox_points[0][3] = cv::Point(dets[i].bbox.left, dets[i].bbox.top - 15);
            cv::fillPoly(img, bbox_point0, num_points, 1, color);
            cv::putText(img, det_info, bbox_points[0][0], cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    }
}
void utils::save_txt(const std::vector<std::vector<Detection>>& objects, const std::vector<std::string>& savePath, std::vector<cv::Mat>& imgsBatch)
{
    if (objects.empty()) { return; }

    for (size_t bi = 0; bi < imgsBatch.size(); bi++)
    {
        if (objects[bi].empty()) { continue; }

        int cur_width = imgsBatch[bi].cols;
        int cur_height = imgsBatch[bi].rows;

        std::ofstream file(savePath[bi]);
        if (file.is_open())
        {
            for (auto& box : objects[bi])
            {
                float cx, cy, w, h;
               
                cx = (box.bbox.left + box.bbox.right) / 2 / cur_width;
                cy = (box.bbox.bottom + box.bbox.top) / 2 / cur_height;
                w = (box.bbox.right - box.bbox.left) / cur_width;
                h = (box.bbox.bottom - box.bbox.top) / cur_height;

                file << box.class_id << " " << cx << " " << cy << " " << w << " " << h << " " << box.conf << "\n";
            }
            file.close();
        }
    }
}

void utils::replace_root_extension(std::vector<std::string>& filePath, const std::string& oldPath, const std::string& newPath, const std::string& extension)
{
    std::transform(filePath.begin(), filePath.end(), filePath.begin(), [&](std::string& str)
        {
            size_t pos = str.find(oldPath);
            if (pos != std::string::npos)
            {
                str.replace(pos, oldPath.length(), newPath);
            }

            size_t extensionPos = str.find_last_of(".");
            if (extensionPos != std::string::npos)
            {
                str.replace(extensionPos, str.length() - extensionPos, extension);
            }
            return str;
        });
}

 std::vector<unsigned char> utils::load_engine_model(const std::string& file_name)
{
    std::vector<unsigned char> engine_data;
    std::ifstream engine_file(file_name, std::ios::binary);
    assert(engine_file.is_open() && "Unable to load engine file.");

    engine_file.seekg(0, engine_file.end);
    int length = engine_file.tellg();
    engine_data.resize(length);

    assert(length > 0 && "engine file is empty.");
    engine_file.seekg(0, engine_file.beg);
    engine_file.read(reinterpret_cast<char*>(engine_data.data()), length);
    engine_file.close();
    return engine_data;
}

