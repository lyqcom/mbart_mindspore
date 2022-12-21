#include <sys/time.h>
#include <sys/stat.h>
#include <gflags/gflags.h>
#include <dirent.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <iosfwd>
#include <vector>
#include <fstream>
#include <sstream>
#include <memory>

#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/types.h"
#include "include/api/serialization.h"
#include "include/dataset/execute.h"
#include "include/dataset/vision.h"

using mindspore::Context;
using mindspore::Serialization;
using mindspore::Model;
using mindspore::Status;
using mindspore::MSTensor;
using mindspore::dataset::Execute;
using mindspore::ModelType;
using mindspore::GraphCell;
using mindspore::kSuccess;
using mindspore::MSTensor;
using mindspore::DataType;

std::vector<std::string> GetAllFiles(std::string const& dirName);
DIR *OpenDir(std::string const& dirName);
std::string RealPath(std::string const& path);
mindspore::MSTensor ReadFileToTensor(std::string const& file);
int WriteResult(std::string const& imageFile, std::vector<mindspore::MSTensor> const& outputs);

DEFINE_string(mindir_path, "", "mindir path");
DEFINE_string(input0_path, ".", "input0 path");
DEFINE_string(input1_path, ".", "input1 path");
DEFINE_int32(device_id, 0, "device id");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (RealPath(FLAGS_mindir_path).empty()) {
        std::cout << "Invalid mindir" << std::endl;
        return 1;
    }

    auto context = std::make_shared<Context>();
    auto ascend310 = std::make_shared<mindspore::Ascend310DeviceInfo>();
    ascend310->SetDeviceID(FLAGS_device_id);
    context->MutableDeviceInfo().push_back(ascend310);
    mindspore::Graph graph;
    Serialization::Load(FLAGS_mindir_path, ModelType::kMindIR, &graph);

    Model model;
    Status ret = model.Build(GraphCell(graph), context);
    if (ret != kSuccess) {
        std::cout << "ERROR: Build failed." << std::endl;
        return 1;
    }

    std::vector<MSTensor> model_inputs = model.GetInputs();
    if (model_inputs.empty()) {
        std::cout << "Invalid model, inputs is empty." << std::endl;
        return 1;
    }

    auto input0_files = GetAllFiles(FLAGS_input0_path);
    auto input1_files = GetAllFiles(FLAGS_input1_path);

    if (input0_files.empty() || input1_files.empty()) {
        std::cout << "ERROR: input data empty." << std::endl;
        return 1;
    }

    std::map<double, double> costTime_map;
    size_t size = input0_files.size();

    for (size_t i = 0; i < size; ++i) {
        struct timeval start = {0};
        struct timeval end = {0};
        double startTimeMs;
        double endTimeMs;
        std::vector<MSTensor> inputs;
        std::vector<MSTensor> outputs;
        std::cout << "Start predict input files:" << input0_files[i] << std::endl;

        auto input0 = ReadFileToTensor(input0_files[i]);
        auto input1 = ReadFileToTensor(input1_files[i]);
        auto input2 = ReadFileToTensor(input2_files[i]);
        inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                            input0.Data().get(), input0.DataSize());
        inputs.emplace_back(model_inputs[1].Name(), model_inputs[1].DataType(), model_inputs[1].Shape(),
                            input1.Data().get(), input1.DataSize());

        gettimeofday(&start, nullptr);
        ret = model.Predict(inputs, &outputs);
        gettimeofday(&end, nullptr);
        if (ret != kSuccess) {
            std::cout << "Predict " << input0_files[i] << " failed." << std::endl;
            return 1;
        }
        startTimeMs = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
        endTimeMs = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
        costTime_map.insert(std::pair<double, double>(startTimeMs, endTimeMs));
        int rst = WriteResult(input0_files[i], outputs);
        if (rst != 0) {
            std::cout << "write result failed." << std::endl;
            return rst;
        }
    }
    double average = 0.0;
    int inferCount = 0;

    for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
        double diff = 0.0;
        diff = iter->second - iter->first;
        average += diff;
        inferCount++;
    }
    average = average / inferCount;
    std::stringstream timeCost;
    timeCost << "NN inference cost average time: "<< average << " ms of infer_count " << inferCount << std::endl;
    std::cout << "NN inference cost average time: "<< average << "ms of infer_count " << inferCount << std::endl;
    std::string fileName = "./time_Result" + std::string("/test_perform_static.txt");
    std::ofstream fileStream(fileName.c_str(), std::ios::trunc);
    fileStream << timeCost.str();
    return 0;
}


std::vector<std::string> GetAllFiles(std::string const& dirName) {
    struct dirent *filename;
    DIR *dir = OpenDir(dirName);
    if (dir == nullptr) {
        return {};
    }
    std::vector<std::string> res;
    while ((filename = readdir(dir)) != nullptr) {
        std::string dName = std::string(filename->d_name);
        if (dName == "." || dName == ".." || filename->d_type != DT_REG) {
            continue;
        }
        res.emplace_back(std::string(dirName) + "/" + filename->d_name);
    }
    std::sort(res.begin(), res.end());
    for (auto &f : res) {
        std::cout << "image file: " << f << std::endl;
    }
    return res;
}

int WriteResult(std::string const& imageFile, std::vector<mindspore::MSTensor> const& outputs) {
    std::string homePath = "./result_Files";
    const int INVALID_POINTER = -1;
    const int ERROR = -2;
    for (size_t i = 0; i < outputs.size(); ++i) {
        size_t outputSize;
        std::shared_ptr<const void> netOutput;
        netOutput = outputs[i].Data();
        outputSize = outputs[i].DataSize();
        int pos = imageFile.rfind('/');
        std::string fileName(imageFile, pos + 1);
        fileName.replace(fileName.find('.'), fileName.size() - fileName.find('.'), '_' + std::to_string(i) + ".bin");
        std::string outFileName = homePath + "/" + fileName;
        FILE * outputFile = fopen(outFileName.c_str(), "wb");
        if (outputFile == nullptr) {
            std::cout << "open result file " << outFileName << " failed" << std::endl;
            return INVALID_POINTER;
        }
        size_t size = fwrite(netOutput.get(), sizeof(char), outputSize, outputFile);
        if (size != outputSize) {
            fclose(outputFile);
            outputFile = nullptr;
            std::cout << "write result file " << outFileName << " failed, write size[" << size <<
                      "] is smaller than output size[" << outputSize << "], maybe the disk is full." << std::endl;
            return ERROR;
        }
        fclose(outputFile);
        outputFile = nullptr;
    }
    return 0;
}

mindspore::MSTensor ReadFileToTensor(std::string const& file) {
    if (file.empty()) {
        std::cout << "Pointer file is nullptr" << std::endl;
        return mindspore::MSTensor();
    }

    std::ifstream ifs(file);
    if (!ifs.good()) {
        std::cout << "File: " << file << " is not exist" << std::endl;
        return mindspore::MSTensor();
    }

    if (!ifs.is_open()) {
        std::cout << "File: " << file << "open failed" << std::endl;
        return mindspore::MSTensor();
    }

    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    mindspore::MSTensor buffer(file, mindspore::DataType::kNumberTypeUInt8, {static_cast<int64_t>(size)}, nullptr, size);

    ifs.seekg(0, std::ios::beg);
    ifs.read(reinterpret_cast<char *>(buffer.MutableData()), size);
    ifs.close();

    return buffer;
}


DIR *OpenDir(std::string const& dirName) {
    if (dirName.empty()) {
        std::cout << " dirName is null ! " << std::endl;
        return nullptr;
    }
    std::string realPath = RealPath(dirName);
    struct stat s;
    lstat(realPath.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        std::cout << "dirName is not a valid directory !" << std::endl;
        return nullptr;
    }
    DIR *dir;
    dir = opendir(realPath.c_str());
    if (dir == nullptr) {
        std::cout << "Can not open dir " << dirName << std::endl;
        return nullptr;
    }
    std::cout << "Successfully opened the dir " << dirName << std::endl;
    return dir;
}

std::string RealPath(std::string const& path) {
    char realPathMem[PATH_MAX] = {0};
    char *realPathRet = nullptr;
    realPathRet = realpath(path.data(), realPathMem);

    if (realPathRet == nullptr) {
        std::cout << "File: " << path << " is not exist.";
        return "";
    }

    std::string realPath(realPathMem);
    std::cout << path << " realpath is: " << realPath << std::endl;
    return realPath;
}