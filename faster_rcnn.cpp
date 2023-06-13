//  Created by Sang Kim on 06/12/2023.
//  Copyright © 2023 Sang Kim <sangkv.work@gmail.com>.

#include <iostream>
#include <fstream>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"

#include "scope_guard.hpp"
#include "tf_utils.hpp"

static TF_Graph* graph = nullptr;
static TF_Session* session = nullptr;
static bool modelLoaded = false;
static std::map<int, std::string> labelMap;

std::vector<int64_t> GetTensorShape(TF_Tensor* tensor)
{
    int nDims = TF_NumDims(tensor);
    std::vector<int64_t> dims;
    for (int i=0; i<nDims; i++)
    {
        dims.push_back(TF_Dim(tensor, i));
    }
    return dims;
}

int loadModel(const char* path, TF_Graph* &graph, TF_Session* &session, const double gpu_memory_fraction=0.85f)
{
    graph = tf_utils::LoadGraph(path);
    //SCOPE_EXIT{ tf_utils::DeleteGraph(graph); }; //Auto-delete on scope exit.
    if(graph == nullptr)
    {
        std::cerr<<"Can't load graph\n";
        return -1;
    }

    session = tf_utils::CreateSession(graph, tf_utils::CreateSessionOptions(gpu_memory_fraction));
    //SCOPE_EXIT{ tf_utils::DeleteSession(session); }; // Auto-delete on scope exit.
    if(session == nullptr)
    {
        std::cerr<<"Can't create session\n";
        return -2;
    }

    return 1;
}

int loadModel(const char* path, const double gpu_memory_fraction=0.75f)
{
    if(!modelLoaded)
    {
        int check = loadModel(path, graph, session, gpu_memory_fraction);
        if(check==1)
            modelLoaded = true;
        return check;
    } else
        return 0;
}

std::map<int, std::string> loadLabelMap(const char* path)
{
    std::map<int, std::string> label;

    std::fstream file;
    file.open(path, std::ios::in);
    if (file.is_open())
    {
        std::string sa;
        while (getline(file, sa))
        {
            std::stringstream s(sa);
            int key;
            std::string s2, value;
            int index = 0;
            while (std::getline(s, s2, ':'))
            {
                if (index == 0)
                    key = std::stoi(s2);
                else
                    value = s2;
                index++;
            }
            label[key] = value;
        }
        file.close();
    }
    return label;
}

int object_detection(TF_Graph* &graph, TF_Session* &session, const cv::Mat &image, int objectSize=320, float confThreshold=0.85f)
{
    cv::Mat img = image.clone();
    //cv::resize(img, img, cv::Size(objectSize, objectSize));
    cv::Mat imdraw = img.clone();
    cv::cvtColor(img, img, CV_BGR2RGB);

    /* Tensor input */
    const std::vector<int64_t> input_dims = {1, img.rows, img.cols, img.channels()};
    const std::vector<int8_t> input_vals(img.data, img.data + (img.rows*img.cols*img.channels())); // initialize vector from C-style array
    const std::vector<TF_Tensor*> input_tensors = {tf_utils::CreateTensor(TF_UINT8, input_dims, input_vals)};
    SCOPE_EXIT{ tf_utils::DeleteTensors(input_tensors); }; // Auto-delete on scope exit.
    const std::vector<TF_Output> input_ops = {{TF_GraphOperationByName(graph, "image_tensor"), 0}};

    /* Tensor output */
    const std::vector<TF_Output> out_ops = {{TF_GraphOperationByName(graph, "detection_boxes"), 0},
                                            {TF_GraphOperationByName(graph, "detection_scores"), 0},
                                            {TF_GraphOperationByName(graph, "detection_classes"), 0},
                                            {TF_GraphOperationByName(graph, "num_detections"), 0}};
    std::vector<TF_Tensor*> output_tensors = {nullptr, nullptr, nullptr, nullptr};
    SCOPE_EXIT{ tf_utils::DeleteTensors(output_tensors); }; // Auto-delete on scope exit.

    // Run session
    TF_Code code = tf_utils::RunSession(session, input_ops, input_tensors, out_ops, output_tensors);

    if(code == TF_OK)
    {
        /* Processing Tensor output */
        std::vector<float> boxes = tf_utils::GetTensorData<float>(output_tensors[0]);
        std::vector<int64_t> boxes_dims = GetTensorShape(output_tensors[0]);
        std::vector<float> scores = tf_utils::GetTensorData<float >(output_tensors[1]);
        std::vector<float> classes = tf_utils::GetTensorData<float >(output_tensors[2]);
        std::vector<float> num_detections = tf_utils::GetTensorData<float >(output_tensors[3]);

        int detectionsCount = num_detections.at(0);

        cv::RNG rng(12345);
        for (int i = 0; i < detectionsCount; ++i)
        {
            if (scores[i] > confThreshold)
            {
                float boxClass = classes[i];

                float x1 = float(img.size().width) * boxes[0*boxes_dims[1] + i*boxes_dims[2] + 1];
                float y1 = float(img.size().height) * boxes[0*boxes_dims[1] + i*boxes_dims[2] + 0];

                float x2 = float(img.size().width) * boxes[0*boxes_dims[1] + i*boxes_dims[2] + 3];
                float y2 = float(img.size().height) * boxes[0*boxes_dims[1] + i*boxes_dims[2] + 2];

                cv::Scalar randomColor = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
                cv::rectangle(imdraw, cv::Point(x1, y1), cv::Point(x2, y2), randomColor, 2);
                //cv::rectangle(imdraw, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2);
                std::ostringstream label;
                label<<labelMap[boxClass]<<", confidence: "<<scores[i]*100<<"%";
                cv::putText(imdraw, label.str(), cv::Point(x1, y1), 1, 1.0, randomColor);
            }
        }
        cv::imwrite("data/output.jpg", imdraw);
        cv::imshow("Display window", imdraw);
        cv::waitKey(0);
        cv::destroyAllWindows();

        return 0;
    } else
    {
        std::cout<<"Error code: "<<tf_utils::CodeToString(code)<<"\n";
        return -1;
    }
}

int faster_rcnn(const cv::Mat &image, int objectSize=320, float confThreshold=0.85f)
{
    return object_detection(graph, session, image, objectSize, confThreshold);
}

int main() {
    int check = loadModel("faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb");
    labelMap = loadLabelMap("faster_rcnn_inception_v2_coco_2018_01_28/labels.txt");
    if (check==0 || check==1)
    {
        cv::Mat image = cv::imread("data/input.jpg");
        int status = faster_rcnn(image);
    }

    return 0;
}
