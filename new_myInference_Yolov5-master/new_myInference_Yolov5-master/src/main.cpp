#include <iostream>
#include "trt_inference.h"

using namespace IMXAIEngine;
using namespace nvinfer1;
using namespace std;

std::vector<IMXAIEngine::trt_input> trt_inputs; 
std::vector<IMXAIEngine::trt_output> trt_outputs;

std::string outputVideoPath = "videotest_2.avi";
std::string outputVideoPath2 = "videotest_3.avi";
cv::Size imageSize(640, 480);  // Thay đổi kích thước theo ý muốn

int main(int argc, char** argv) {
    cudaSetDevice(kGpuId);
    TRT_Inference test1;
    std::vector<std::string> file_names;

    if(string( argv[1]) == "-s"){
        test1.trt_APIModel(argc, argv);
        return 0;
    }
    else if (string(argv[1]) == "-d" && string(argv[2])== "-i")
    {
        test1.init_inference(string(argv[3]), argv[4],  file_names);
        cout <<"Dang thuc hien voi anhhhh nha" << endl;

        //Input
        std::string folder= std::string(argv[4]); // luu ten thu muc chua anh
        for(int i=0; i< (int)file_names.size(); i++){

            std::cout << "Thuc hien voi anh:" << i <<std::endl;

            cv::Mat img = cv::imread(folder + "/" + file_names[i]);
            IMXAIEngine:: trt_input trt_input;
            if(!img.empty()) {
                //input_img.push_back(img); // danh so ID o day luon
                trt_input.input_img= img;
                trt_input.id_img = i;
                trt_inputs.push_back(trt_input);
                std::cout<< "thanh cong voi anh" << i <<std::endl;
                }
            else std::cout << "That bai" << std::endl;
        }

        // Dectection
        test1.trt_detection(trt_inputs, trt_outputs);

        // Print Output
        std::cout << "so luong ket qua:" << trt_outputs.size() << std::endl;
        for (int i = 0; i < (int) trt_outputs.size(); i++) 
        {
        auto x = trt_outputs[i];
        std::cout << "ID anh: " <<x.id << std::endl;
        std::cout << x.results.size() << std::endl;
        for (int j = 0; j < (int)x.results.size(); j++)
        {
            std::cout << "Bounding box: " << x.results[j].ClassID<<" ,P= " << x.results[j].Confidence<<" ,bounding box: " << x.results[j].bbox[0]<<" " << x.results[j].bbox[1]<<" " << x.results[j].bbox[2]<<" " << x.results[j].bbox[3] << std::endl;
        }
        }
        // Realease
        test1.trt_release();
        return 1;
    }
    /// Input == Video
    else if(string(argv[1]) == "-d" && string(argv[2]) =="-v" ){
        test1.init_inference(string(argv[3]), argv[4],  file_names);
        std::cout <<"Dang thuc hien voi video" <<std::endl;
        // chuyen video thanh anh
        std::string video_path = std::string(argv[4]);
        cv::VideoCapture cap(video_path);
        if(!cap.isOpened()){
            std::cout <<" Khong the mo video" <<std::endl;
        }    
        int id_img=0;
        IMXAIEngine::trt_input trt_input;

        cv::Mat tmp;
        cap >> tmp;
        int w= cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h= cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        std::cout<<"Chieu dai: " << w << "chieu rong: "<< h <<endl;
        cv::Size imageSize1(w, h);

        cv::VideoWriter videoWriter(outputVideoPath, cv::VideoWriter::fourcc('H', '2', '6', '4'), 25, imageSize);
        cv::VideoWriter videoWriter2(outputVideoPath2, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25, imageSize);

        //Input vaf detect
        while(true){
            cv::Mat img;
            cap >> img;
            if(img.empty()){
                std::cout<<"Het video" <<std::endl;
                break;
            }

            trt_input.id_img= id_img;
            trt_input.input_img= img;
            trt_inputs.push_back(trt_input);

            if((id_img+1) % 48 ==0){
                
                std::cout << "So luong dau vao: " << trt_inputs.size() << std::endl;
                test1.trt_detection(trt_inputs, trt_outputs);
                std::cout << "So luong dau ra: " << trt_outputs.size()<< std::endl;
                
                for(int j=0; j< 48; j++){
                    cv::Mat img1 = trt_inputs[j].input_img;
                    cv::resize(img1, img1, imageSize);                    
                    videoWriter.write(img1);                                                                                 
                }
                trt_inputs.clear();
                std::vector< IMXAIEngine::trt_input> ().swap(trt_inputs) ;
                trt_outputs.clear();
                std::vector< IMXAIEngine::trt_output> ().swap(trt_outputs) ;
            }
            std::cout << "Thuc hien voi anh: " << id_img << std::endl;
            id_img ++;
        }
        // Đóng video writer
        videoWriter.release();
        videoWriter2.release();
        std::cout << "Video đã được tạo thành công." << std::endl;

        test1.trt_release();
        return 1;
    }
    
}
