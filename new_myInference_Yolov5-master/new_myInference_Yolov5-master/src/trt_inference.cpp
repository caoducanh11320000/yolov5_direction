#include "trt_inference.h"


using namespace IMXAIEngine;
using namespace nvinfer1;

static Logger gLogger;
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

const std::string names[2] = {"basketball", "players"};

cudaStream_t stream;
  // Prepare cpu and gpu buffers
float* gpu_buffers[2];
float* cpu_output_buffer = nullptr;

TRT_Inference::TRT_Inference(){
    printf("khoi tao inference \n");
}

// Newwwwwwwwwwwwwwwww
using namespace std;

/// @brief A ben phia Line am, B ben phia Line duong
#define AB  1
#define BA -1

struct Line {
    float a, b, c; // Phương trình đường thẳng: ax + by + c = 0
};
enum Count_type{
    negative_to_positive =1,
    positive_to_negative,
    both
};

vector<int> ignore_map;
std::map<int, vector<Rect>> tracking_map;
std::map<int, int> car_direction;

static int count_nega_posi = 0;
static int count_posi_nega = 0;
Line line; 
Count_type count_type= both;

Line findLine(float x1, float y1, float x2, float y2) {
    Line line;
    line.a = y2 - y1;
    line.b = x1 - x2;
    line.c = x1 * (y1 - y2) + y1 * (x2 - x1);
    return line;
}

int check_line(Rect position){
    float rs = line.a* position.x + line.b* position.y + line.c;
    if(rs > 0) {
        return 1; 
    }
    return -1; // duoi line return 1, tren line return -1
}

int object_direction(vector<Rect> position_map) {
    int n = position_map.size();
    if(n < 2) return 0;

    int rs = check_line(position_map[0])*check_line(position_map[n-1]);
    if(rs < 0){ // Object vua moi di qua Line
        if(check_line(position_map[0]) <0) return AB;
        return BA;
    }
    else return 0;
}

//////// Enddddddddddddddddddd

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, bool& is_p6, float& gd, float& gw, std::string& img_dir) {
  if (argc < 4) return false;
  if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
    wts = std::string(argv[2]);
    engine = std::string(argv[3]);
    auto net = std::string(argv[4]);
    if (net[0] == 'n') {
      gd = 0.33;
      gw = 0.25;
    } else if (net[0] == 's') {
      gd = 0.33;
      gw = 0.50;
    } else if (net[0] == 'm') {
      gd = 0.67;
      gw = 0.75;
    } else if (net[0] == 'l') {
      gd = 1.0;
      gw = 1.0;
    } else if (net[0] == 'x') {
      gd = 1.33;
      gw = 1.25;
    } else if (net[0] == 'c' && argc == 7) {
      gd = atof(argv[5]);
      gw = atof(argv[6]);
    } 
    else {
      return false;
    }
    if (net.size() == 2 && net[1] == '6') {
      is_p6 = true;
    }
  } else if (std::string(argv[1]) == "-d" && argc == 4) {
    engine = std::string(argv[2]);
    img_dir = std::string(argv[3]);
  } else {
    return false;
  }
  return true;
}

void serialize_engine(unsigned int max_batchsize, bool& is_p6, float& gd, float& gw, std::string& wts_name, std::string& engine_name) {
  // Create builder
  IBuilder* builder = createInferBuilder(gLogger);
  IBuilderConfig* config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an engine
  ICudaEngine *engine = nullptr;
  if (is_p6) {
    engine = build_det_p6_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
  } else {
    engine = build_det_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
  }
  assert(engine != nullptr);

  // Serialize the engine
  IHostMemory* serialized_engine = engine->serialize();
  assert(serialized_engine != nullptr);

  // Save engine to file
  std::ofstream p(engine_name, std::ios::binary);
  if (!p) {
    std::cerr << "Could not open plan output file" << std::endl;
    assert(false);
  }
  p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

  // Close everything down

  delete engine;
  delete config;
  delete serialized_engine;
  delete builder;
}

void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer) {
  assert(engine->getNbBindings() == 2);
  // In order to bind the buffers, we need to know the names of the input and output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int inputIndex = engine->getBindingIndex(kInputTensorName);
  const int outputIndex = engine->getBindingIndex(kOutputTensorName);
  assert(inputIndex == 0);
  assert(outputIndex == 1);
  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, kBatchSize * kOutputSize * sizeof(float)));

  *cpu_output_buffer = new float[kBatchSize * kOutputSize];
}

void infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize) {
  context.enqueue(batchsize, gpu_buffers, stream, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
}

// Thay doi thong so truyen
trt_error TRT_Inference::trt_APIModel(int argc, char** argv){
    
    std::string wts_name = "";
    std::string engine_name = "";
    bool is_p6 = false;
    float gd = 0.0f, gw = 0.0f;
    std::string img_dir;

    if(!parse_args(argc, argv, wts_name, engine_name, is_p6, gd, gw, img_dir)){
        return TRT_RESULT_ERROR;
    }
    
    if (!wts_name.empty()) {
        serialize_engine(kBatchSize, is_p6, gd, gw, wts_name, engine_name);       
    }

    return TRT_RESULT_SUCCESS;
}


// Giu nguyen Thong so
trt_error TRT_Inference::init_inference(std::string engine_name , const char * input_folder, std::vector<std::string> &file_names){
   
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
    std::cerr << "read " << engine_name << " error!" << std::endl;
    assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    // Phan nay co the can sua thanh THis, neu ko chay duoc

    this->runtime = createInferRuntime(gLogger);
    assert(runtime);
    this->engine = (runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(engine);
    this->context = (engine)->createExecutionContext();
    assert(context);
    delete[] serialized_engine;

    // Luu ten anh vao vector
    if (read_files_in_dir(input_folder, file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        //return TRT_RESULT_ERROR;
    }

    // Phan thu nghiem ----------------------------------
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Init CUDA preprocessing
    cuda_preprocess_init(kMaxInputImageSize);

    prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &cpu_output_buffer);

    return TRT_RESULT_SUCCESS;
}


trt_error TRT_Inference::trt_detection(std::vector<IMXAIEngine::trt_input> &trt_inputs, std::vector<IMXAIEngine::trt_output> &trt_outputs){
    
    line = findLine(20, 30, 240, 270); // Xac dinh duong thang

    auto start = std::chrono::system_clock::now();
    std::cout <<" Thuc hien do Inference" << std::endl;
    for (size_t i = 0; i < trt_inputs.size(); i += kBatchSize) {

    // Get a batch of images
    std::vector<cv::Mat> img_batch;  //// day va vector chua anh

    for (size_t j = i; j < i + kBatchSize && j < trt_inputs.size(); j++) {
      //cv::Mat img = cv::imread(img_dir + "/" + file_names[j]);
      cv::Mat img = trt_inputs[j].input_img;
      img_batch.push_back(img);
      //img_name_batch.push_back(file_names[j]);
      std::cout << "Thuc hien thanh cong voi anh: "<< j<<std::endl;
    }
    
    // Preprocess
    std:: cout<<"Preprocess" << std::endl;
    cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);

    // Run inference

    infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer, kBatchSize);
    //auto end = std::chrono::system_clock::now();
    //std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // NMS
    std::vector<std::vector<Detection>> res_batch;
    batch_nms(res_batch, cpu_output_buffer, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);

    // Draw bounding boxes
    //draw_bbox(img_batch, res_batch); 
    // Input 
    for(size_t n =0; n <img_batch.size(); n++){
      trt_inputs[n +i].input_img = img_batch[n];
      trt_inputs[n +i].id_img = n+i;
    }
    //Output
    for (size_t m = 0; m < img_batch.size(); m++) {
      auto& res = res_batch[m];
      cv::Mat img = img_batch[m];
      //cv::Mat img = img_batch[i];
      std::vector<trt_results> image_result;
      IMXAIEngine::trt_output out_img;
      for (size_t j = 0; j < res.size(); j++) {
            trt_results boundingbox_result;
            boundingbox_result.ClassID = res[j].class_id;
            boundingbox_result.Confidence = res[j].conf;
            boundingbox_result.bbox.x = res[j].bbox[0];
            boundingbox_result.bbox.y = res[j].bbox[1];
            boundingbox_result.bbox.weight = res[j].bbox[2];
            boundingbox_result.bbox.hight = res[j].bbox[3];

            //image_result.push_back(boundingbox_result);
            out_img.results.push_back(boundingbox_result);

            boundingbox_result.obj_id = 1; // gan object id mac dinh = 1
            // Count car
            int id = boundingbox_result.obj_id;
            if (find(ignore_map.begin(), ignore_map.end(), id) != ignore_map.end()) continue;

            while(tracking_map[id].size() > 5){
                tracking_map[id].erase(tracking_map[id].begin());
            }
            tracking_map[id].push_back(boundingbox_result.bbox); // push vi tri object vector ( cai chua vi tri di chuyen cua no )
            // Phan nay co the ve luon huong di chuyen neu muon tracking_map[id]: se truy cap dc vao vector chua ca point
            car_direction[id] = object_direction(tracking_map[id]);

            bool ignore_flag = false;
            if(count_type == negative_to_positive){
              if(car_direction[id] == AB) {
                  count_nega_posi ++;
                  ignore_flag = true;
              }
            }
            else if(count_type == positive_to_negative){
              if(car_direction[id] == BA) {
                  count_posi_nega ++;
                  ignore_flag = true;
              }
            }
            else if(count_type == both){
              if(car_direction[id] == AB) {
                  count_nega_posi ++;
                  ignore_flag = true;
              }
              else if(car_direction[id] == BA) {
                  count_posi_nega ++;
                  ignore_flag = true;
              }
            }
            if(ignore_flag){
                // Bo vao ignore_map, xoa direction va tracking
                ignore_map.push_back(id);
                car_direction.erase(id);
                tracking_map.erase(id);
            }

            cv::Rect r = get_rect(img, res[j].bbox);
            cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            std::string name = names[(int)res[j].class_id];
            float confidence = res[j].conf;
            cv::putText(img, name +": "+ std::to_string(confidence)+ " _ID: " + std::to_string(id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 2.5, cv::Scalar(0, 0x00, 255), 2);
            cv::putText(img, "count_nega_posi: "+ std::to_string(count_nega_posi), cv::Point(70, 50), cv::FONT_HERSHEY_PLAIN, 2.5, cv::Scalar(0, 0x00, 255), 2);
            cv::putText(img, "count_posi_nega: "+ std::to_string(count_posi_nega), cv::Point(50, 50), cv::FONT_HERSHEY_PLAIN, 2.5, cv::Scalar(0, 0x00, 255), 2);

            cv::Point point_1 = get_point(img, cv:Point(20, 30)); // Ung voi cac diem di qua Line
            cv::Point point_2 = get_point(img, cv::Point(240, 270));
            cv::line(img, point_1, point_2, cv::Scalar(0, 0, 255), 2);
      }
      // Thêm image_result vào results
      out_img.id= m+i ; /// phan ID nay can xem lai, voi batch size=1 thi dung, con neu batchsize khac thi chua chac
      trt_outputs.push_back(out_img);

    }


    //Save images
    // std::string path = "../images/";
    // for (size_t j = 0; j < img_batch.size(); j++) {
    //   cv::imwrite(path + "player_b4_" + std::to_string(j + i) + ".png", img_batch[j]);  // May be duong dan can thay doi
    // }
    
    }
    auto end = std::chrono::system_clock::now();
    std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    return TRT_RESULT_SUCCESS;
}

trt_error TRT_Inference::trt_release(){
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(gpu_buffers[0]));
    CUDA_CHECK(cudaFree(gpu_buffers[1]));
    delete[] cpu_output_buffer;
    cuda_preprocess_destroy();

    return TRT_RESULT_SUCCESS;
}