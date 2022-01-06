#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
using namespace std;

vector<string> classes;//储存名字的容器
float confThreshold = 0.5;//置信度阈值
float nmsThreshold = 0.4;//非最大抑制阈值
// int inpWidth = 416;//网络输入图片宽度
// int inpHeight = 416;//网络输入图片高度
int inpWidth = 320;
int inpHeight = 320;
//移除低置信度边界框
void postprocess(cv::Mat& frame,const vector<cv::Mat>& out);
//画出预测边界框
void drawPred(int classId,float conf,int left,int top,int right,int bottom,cv::Mat& frame);
//取得输出层的名字
vector<cv::String> getOutputNames(const cv::dnn::Net& net);
int main(int argc, char const *argv[])
{
    //将类名存进容器
    string classesFile = "./coco.names";//coco.names包含80种不同的类名
    ifstream ifs(classesFile.c_str());
    string line;
    while(getline(ifs,line))classes.push_back(line);

    //取得模型的配置和权重文件
    cv::String modelConfiguration = "./model_file/yolov3-tiny.cfg";
    cv::String modelWeights = "./model_file/yolov3-tiny.weights";

    //加载网络
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration,modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);
    // net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    // net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    //打开视频文件或者图形文件或者相机数据流
    string str, outputFile;
    // cv::VideoCapture cap("demo.mp4");
    // cv::VideoWriter video;
    cv::Mat frame, blob;
    //开启摄像头
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    //创建窗口
    // static const string kWinName = "Deep learning object detection in OpenCV";
    // cv::namedWindow(kWinName,cv::WINDOW_AUTOSIZE);

    //处理每帧
    int frame_cnt = 0;
    while(cv::waitKey(1)<0){
        //取每帧图像
        cap >> frame;
        //如果视频播放完则停止程序
        if(frame.empty()){
            break;
        }
        //在dnn中从磁盘加载图片
        cv::dnn::blobFromImage(frame,blob,1/255.0,cv::Size(inpWidth,inpHeight));
        //设置输入网络
        net.setInput(blob);
        //设置输出层
        vector<cv::Mat> outs;//储存识别结果
        net.forward(outs,getOutputNames(net));
        //移除低置信度边界框
        postprocess(frame,outs);
        //显示s延时信息并绘制
        vector<double> layersTimes;
        double freq = cv::getTickFrequency()/1000;
        double t=net.getPerfProfile(layersTimes)/freq;
        string label = cv::format("Infercence time for a frame:%.2f ms",t);
        cv::putText(frame,label,cv::Point(0,15),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,255,255));
        //绘制识别框
        cv::Mat detecteFrame;
        frame.convertTo(detecteFrame,CV_8U);
        // cv::imshow(kWinName,frame);
        cout << "Frame: " << frame_cnt++ << ", time: " << t << "ms" << endl;
        cv::imwrite("output/frame_%d.jpg",frame);
        cv::waitKey(2);
    }
    cap.release();
    return 0;
}
//移除低置信度边界框
void postprocess(cv::Mat& frame,const vector<cv::Mat>& outs){
    vector<int> classIds;//储存识别类的索引
    vector<float> confidences;//储存置信度
    vector<cv::Rect> boxes;//储存边框

    for(size_t i=0;i<outs.size();i++){
        //从网络输出中扫描所有边界框
        //保留高置信度选框
        //目标数据data:x,y,w,h为百分比，x,y为目标中心点坐标
        float* data = (float*)outs[i].data;
        for(int j=0;j<outs[i].rows;j++,data+=outs[i].cols){
            cv::Mat scores = outs[i].row(j).colRange(5,outs[i].cols);
            cv::Point classIdPoint;
            double confidence;//置信度
            //取得最大分数值与索引
            cv::minMaxLoc(scores,0,&confidence,0,&classIdPoint);
            if(confidence>confThreshold){
                int centerX = (int)(data[0]*frame.cols);
                int centerY = (int)(data[1]*frame.rows);
                int width = (int)(data[2]*frame.cols);
                int height = (int)(data[3]*frame.rows);
                int left = centerX-width/2;
                int top = centerY-height/2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }

        }
        
    }

    //低置信度
    vector<int> indices;//保存没有重叠边框的索引
    //该函数用于抑制重叠边框
    cv::dnn::NMSBoxes(boxes,confidences,confThreshold,nmsThreshold,indices);
    for(size_t i=0;i<indices.size();i++){
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        drawPred(classIds[idx],confidences[idx],box.x,box.y,
        box.x+box.width,box.y+box.height,frame);
    }
}

//绘制预测边界框
void drawPred(int classId,float conf,int left,int top,int right,int bottom,cv::Mat& frame){
    //绘制边界框
    cv::rectangle(frame,cv::Point(left,top),cv::Point(right,bottom),cv::Scalar(255,178,50),3);

    string label = cv::format("%.2f",conf);
    if(!classes.empty()){
        CV_Assert(classId < (int)classes.size());
        label = classes[classId]+":"+label;//边框上的类别标签与置信度
    }
    //绘制边界框上的标签
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label,cv::FONT_HERSHEY_SIMPLEX,0.5,1,&baseLine);
    top = max(top,labelSize.height);
    cv::rectangle(frame,cv::Point(left,top-round(1.5*labelSize.height)),cv::Point(left+round(1.5*labelSize.width),top+baseLine),cv::Scalar(255,255,255),cv::FILLED);
    cv::putText(frame, label,cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75,cv::Scalar(0, 0, 0), 1);
}

//从输出层得到名字
vector<cv::String> getOutputNames(const cv::dnn::Net& net){
    static vector<cv::String> names;
    if(names.empty()){
        //取得输出层指标
        vector<int> outLayers = net.getUnconnectedOutLayers();
        vector<cv::String> layersNames = net.getLayerNames();
        //取得输出层名字
        names.resize(outLayers.size());
        for(size_t i =0;i<outLayers.size();i++){
            names[i] = layersNames[outLayers[i]-1];
        }
    }
    return names;
}
