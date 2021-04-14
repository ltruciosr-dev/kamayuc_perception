#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/flann.hpp>
#include "aruco/markerdetector.h"
#include <typeinfo>
#include <opencv2/aruco.hpp>
#include <string>  
/*
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
*/
#define CL_BLU      cv::Scalar(255, 0,      0   )
#define CL_GREEN    cv::Scalar(0,   255,    0   )
#define CL_RED      cv::Scalar(0,   0,      255 )
#define CL_WHITE    cv::Scalar(255, 255,    255 )

using namespace cv;
using namespace std;
//static const std::string OPENCV_WINDOW = "Image window";

cv::Mat src,dst;
int minSizeMM,thStone,thShadow,removeShadow;
float pix2mm;
static const std::string OPENCV_WINDOW = "stones";

void MorphClose(const cv::Mat &imgIn,cv::Mat &imgOut,int minThickess=2);

void BrightnessAndContrastAuto(const cv::Mat &src, cv::Mat &dst, float clipHistPercent=0)
{

    CV_Assert(clipHistPercent >= 0);
    CV_Assert((src.type() == CV_8UC1) || (src.type() == CV_8UC3) || (src.type() == CV_8UC4));

    int histSize = 256;
    float alpha, beta;
    double minGray = 0, maxGray = 0;

    //to calculate grayscale histogram
    cv::Mat gray;
    if (src.type() == CV_8UC1) gray = src;
    else if (src.type() == CV_8UC3) cvtColor(src, gray, CV_BGR2GRAY);
    else if (src.type() == CV_8UC4) cvtColor(src, gray, CV_BGRA2GRAY);
    if (clipHistPercent == 0)
    {
        // keep full available range
        cv::minMaxLoc(gray, &minGray, &maxGray);
    }
    else
    {
        cv::Mat hist; //the grayscale histogram

        float range[] = { 0, 256 };
        const float* histRange = { range };
        bool uniform = true;
        bool accumulate = false;
        calcHist(&gray, 1, 0, cv::Mat (), hist, 1, &histSize, &histRange, uniform, accumulate);

        // calculate cumulative distribution from the histogram
        std::vector<float> accumulator(histSize);
        accumulator[0] = hist.at<float>(0);
        for (int i = 1; i < histSize; i++)
        {
            accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
        }

        // locate points that cuts at required value
        float max = accumulator.back();
        clipHistPercent *= (max / 100.0); //make percent as absolute
        clipHistPercent /= 2.0; // left and right wings
        // locate left cut
        minGray = 0;
        while (accumulator[minGray] < clipHistPercent)
            minGray++;

        // locate right cut
        maxGray = histSize - 1;
        while (accumulator[maxGray] >= (max - clipHistPercent))
            maxGray--;
    }

    // current range
    float inputRange = maxGray - minGray;

    alpha = (histSize - 1) / inputRange;   // alpha expands current range to histsize range
    beta = -minGray * alpha;             // beta shifts current range so that minGray will go to 0

    // Apply brightness and contrast normalization
    // convertTo operates with saurate_cast
    src.convertTo(dst, -1, alpha, beta);

    // restore alpha channel from source 
    if (dst.type() == CV_8UC4)
    {
        int from_to[] = { 3, 3};
        cv::mixChannels(&src, 4, &dst,1, from_to, 1);
    }
    return;
}

void onStonesTb(int, void*)
{
  cv::Mat blur,bwStones,bwShadow;
  std::vector<std::vector<cv::Point> > contours;
  char buf[80];

  cv::GaussianBlur(src,blur,cv::Size(),2,2);

  // convert to HSV
  cv::Mat src_hsv,brightness,saturation;
  std::vector<cv::Mat> hsv_planes;
  //cvtColor(blur,src_hsv,cv2::COLOR_BGR2HSV);
  cvtColor(blur, src_hsv, CV_BGR2HSV);
  split(src_hsv, hsv_planes);
  saturation = hsv_planes[1];
  brightness = hsv_planes[2];

  int minSizePx = cvRound(minSizeMM/pix2mm);
  int closerSize = minSizePx /2.0;

  //SELECT STONES (INCLUDING SHADOW)
  MorphClose(saturation,saturation,closerSize);
  BrightnessAndContrastAuto(saturation,saturation,1);
  //threshold(saturation,bwStones,thStone,255,THRESH_BINARY_INV); //Get 0..thStone
  threshold(saturation, bwStones,thStone,255,cv::THRESH_BINARY_INV);
  //show the selection
  findContours(bwStones, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
  for (int i = 0; i < contours.size(); i++)
    cv::drawContours(saturation,contours,i,CL_WHITE,2);
  imshow("Threshold Stones+Shadow on Saturation",saturation);

  if(removeShadow)
  {
    //SELECT DARK AREAS (MAYBE SHADOWS)
    MorphClose(brightness,brightness,closerSize);
    BrightnessAndContrastAuto(brightness,brightness,1);
    //cv::threshold(brightness,bwShadow,thShadow,255,THRESH_BINARY); //Get thShadow..255
    threshold(brightness, bwShadow,thShadow,255,cv::THRESH_BINARY);
    //show the selection
    contours.clear();
    findContours(bwShadow, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); i++)
      cv::drawContours(brightness,contours,i,CL_WHITE,2);
    imshow("Threshold Shadow on Brightness",brightness);

    //remove shadows from stones
    cv::bitwise_and(bwStones,bwShadow,bwStones);
  }

  //show the result
  src.copyTo(dst);
  findContours(bwStones, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
  cv::Point2f centroid;
  for (int i = 0; i < contours.size(); i++)
  {

    //draw the contour
    polylines(dst, contours[i], true, CL_GREEN,2);
    //get the bounding rect
    cv::Rect rect = cv::boundingRect(contours[i]);
    //ignore small objects
    if ( max(rect.width,rect.width) < minSizePx)
      continue;

    //calculate moments
    cv::Moments M = cv::moments(contours[i], false);
    //reject if area is 0
    double area = M.m00;
    if (area <= 0)
      continue;

    // OK! THE STONE HAS BEEN SELECTED 
    cv::rectangle(dst,rect,CL_RED,2);
    //get and draw the center
    centroid.x = cvRound(M.m10 / M.m00);
    centroid.y = cvRound(M.m01 / M.m00);
    cv::circle(dst,centroid,3,CL_RED,1);
    sprintf(buf,"%.0fx%.0fmm",rect.width*pix2mm,rect.height*pix2mm);
    cv::putText(dst,buf,rect.tl(),CV_FONT_HERSHEY_PLAIN,1.2,CL_BLU,1,CV_AA);
  }
  imshow(OPENCV_WINDOW,dst);
}

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_; //subscribe to a topic
  image_transport::Publisher image_pub_; //processing or something and then publish it

public:
  ImageConverter()
    : it_(nh_)
  {
    // Subscribe to input video feed and publish output video feed, image topic
    image_sub_ = it_.subscribe("image", 1,
      &ImageConverter::imageCallback, this);
    //image_pub_ = it_.advertise("/image_converter_new2_/output_video2", 1);

    //cv::namedWindow(OPENCV_WINDOW);
  }

  ~ImageConverter()
  {
    //cv::destroyWindow(OPENCV_WINDOW);
  }

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8); //convertion from ROS to CV
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // Draw an example circle on the video stream, later analysis w cv c: now to whatever you want
    //if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60)
    //  cv::circle(cv_ptr->image, cv::Point(50, 50), 10, CV_RGB(255,0,0));
    /*
    cv::Mat idk;
    cv::Mat orig;
    orig = cv_ptr->image;
    idk = cv_ptr->image;
    cv::imshow("coco se quemo", idk);
    char key = (char) cv::waitKey(3);
    if (key == 27)
        exit(1);
    */
    cv::Mat inputImage;
    cv::Mat orig;
    inputImage = cv_ptr->image;
    cv::Mat outputImage = inputImage.clone();
    //cv::imshow("out", outputImage);
    char key = (char) cv::waitKey(3);
    if (key == 27)
        exit(1);

    int knowDistancePX = 170;
    int knowDistanceMM = 150;
    pix2mm = (float)knowDistancePX / knowDistanceMM;

    //Scale down because it's big... let's say to 50%
    float scale = 0.5;
    //resize(orig,src,cv::Size(),scale,scale);
    pix2mm = pix2mm / scale;

    //set defaults
    minSizeMM=20;  // width in millimetre
    thStone=39;    // max saturation for stones
    thShadow=54;   // max brightness for shadow
    removeShadow=1; // try to remove shadows (1=Yes 0=No)

    src = cv_ptr->image;
    cv::imshow(OPENCV_WINDOW,src);
    cv::createTrackbar("Rocks size", OPENCV_WINDOW, &minSizeMM,100, onStonesTb, 0);
    //cv::createTrackbar("ThStone", OPENCV_WINDOW, &thStone, 255, onStonesTb, 0);
    //cv::createTrackbar("ThShadow", OPENCV_WINDOW, &thShadow, 255, onStonesTb, 0);
    //cv::createTrackbar("Remove Shadow?", OPENCV_WINDOW, &removeShadow, 1, onStonesTb, 0);
    onStonesTb(0,0);
    //cv::waitKey(0);

    /*    
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::detectMarkers(inputImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
    cv::Mat outputImage = inputImage.clone();
    
    // if at least one marker detected
    if (markerIds.size() > 0) {
        cv::aruco::drawDetectedMarkers(outputImage, markerCorners, markerIds);
        // draw axis for each marker
        cout << "si detecte creo";
    }
    cout << "no ";
    cv::imshow("out", outputImage);
    char key = (char) cv::waitKey(3);
    if (key == 27)
        exit(1);
    */

    //cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    //cv::waitKey(3);
    //cout << typeid(cv_ptr->image).name() << endl;
    // Output modified video stream
    //Convert this message to a ROS sensor_msgs::Image message. 

    //image_pub_.publish(cv_ptr->toImageMsg()); //from opencv to ros
  }
};
//helper function
//void MorphClose(const cv::Mat &imgIn,cv::Mat &imgOut,int minThickess=2);

// ON TRACK BAR FUNCTION

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter2_new_");
  ImageConverter ic;
  ros::spin();
  return 0;
}

void MorphClose(const cv::Mat &imgIn,cv::Mat &imgOut,int minThickess)
{
  int size = minThickess / 2;
  cv::Point2f anchor = cv::Point(size, size);
  cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * size + 1, 2 * size + 1), anchor);
  morphologyEx(imgIn, imgOut, cv::MORPH_CLOSE, element, anchor);
}