#include <opencv2/opencv.hpp>
#include "opencv2/opencv.hpp"
#include <opencv/cv.h>
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <limits.h>
#include <math.h>


using namespace std;
using namespace cv;

int Y_MIN = 0;
int Y_MAX = 255;
int Cr_MIN = 133;
int Cr_MAX = 173;
int Cb_MIN = 77;
int Cb_MAX = 127;

IplImage* imfill(IplImage* src)
{
    CvScalar white = CV_RGB( 255, 255, 255 );

    IplImage* dst = cvCreateImage( cvGetSize(src), 8, 3);
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* contour = 0;

    cvFindContours(src, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
    cvZero( dst );

    for( ; contour != 0; contour = contour->h_next )
    {
        cvDrawContours( dst, contour, white, white, 0, CV_FILLED);
    }

    IplImage* bin_imgFilled = cvCreateImage(cvGetSize(src), 8, 1);
    cvInRangeS(dst, white, white, bin_imgFilled);

    return bin_imgFilled;
}
bool experimentalconditions(Vec3b rgbpixel,Vec3f ycbcrpixel,Vec3b hsvpixel)
{
int B=rgbpixel.val[0];
int G=rgbpixel.val[1];
int R=rgbpixel.val[2];
int H=hsvpixel.val[0];
int S=hsvpixel.val[1];
int V=hsvpixel.val[2];
int Y=ycbcrpixel.val[0];
int CR=ycbcrpixel.val[1];
int CB=ycbcrpixel.val[2];
bool cond1 = (R>50) && (G>40) && (B>20) && (R-G >= 10) && (R>G) && (R>B) && ((max(max(R,G),B)-min(min(R,G),B))>10);
bool cond2 =(R>220) && (G>210) && (B>170) && (abs(R-G)<=15) && (R>B) && (G>B);
bool cond3 = (CB>=77) && (CB<=130);
bool cond4 = (CR>=133) && (CR<=173);
bool cond5 = (H>=0) && (H<=50);
bool cond6 = (S>=0.1) && (S<=0.9);
// return (cond1 || cond2) && cond3 && cond4 && cond5 && cond6;
return (cond1 || cond2) && cond3 && cond4;
// return true;
}

Mat getSkin(Mat input)
{
    Mat hsvimg,ycrcbimg;
    // cvtColor(input, skin, CV_RGB2GRAY);
    // // then adjust the threshold to actually make it binary
    // threshold(skin, skin, 100, 255, CV_THRESH_BINARY);

    //Grayscale matrix
    cv::Mat grayscaleMat (input.size(), CV_8U);

    //Convert BGR to Gray
    cv::cvtColor( input, grayscaleMat, CV_BGR2GRAY );

    //Binary image
    cv::Mat skin(grayscaleMat.size(), grayscaleMat.type());

    //Apply thresholding
    cv::threshold(grayscaleMat, skin, 100, 255, cv::THRESH_BINARY);
    for (int row = 0; row < input.rows; row++)
    {
        for (int col = 0; col < input.cols; col++)
        {
            skin.at<uchar>(row,col)=0;      
        }
    }
    // convert our RGB image to YCrCb
    cvtColor(input,ycrcbimg,cv::COLOR_BGR2YCrCb);
    // convert our RGB image to YCrCb
    cvtColor(input,hsvimg,CV_BGR2HSV);
    // uncomment the following line to see the image in YCrCb Color Space and hsvimg
    // imshow("YCrCb Color Space",hsvimg);
    // imshow("HSV Color Space",skin);
    // filter the image using experimentally found values
    // cv::inRange(skin,cv::Scalar(Y_MIN,Cr_MIN,Cb_MIN),cv::Scalar(Y_MAX,Cr_MAX,Cb_MAX),skin);
    for (int row = 0; row < input.rows; row++)
    {
        for (int col = 0; col < input.cols; col++)
        {
            Vec3f ycbcrpixel = ycrcbimg.at<Vec3b>(row, col);
            Vec3b rgbpixel=input.at<Vec3b>(row, col);
            Vec3b hsvpixel=hsvimg.at<Vec3b>(row, col);

            if(experimentalconditions(rgbpixel,ycbcrpixel,hsvpixel))
                {
                    skin.at<uchar>(row,col)=255;
                }            
        }
    }

    // cv::Mat skin;
    // //first convert our RGB image to YCrCb
    // cv::cvtColor(input,skin,cv::COLOR_BGR2YCrCb);
    // //uncomment the following line to see the image in YCrCb Color Space
    // cv::imshow("YCrCb Color Space",skin);
    // //filter the image in YCrCb color space
    // cv::inRange(skin,cv::Scalar(Y_MIN,Cr_MIN,Cb_MIN),cv::Scalar(Y_MAX,Cr_MAX,Cb_MAX),skin);
    // for (int row = 0; row < skin.rows; row++)
    // {
    //     for (int col = 0; col < skin.cols; col++)
    //     { 
    //         int pix=(int)skin.at<uchar>(row,col);
    //         cout<<pix<<endl; 
    //     }
    // }
    return skin;
}

void FindBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs)
{
    blobs.clear();

    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground

    cv::Mat label_image;
    binary.convertTo(label_image, CV_32SC1);

    int label_count = 2; // starts at 2 because 0,1 are used already

    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);
        for(int x=0; x < label_image.cols; x++) {
            if(row[x] != 1) {
                continue;
            }

            cv::Rect rect;
            cv::floodFill(label_image, cv::Point(x,y), label_count, &rect, 0, 0, 4);

            std::vector <cv::Point2i> blob;

            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                int *row2 = (int*)label_image.ptr(i);
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if(row2[j] != label_count) {
                        continue;
                    }

                    blob.push_back(cv::Point2i(j,i));
                }
            }

            blobs.push_back(blob);

            label_count++;
        }
    }
}

void cvFillHoles(cv::Mat &input)
{
    //assume input is uint8 B & W (0 or 1)
    //this function imitates imfill(image,'hole')
    //cv::Mat result=input.clone();
    cv::Mat holes=input.clone();
    cv::floodFill(holes,cv::Point2i(0,0),cv::Scalar(1));
    for(int i=0;i<input.rows*input.cols;i++)
    {
        if(holes.data[i]==0)
            input.data[i]=1;
    }
    return ;
}

Mat drawrectangle(Mat img,vector<vector<Point2i> >faceblobs)
{
        Mat output = img.clone();
        for(int i=0;i<faceblobs.size();i++)
        {
        int xmin=INT_MAX,ymin=INT_MAX,xmax=INT_MIN,ymax=INT_MIN;
        for(size_t j=0; j < faceblobs[i].size(); j++) {
            int x = faceblobs[i][j].x;
            int y = faceblobs[i][j].y;
            if(x<xmin){xmin=x;}
            if(x>xmax){xmax=x;}
            if(y<ymin){ymin=y;}
            if(y>ymax){ymax=y;} 
        }
       // cout << xmin << xmax << ymin << ymax;
        for(int x=0;x<img.cols;x++)
        {
            for(int y=0;y<img.rows;y++)
            {
                if(x==xmin && y<=ymax && y>=ymin)
                {
                    output.at<cv::Vec3b>(y,x)[0] = 0;
                    output.at<cv::Vec3b>(y,x)[1] = 0;
                    output.at<cv::Vec3b>(y,x)[2] = 255;
                }
                if(x==xmax && y<=ymax && y>=ymin)
                {
                    output.at<cv::Vec3b>(y,x)[0] = 0;
                    output.at<cv::Vec3b>(y,x)[1] = 0;
                    output.at<cv::Vec3b>(y,x)[2] = 255;
                }                
                if(y==ymin && x<=xmax && x>=xmin)
                {
                    output.at<cv::Vec3b>(y,x)[0] = 0;
                    output.at<cv::Vec3b>(y,x)[1] = 0;
                    output.at<cv::Vec3b>(y,x)[2] = 255;
                }                
                if(y==ymax && x<=xmax && x>=xmin)
                {
                    output.at<cv::Vec3b>(y,x)[0] = 0;
                    output.at<cv::Vec3b>(y,x)[1] = 0;
                    output.at<cv::Vec3b>(y,x)[2] = 255;
                }
            }
        }

        }
        
        return output;
}


Mat getconnectedcomponents (Mat originalimage,Mat img)
{
    cv::Mat output = cv::Mat::zeros(img.size(), CV_8UC3);
    Mat binary;
    std::vector < std::vector<cv::Point2i > > blobs;
    cv::threshold(img, binary, 0.0, 1.0, cv::THRESH_BINARY);
    FindBlobs(binary, blobs);
    for(size_t i=0; i < blobs.size(); i++) {
        unsigned char r = 255 * (rand()/(1.0 + RAND_MAX));
        unsigned char g = 255 * (rand()/(1.0 + RAND_MAX));
        unsigned char b = 255 * (rand()/(1.0 + RAND_MAX));

        for(size_t j=0; j < blobs[i].size(); j++) {
            int x = blobs[i][j].x;
            int y = blobs[i][j].y;

            output.at<cv::Vec3b>(y,x)[0] = b;
            output.at<cv::Vec3b>(y,x)[1] = g;
            output.at<cv::Vec3b>(y,x)[2] = r;
        }
    }

 // output = drawrectangle(originalimage,blobs[1]);
return output;

}
vector<vector<Point2i> > detectallfaces(vector<vector<Point2i> > blobs)
{
    vector<vector<Point2i> > faceblobs;
    for(int i=0;i<blobs.size();i++)
    {

        int xmin=INT_MAX,ymin=INT_MAX,xmax=INT_MIN,ymax=INT_MIN;
        for(size_t j=0; j < blobs[i].size(); j++) {
            int x = blobs[i][j].x;
            int y = blobs[i][j].y;
            if(x<xmin){xmin=x;}
            if(x>xmax){xmax=x;}
            if(y<ymin){ymin=y;}
            if(y>ymax){ymax=y;} 
        }
        // faceblobs.push_back(blobs[i]);
        float ratio;
        if(ymax-ymin !=0)
        {
        float a=(xmax-xmin)*1.0;
        float b=(ymax-ymin)*1.0;
        ratio = b/a;
        // cout<<ratio<<endl;
        //ratio >=0.4 && ratio <= 1.1 && ecc >= 0.25 && ecc<=0.97
        float ecc;
        if(ratio<=1)
        {
             ecc=sqrt(1-((b*b)/(a*a)));       
        }
        else
        {
            ecc=sqrt(1-((a*a)/(b*b)));
        }

        // cout<<ecc<<endl;
        if(blobs[i].size() > 1500 && ratio >=0.4  && ratio <= 1.8 && ecc >= 0.25 && ecc<=0.97)
        {
             faceblobs.push_back(blobs[i]);
        }  
        }
    }

    return faceblobs;
}
Mat getface (Mat originalimage,Mat img)
{
    cv::Mat output = cv::Mat::zeros(img.size(), CV_8UC3);
    cv::Mat binary;
    std::vector < std::vector<cv::Point2i > > blobs;
    cv::threshold(img, binary, 0.0, 1.0, cv::THRESH_BINARY);
    FindBlobs(binary, blobs);
    for(size_t i=0; i < blobs.size(); i++) {
        unsigned char r = 255 * (rand()/(1.0 + RAND_MAX));
        unsigned char g = 255 * (rand()/(1.0 + RAND_MAX));
        unsigned char b = 255 * (rand()/(1.0 + RAND_MAX));

        for(size_t j=0; j < blobs[i].size(); j++) {
            int x = blobs[i][j].x;
            int y = blobs[i][j].y;

            output.at<cv::Vec3b>(y,x)[0] = b;
            output.at<cv::Vec3b>(y,x)[1] = g;
            output.at<cv::Vec3b>(y,x)[2] = r;
        }
    }
   vector<vector<Point2i> > faceblobs;
   faceblobs = detectallfaces(blobs);
    output = drawrectangle(originalimage,faceblobs);
return output;

}



int main(int argc,char **argv)
{
if(argc==3)
{
    if(argv[1][0]=='1')
    {
    Mat image;
    Mat skinMat;
    image = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    if(! image.data )
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    imshow("Original Image",image);
    skinMat= getSkin(image);
    imshow("Skin Image",skinMat);
    IplImage temp_image = skinMat;
    IplImage* result =imfill(&temp_image);
    cv::Mat noholesimage(result);
    imshow("afterfillingholes",noholesimage);
    Mat connectedcomponents=noholesimage.clone();
    connectedcomponents=getconnectedcomponents(image,connectedcomponents);
    imshow("connected components",connectedcomponents);
    noholesimage=getface(image,noholesimage);
    imshow("Face detected",noholesimage);
    waitKey(0);
    }
   else
   {
    VideoCapture cap(argv[2]);
    if(!cap.isOpened())
    return -1;
    namedWindow("Video",1);
    for(;;)
    {
    Mat frame;
    Mat skinMat;
    cap >> frame;   
    imshow("Video", frame);
    skinMat= getSkin(frame);
    imshow("Skin Image",skinMat);
    IplImage temp_image = skinMat;
    IplImage* result =imfill(&temp_image);
    cv::Mat noholesimage(result);
    imshow("afterfillingholes",noholesimage);
    Mat connectedcomponents=noholesimage.clone();
    connectedcomponents=getconnectedcomponents(frame,connectedcomponents);
    imshow("connected components",connectedcomponents);
    noholesimage=getface(frame,noholesimage);
    imshow("Face detected",noholesimage);
    if(waitKey(30) >= 0) break;
    }
   }
}
else
{
    VideoCapture capture;
    capture.open(0);
    capture.set(CV_CAP_PROP_FRAME_WIDTH,320);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT,480);
    Mat cameraFeed;
    Mat skinMat;
    while(1){
    capture.read(cameraFeed);
    imshow("Original Image",cameraFeed);
    skinMat= getSkin(cameraFeed);
    imshow("Skin Image",skinMat);
    IplImage temp_image = skinMat;
    IplImage* result =imfill(&temp_image);
    cv::Mat noholesimage(result);
    imshow("afterfillingholes",noholesimage);
    Mat connectedcomponents=noholesimage.clone();
    connectedcomponents=getconnectedcomponents(cameraFeed,connectedcomponents);
    imshow("connected components",connectedcomponents);
    noholesimage=getface(cameraFeed,noholesimage);
    imshow("Face detected",noholesimage);
    waitKey(30);
}

}
return 0;
}
