//
//  GestureDetection.cpp
//  ArmTest
//
//  Created by Jonecia Keels on 4/20/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

/*
 *  main.h
 *  HandRecognition
 *
 *  Created by Jonecia Keels on 2/16/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "math.h"
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <fstream>
using namespace std;
#include "GestureDetection.h"


bool verbose = false; //set true for output messages
bool debug = true; //set true for debugging messages

int main( int argc, char** argv ) {    //value to get the action of the user

    //create HandGesture object
    Simon mySimon;
    //IplImage *hand_img = cvLoadImage("/Users/jkeels12/Desktop/gesture_pics/b_5_isolated.png", CV_LOAD_IMAGE_UNCHANGED);
    IplImage *hand_img = mySimon.takePicture();
    cvNamedWindow( "Original Image", CV_WINDOW_AUTOSIZE );
    cvShowImage("Original Image", hand_img);    
    IplImage *binary_img = mySimon.skinDetection(hand_img);
    //perform hand recognition
    string answer = mySimon.gestureRecognition(mySimon.getMorphologicalCharacteristics(binary_img));
    cout << answer << endl;
    cvWaitKey();
        
	return 0;
    
}

void Simon::thresholdRGB(IplImage *hand_img) {
    
    
    
    
}

void Simon::readRGBValues(IplImage* hand_img) {
    
    int blue=-1, green=-1, red=-1;
    
    //arrays to store color count
    int redCount[255];
    int greenCount[255];
    int blueCount[255];
    
    //intialize everything to 0
    for(int i = 0; i < 256; i++) {
        redCount[i] = 0;
        greenCount[i] = 0;
        blueCount[i] = 0;
    }
    
    int pixelCount = 0;
    
    for (int row = 0; row < hand_img->height; row++) {
        
        // We compute the ptr directly as the head of the relevant row 
        uchar* ptr = (uchar*)(hand_img->imageData + row * hand_img->widthStep);
        
        for (int col = 0; col < hand_img->width; col++) {
            blue = ptr[3*col+0];
            green = ptr[3*col+1];
            red = ptr[3*col+2];
            
            pixelCount++;
            
            redCount[red]++;
            greenCount[green]++;
            blueCount[blue]++;
                    
            if(verbose)
                if(red != 0 && blue != 0 && green != 0)
                    cout << "RGB(" << red << "," << green << "," << blue << ")\n";
            if(debug) {
                
//                if(red >= 25 && red <= 100 && green >= 10 && green <= 30 && blue <= 30) {
//                if(red >= 70 && red <= 135 && green >= 30 && green <= 100 && blue <= 70) {
                if(red >= 70 && red <= 140 && green >= 30 && green <= 110 && blue <= 130) {
                    
                    //paint the value black
                    ptr[3*col+0] = 255.0;
                    ptr[3*col+1] = 0.0;
                    ptr[3*col+2] = 0.0;
                    
                }
            }
            
        }
        
        
    }
    
    int max_red = 0;
    int r_i, g_i, b_i;
    int max_green = 0;
    int max_blue = 0;
    
    //output the max rgb values
    for(int i = 0; i < 255; i ++) {
        if(redCount[i] > max_red && redCount[i] != 0 && redCount[i] < 250) {
            max_red = redCount[i];
            r_i = i;
        }
        
        if(greenCount[i] > max_green && greenCount[i] != 0 && greenCount[i] < 250) {
            max_green = greenCount[i];
            g_i = i;
        }
        
        if(blueCount[i] > max_blue && blueCount[i] != 0 && blueCount[i] < 250) {
            max_blue = blueCount[i];
            b_i = i;
        }
    }
    
    if(verbose)
        cout << "max rgb values is: (" << r_i << "," << g_i << "," << b_i << ")\n";
}

IplImage* Simon::takePicture() {
    
    CvCapture *capture = cvCreateCameraCapture(0);
    int key = 0;
    IplImage *frame = 0;
    if(!capture)
        cout << "ERROR!! CANNOT ACCESS WEBCAM!!\n";
    
    //create window for the video
    
    cvNamedWindow("Press 'q' to take a picture", 1);
    while(key != 'q') {
        frame = cvQueryFrame(capture);
        if(!frame)
            break;
        //display current frame
        cvShowImage("Press 'q' to take a picture", frame);
        
        key = cvWaitKey(1);
    }
    
    //get the last seen image frame
    IplImage *src = frame;
    return src;
    
}

IplImage* Simon::segmentImage(IplImage *hand_img) {
       
    IplImage *seg_img = cvCreateImage(cvGetSize(hand_img), 8, hand_img->nChannels);
    cvPyrMeanShiftFiltering(hand_img, seg_img, 20, 40);
    return seg_img;
    
}


IplImage* Simon::skinDetection(IplImage *hand_img) {
    
    //get the size of the image
    CvSize size = cvGetSize(hand_img);
    //converting image to hsv with a mask
    IplImage *hsv_image = cvCreateImage(size, 8, 3);
    IplImage *hsv_mask = cvCreateImage(size, 8, 1);

    
    //hsv threshold
    //CvScalar hsv_min = cvScalar(120, 20, 40, 0);
    //CvScalar hsv_max = cvScalar(180, 150, 255, 0);
    //CvScalar hsv_min = cvScalar(150, 20, 40, 0);
    //CvScalar hsv_max = cvScalar(180, 150, 255, 0);
    
    CvScalar hsv_min = cvScalar(0, 20, 40, 0);
    CvScalar hsv_max = cvScalar(50, 150, 255, 0);

    //CvScalar  hsv_min = cvScalar(0, 30, 80, 0);
	//CvScalar  hsv_max = cvScalar(20, 150, 255, 0);
    
    //convert to hsv
    cvCvtColor(hand_img, hsv_image, CV_BGR2HSV);
    IplImage *seg_img = segmentImage(hsv_image);
    
    //show the hsv image
    cvNamedWindow("HSV Image", 1);
    cvShowImage("HSV Image", hsv_image);
    
    //get pixel information
    getPixelInformation(seg_img);
    
    //apply the threshold
    cvInRangeS(seg_img, hsv_min, hsv_max, hsv_mask);
    //show the final image
    cvNamedWindow("Hand Detection", 1);
    cvShowImage("Hand Detection", hsv_mask);

    
    hsv_mask->origin = 1;
    
    //get the contours
    getContours(hsv_mask);

    //show segmented image
    //cvNamedWindow("Segmented HSV", 1);
    //cvShowImage("Segmented HSV", seg_img);
    
    //show the segmented hsv histogram
    //buildHistogram(seg_img);
    
    seperateHSVChannels(seg_img);
    
    return hsv_mask;
}

void Simon::edgeDetection(IplImage *hand_img) {
    
   /* CvSize size = cvGetSize(hand_img);
    IplImage *gray_img = cvCreateImage(size, IPL_DEPTH_8U, 1);*/
    IplImage *edge_img = cvCreateImage(cvGetSize(hand_img), IPL_DEPTH_8U, 1);
    
    //convert color image to grayscale
    //cvCvtColor(hand_img, gray_img, CV_BGR2GRAY);
    
    double min_threshold = 100;
    double max_threshold = 150;
    
    //perform edge detection
    cvCanny(hand_img, edge_img, min_threshold, max_threshold);
    
    //show the image
    cvNamedWindow("Edge Image", 1);
    cvShowImage("Edge Image", edge_img);
    
    
}

void getPixelInformation(IplImage *raw_image) {
    
    //count the amount of skin pixels detected
    int skinColorCount;
    
    //get different image information
    int width = raw_image->width;
    int height = raw_image->height;
    int channel = raw_image->nChannels;

    //traverse through first image each pixel and search for hand pixel
    for (int i = 0; i < width * height * channel; i += channel) {
        //to tell which is a hand pixel
        if((int) raw_image->imageData[i] == -1 && (int) raw_image->imageData[i+1] == -1 && (int) raw_image->imageData[i+2] == -1) {
            ///since HSV stores either a 0(noise) or -1(hand) pixel value, its easy
            skinColorCount++;
        }
    }
    
    cout << "The amount of skin pixels is: " << skinColorCount << endl;
    
}

void Simon::getContours(IplImage *binary_img) {
    
    
    //create rgb image
    IplImage* img_out       = cvCloneImage( binary_img );  // return image
    CvMemStorage* storage   = cvCreateMemStorage( 0 );    // container of retrieved contours
    CvSeq* contours         = NULL;
    CvScalar black          = CV_RGB( 0, 0, 0 ); // black color
    CvScalar white          = CV_RGB( 255, 255, 255 );   // white color
    double area;
    
    /******ELIMINATE NOISE FROM THE IMAGE************/
    
    // find contours in binary image
    cvFindContours( binary_img, storage, &contours, sizeof( CvContour ), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE );  
    
    while(contours)  { // loop over all the contours
    
        area = cvContourArea( contours, CV_WHOLE_SEQ );
        if( fabs( area ) <= 400 ) {  // if the area of the contour is less than threshold remove it
            // draws the contours into a new image
            cvDrawContours( img_out, contours, black, black, -1, CV_FILLED, 8 ); // removes white dots
        } else {
            cvDrawContours( img_out, contours, white, white, -1, CV_FILLED, 8 ); // fills in holes
        }
        contours = contours->h_next;    // jump to the next contour
    }
    
    
    
    /******************END NOISE ELIMINATION**********/
    
    /******************IDENTIFY LARGEST BLOB(HAND) ********************/
   // cvReleaseMemStorage( &storage );
   // return img_out;
   
    int maxLength = 0;
    int x_center_first = 0;
    int y_center_first = 0;
    IplImage *dst = cvCloneImage(img_out);//(cvGetSize(img_out), 8, 1);
    
    contours = 0;
    CvMemStorage* 	img_storage = cvCreateMemStorage(0);
    cvFindContours( dst, img_storage, &contours );
    //cvZero(dst);
    
    // find contours in binary image
    //cvFindContours( dst, storage, &contours, sizeof( CvContour ), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE );  
    
    if(contours){
        cvDrawContours(dst, contours, cvScalarAll(255), cvScalarAll(255), 0, CV_FILLED, 8);
    }
    
    CvRect rects; 
    //need to find the max length of the rectangle blob. This is so that we can determine where the wrist and dumbbell is (largest rectangle == wrist+dumbbell)
    //get x,y of the center of biggest blob
    
    //create a duplicate
    
    int r00, r10, r01, max_x, max_w, max_h, max_y;
    int rx,ry,rh, rw;
    //draw rectangles around the blob by traversing through all the blobs
    for(CvSeq *c = contours; c != NULL; c = c-> h_next) { 
        rects = cvBoundingRect(c);                  
        
        //calculate the length of the current rectangle
        if((rects.x + rects.width) > maxLength) {
            maxLength = rects.x + rects.width;
            max_x = rects.x;
            max_y = rects.y;
            max_w = rects.width;
            max_h = rects.height;
            CvMoments moment; 
            cvMoments(dst, &moment , 1);
            r00 = cvGetSpatialMoment(&moment,0,0);
            r10 = cvGetSpatialMoment(&moment,1,0);
            r01 = cvGetSpatialMoment(&moment,0,1);
            x_center_first = (int)(r10/r00);
            y_center_first = (int)(r01/r00);
            rx = rects.x;
            ry = rects.y;
            rh = rects.height;
            rw = rects.width;
        }
    
        if((rects.x + rects.width) > 400 ) //this is the hand blob
            cvRectangle(dst, cvPoint(rects.x,rects.y),cvPoint(rects.x + rects.width, rects.y + rects.height), cvScalarAll(255), 1);
        
    }
    
    
    if(verbose) 
        cout << "MAX LENGTH = " << maxLength << endl;
    
    //draw circle on center of biggest blob
    //cvRectangle(img, cvPoint(rx,ry), cvPoint(rx + rw, ry + rh), cvScalarAll(255), 1);
    cvCircle(dst, cvPoint(x_center_first, y_center_first), 25, cvScalarAll(150), -1, 8); 
    
    //printf("The center point is (%d, %d)\n", x_center_first, y_center_first);*/
    
    
    cvNamedWindow("find contour", 1);
    cvShowImage("find contour", rotateImage(dst));
    
    
    
}

void Simon::buildHistogram(IplImage *hsv_image) {
    
    //seperate hsv image into colors
    IplImage *h_plane = cvCreateImage(cvGetSize(hsv_image), 8, 1);
    IplImage *s_plane = cvCreateImage(cvGetSize(hsv_image), 8, 1);
    IplImage *v_plane = cvCreateImage(cvGetSize(hsv_image), 8, 1);
    IplImage *planes[] = {h_plane, s_plane};
    cvCvtPixToPlane(hsv_image, h_plane, s_plane, v_plane, 0);
    
    
    //build and fill up the histogram
    int h_bins = 30;
    int s_bins = 32;
    CvHistogram *histogram; {
        int histogram_size[] = {h_bins, s_bins};
        float h_ranges[] = {0, 180};
        float s_ranges[] = {0, 255};
        float *ranges[] = {h_ranges, s_ranges};
        histogram = cvCreateHist(2, histogram_size, CV_HIST_ARRAY, ranges, 1);
    }
    
    
    
    
    //calculate and compute the histogram
    cvCalcHist(planes, histogram, 0, 0);
    //normalize histogram
    cvNormalizeHist(histogram, 1.0);
    
    //create histogram image
    int scale = 10;
    IplImage *hist_img = cvCreateImage(cvSize(h_bins*scale, s_bins*scale), 8, 3);
    cvZero(hist_img);
    
    //populate histogram image
    float max_value = 0;
    cvGetMinMaxHistValue(histogram, 0, &max_value, 0, 0);
    
    for(int h = 0; h < h_bins; h++) {
        for(int s = 0; s < s_bins; s++) {
            float bin_val = cvQueryHistValue_2D(histogram, h, s);
            int intensity = cvRound(bin_val * 255 / max_value);
            cvRectangle(hist_img, cvPoint(h*scale, s*scale), cvPoint((h+1)*scale-1, (s+1)*scale-1), CV_RGB(intensity, intensity, intensity), CV_FILLED);
        }
    }
    
    //show histogram
    cvNamedWindow("Histogram", 1);
    cvShowImage("Histogram", hist_img);
}

void Simon::seperateHSVChannels(IplImage *hsv_image) {
    
    cvCvtColor(hsv_image, hsv_image, CV_BGR2HSV);
    
    IplImage *hue, *sat, *val, *hsv;
    CvSize size = cvGetSize(hsv_image);
    hue = cvCreateImage(size, hsv_image->depth, 1);
    sat = cvCreateImage(size, hsv_image->depth, 1);
    val = cvCreateImage(size, hsv_image->depth, 1);
    hsv = cvCreateImage(size, hsv_image->depth, 3);
    
    cvZero(hue);
    cvZero(sat);
    cvZero(val);
    cvZero(hsv);
    
    cvSplit(hsv, hue, sat, val, 0);
    
    //find the hue values and print the clusters
    int hueValues[360];
    //fill the histogram
    for(int row = 0; row < hue->height; row++) {
        uchar* ptr = (uchar*)(hue->imageData + row * hue->widthStep);
        for(int col = 0; col < hue->width; col++) {
            hueValues[ptr[col]]++;
        }
    }
    
    //print
    /*for(int i = 0; i < 360; i++) {
        if(hueValues[i] > 1000000000)
            cout << i << endl;
    }*/
    
    for(int i = 0; i < 360; i++)
        if(hueValues[i] > 1000000000)
            if(verbose)
                cout << i << " occurs " << hueValues[i] << " times\n";
    
    
}

float* Simon::getMorphologicalCharacteristics(IplImage *binary_img) {
    
    CvMemStorage* storage   =  cvCreateMemStorage();    // container of retrieved contours
    CvSeq* contours         = NULL;
    CvSeq* result;
    float area = 0.0;
    
    // find contours in binary image
    cvFindContours( binary_img, storage, &contours, sizeof( CvContour ), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE );  
    
    area = 0;
    float width = 0;
    float height = 0;
    float largestArea = 0.0;
    
    //iterate through points
    while(contours != NULL) {
        //if(cvContourArea( contours, CV_WHOLE_SEQ ) > area) {
            area = fabs(cvContourArea( contours));
        if(area > largestArea) {
            largestArea = area;
            width = cvBoundingRect(contours, 0).width;
            height = cvBoundingRect(contours, 0).height;
            result = cvApproxPoly(contours, sizeof(CvContour), storage, CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0);
        }
        
        contours = contours->h_next;    // jump to the next contour

    }
    
    //get the pixel count of the white pixels
    int pixelCount = 0;
    int whitePixel;
    
    for (int row = 0; row < binary_img->height; row++) {
        
        // We compute the ptr directly as the head of the relevant row 
        uchar* ptr = (uchar*)(binary_img->imageData + row * binary_img->widthStep);
        
        for (int col = 0; col < binary_img->width; col++) {
            whitePixel = ptr[3*col];
            if(verbose)
                cout << "white pixel: " << whitePixel << endl;
            if(whitePixel == 1)
                pixelCount++;
        }
        
    }
    
    float sides = result->total;
        
    cout << "pixel count = " << pixelCount << endl;
    cout << "width = " << width << endl;
    cout << "height = " << height << endl;
    cout << "area = " << largestArea << endl;
    cout << "sides = " << result->total << endl;
    
    //store into an area and return area
    float *characteristics;
    float temp[5] = {pixelCount, width, height, largestArea, sides};
    characteristics = temp;
    
    return characteristics;
        
}

IplImage* rotateImage(IplImage *img) {
    cvFlip(img, NULL, 0);
    return img;
}

string Simon::gestureRecognition(float* characteristics) {
    
    //0 = pixel count
    //1 = width
    //2 = length
    //3 = area
    //4 = sides
    
    float pixelCount, width, length, area, sides;    
    
    pixelCount = characteristics[0];
    width = characteristics[1];
    length = characteristics[2];
    area = characteristics[3];
    sides = characteristics[4];
    
    if(pixelCount >= 56000 && pixelCount <= 63000 && width >= 190 && width <= 230 && length >= 440 && length <= 470 && area >= 60000 && area <= 72000 && sides >= 4 && sides <= 8)
        return "This is B";
    if(pixelCount >= 48000 && pixelCount <= 80000 && width >= 280 && width <= 470 && length >= 370 && length <= 440 && area >= 51000 && area <= 79000 && sides >= 5 && sides <= 10)
        return "This is C";
    if(pixelCount >= 37000 && pixelCount <= 50000 && width >= 220 && width <= 370 && length >= 440 && length <= 470 && area >= 50000 && area <= 60000 && sides >= 6 && sides <= 11)
        return "This is D";
    if(pixelCount >= 54000 && pixelCount <= 67000 && width >= 220 && width <= 260 && length >= 440 && length <= 460 && area >= 63000 && area <= 75000 && sides >= 6 && sides <= 9)
        return "This is F";
    if(pixelCount >= 49000 && pixelCount <= 58000 && width >= 240 && width <= 310 && length >= 410 && length <= 445 && area >= 56000 && area <= 69000 && sides >= 6 && sides <= 9)
        return "This is I";
    if(pixelCount >= 39000 && pixelCount <= 70000 && width >= 380 && width <= 550 && length >= 430 && length <= 475 && area >= 40000 && area <= 95000 && sides >= 5 && sides <= 10)
    return "This is L";
    if(pixelCount >= 30000 && pixelCount <= 70000 && width >= 210 && width <= 380 && length >= 415 && length <= 485 && area >= 44000 && area <= 62000 && sides >= 4 && sides <= 12)
        return "This is R";
    if(pixelCount >= 33000 && pixelCount <= 66000 && width >= 210 && width <= 365 && length >= 430 && length <= 490 && area >= 35000 && area <= 69000 && sides >= 7 && sides <= 13)
        return "This is V";
    if(pixelCount >= 45000 && pixelCount <= 74000 && width >= 190 && width <= 375 && length >= 400 && length <= 480 && area >= 50000 && area <= 76000 && sides >= 6 && sides <= 12)
        return "This is L";
    
    
    return "not recognized";
}


//text to speech 