//
//  GestureDetection.h
//  ArmTest
//
//  Created by Jonecia Keels on 4/20/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef ArmTest_GestureDetection_h
#define ArmTest_GestureDetection_h

class Simon {
    
public:
    IplImage* takePicture();
    IplImage* segmentImage(IplImage*);
    IplImage* skinDetection(IplImage*);
    void edgeDetection(IplImage*);
    void getContours(IplImage*);
    void buildHistogram(IplImage*);
    void seperateHSVChannels(IplImage*);
    void readRGBValues(IplImage*);
    void thresholdRGB(IplImage*);
    string gestureRecognition(float*);
    float* getMorphologicalCharacteristics(IplImage*);
};

void getPixelInformation(IplImage*);
IplImage* rotateImage(IplImage*);


#endif
