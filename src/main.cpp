#include "types.h"
#include <iostream>
#include <time.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/objdetect.hpp"

using namespace cv;
using namespace std;

/**Global variables for Problem 1.1 and 1.2*/
time_t start, end;
int f = 0; //number of frames
double s = 0; //seconds
int fps = 0; //frames per second
VideoCapture camera;

/**Additional variables for Problem 1.2*/
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

/**Function Headers*/
void detectAndDisplay(Mat img);

int main(int argc, const char** argv) {
    
    /**Problem 1.1*/
    
    //starting timer
    time(&start);
    
    //checking whether the camera works or not
	if (!camera.open(0)) {
		printf("Can't find a camera\n");
		return 1;
	};
	
	/** Main loop where it will count the number
    of frames and overall seconds it took */
	Mat img;
	while(true) {
        	f++;
		camera >> img;
		imshow("Camera", img);
		int key = waitKey(5);
		if (key == 27 || key == 'q') break;
	}
    
    time(&end) //end timer
    s = difftime (end, start);
       
    fps = f / s;
    cout << "Frames: " << f << ", Seconds: " << s << ", Frames per second: " << fps << endl;
    
    camera.release();

    
    /**Problem 1.2*/
    
    String face_cascade_name = samples::findFile( parser.get<String>("face_cascade") );
    String eyes_cascade_name = samples::findFile( parser.get<String>("eyes_cascade") );
    
    if(!face_cascade.load( face_cascade_name )){
        cout << "--(!)Error loading face cascade\n";
        return 1;
    };
    if( !eyes_cascade.load( eyes_cascade_name ) )
    {
        cout << "--(!)Error loading eyes cascade\n";
        return 1;
    };
    
    time(&start); //starting timer again for problem 2
    if (!camera.open(0)) {
        printf("Can't find a camera\n");
        return 1;
    };
    
    Mat img;
    while(true) {
        camera >> img;
        imshow("Camera with Cascade Classifier", img);
        detectAndDisplay(img);
        int key = waitKey(5);
        if (key == 27 || key == 'q') break;
    
    }
    
    time(&end) //end timer
    s = difftime (end, start);
    
    fps = f / s;
    cout << "Frames: " << f << ", Seconds: " << s << ", Frames per second: " << fps << endl;
    camera.release();

    return 0;
}
    
void detectAndDisplay( Mat frame ){
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
   
    //-- Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale( frame_gray, faces );
        
    for ( size_t i = 0; i < faces.size(); i++ ){
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4 );
        Mat faceROI = frame_gray( faces[i] );
        
        //-- In each face, detect eyes
        std::vector<Rect> eyes;
        eyes_cascade.detectMultiScale( faceROI, eyes );
            
   	for ( size_t j = 0; j < eyes.size(); j++ ){
        	Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
       		int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
        	circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4 );
    	}
     }
     //-- Show what you got
    imshow( "Capture - Face detection", frame );
}
