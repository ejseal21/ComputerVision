/*
	Ethan Seal and Henry Doud
	S19

	Compile command (macos)

	clang++ -o vid -I /opt/local/include vidDisplay.cpp -L /opt/local/lib -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio

	use the makefiles provided

	make vid

*/
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>

using namespace cv;
Mat camMat = Mat::zeros(3,3, CV_32FC1);
std::vector<float> distortLine;

void readFile(std::string filename){
    std::string line;
	std::ifstream nameFileout;

	nameFileout.open(filename);
	if (!nameFileout){
		std::cout << "unable to open\n";
		exit(1);
	}

    bool loopMat = true;
    int i = 0;
    int j = 0;
	while (std::getline(nameFileout, line))
	{
		std::stringstream ss(line);        
        float value;
       
        ss.ignore(); // ignore '['
        
        while ((ss >> value)){
            if (loopMat){ //camera matrix
                ::camMat.at<float>(j,i) = value;
                if (ss.peek() == ',' || ss.peek() == ' '){
                    ss.ignore();
                    i++;
                }
                else if (ss.peek() == ';'){ // new row
                    ss.ignore();
                    j++;
                    i=0;
                }
                else if (ss.peek() == ']'){ // break at end of matrix
                    loopMat = false;
                }
            }
            
            else{ // distort
                ::distortLine.push_back(value);
                if (ss.peek() == ',' || ss.peek() == ' ')
                    ss.ignore();
            }

        };

        std::cout << ::camMat << "\n";
        std::cout << "distort:\n";
        for (i=0;i<distortLine.size();i++){
            std::cout << distortLine[i] << ", ";
        }
        std::cout << "\n";

	    
	}
	nameFileout.close();

}

int main(int argc, char *argv[]) {
	
    readFile("writeout.txt");
    std::cout << "camMat\n" << ::camMat << "\n";
            
    cv::VideoCapture *capdev;
	char label[256];
	int quit = 0;
	int frameid = 0;
	char buffer[256];
	std::vector<int> pars;

	pars.push_back(5);

	if( argc < 2 ) {
	    printf("Usage: %s <label>\n", argv[0]);
	    exit(-1);
	}

	// open the video device
	capdev = new cv::VideoCapture(0);
	if( !capdev->isOpened() ) {
		printf("Unable to open video device\n");
		return(-1);
	}

	strcpy(label, argv[1]);

	cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
		       (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

	printf("Expected size: %d %d\n", refS.width, refS.height);

	cv::namedWindow("Video", 1); // identifies a window?
    Mat frame, gray, dst, dst_norm, dst_norm_scaled;
    Mat labels, stats, centroids, dstThresh;
    

            
	for(;!quit;) {
		*capdev >> frame; // get a new frame from the camera, treat as a stream

		if( frame.empty() ) {
		  printf("frame is empty\n");
		  break;
		}

	    cvtColor(frame, gray, CV_BGR2GRAY);
        dst = Mat::zeros( frame.size(), CV_32FC1 );
        
        int blockSize = 2;
        int apertureSize = 3;
        double k = 0.04;

        //get the harris corner matrix
        cornerHarris(gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);


        Mat dstCopy = dst.clone();

        //loop over every pixel
        //if it's not a local max, set it to 0
        for (int i=0; i<dst.rows; i++){
            for (int j=0; j<dst.cols; j++){
                for (int x=-1; x<2; x++){
                    for (int y=-1; y<2; y++){
                        if (x!=0 && y!=0){
                            if (dst.at<float>(i,j) <= dst.at<float>(i+x,j+y)){
                                dstCopy.at<float>(i,j) = 0;
                            }
                        }
                    }
                }
            }
        }

        //morphology to prevent weird errors with normalizing/drawing circles / help with normalization
        morphologyEx(dstCopy, dstCopy, MORPH_CLOSE, getStructuringElement(MORPH_RECT,Size(5,5))); 
       
        //normalization
        normalize( dstCopy, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
        convertScaleAbs( dst_norm, dst_norm_scaled );

        //draw a circle on all of the corners that come out of the above process
        for (int i=0; i<dst.rows; i++){
            for (int j=0; j<dst.cols; j++){
                if (dst_norm.at<float>(i,j) > 10){
                    circle(frame, Point(j,i), 5, Scalar({0,255,0}), 3);
                }
            }
        }
      
        /// Showing the result
        namedWindow("Window", CV_WINDOW_AUTOSIZE );
        // std::cout << dst_norm_scaled << "\n\n\n\n\n\n\n\n";
        imshow( "Window", dst_norm_scaled );

		imshow("Video", frame);

		int key = cv::waitKey(10);
		switch(key) {
			case 'q':
				quit = 1;
				break;
				
			case 'c': // capture a photo if the user hits c
				sprintf(buffer, "../data/%s.%03d.png", label, frameid++);
				cv::imwrite(buffer, frame, pars);
				printf("Image written: %s\n", buffer);
				break;
		}

	}

	// terminate the video capture
	printf("Terminating\n");
	delete capdev;

	return(0);
}
