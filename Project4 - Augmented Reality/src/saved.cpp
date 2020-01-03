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


Mat drawPyramid(Mat frame){
    cv::Mat gray;
	std::vector<Point2f> corners;
	std::vector<std::vector<Point2f>> cornerList;
	std::vector<Point3f> pointSet;
	std::vector<std::vector<Point3f>> pointList; 

	//size of the pattern of the checkerboard
    Size patternSize = Size(9,6);
	for (int y=0; y>-6; y--){
		for (int x=0; x<9; x++){
			pointSet.push_back(Point3f(x,y,0));
		}
	}

	cvtColor(frame, gray, CV_BGR2GRAY);

	//copied a lot of this from documentation
    
	bool patternFound = findChessboardCorners(gray, patternSize, corners);
    if (patternFound){
		cornerSubPix(gray,corners,Size(11,11),Size(-1,-1),
				TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
	}

	Mat tvec;
	Mat rvec;
	std::vector<Point2f> outputCorners;
	std::vector<Point2f> outputExtra;
	std::vector<Point2f> midOut;
	//all the points in world coordinates
	std::vector<Point3f> externalCorners{Point3f(-1,1,0), Point3f(9,1,0), Point3f(-1,-6,0), Point3f(9,-6,0)};
	std::vector<Point3f> pyramidPoints{Point3f(4,-3,0), Point3f(4,-6,0), Point3f(8,-6,0), Point3f(8,-3,0), Point3f(6,-4.5,7)};
	std::vector<Point3f> extraPoints{Point3f(6,-4.5,7), Point3f(6,-4.5,0), Point3f(4,-3,0), Point3f(4,-6,0), Point3f(8,-6,0), Point3f(8,-3,0)};
	std::vector<Point3f> midPoint{Point3f(4.5,-3,0)};
	
	if (corners.size() == pointSet.size()){
		//get rvec and tvec
		solvePnP(pointSet, corners, ::camMat, ::distortLine, rvec, tvec);
		std::cout << "\ntvec\n" << tvec << "\nrvec\n" << rvec << "\n";
		
		//convert all the coordinates to image coordinates
		projectPoints(externalCorners, rvec, tvec, ::camMat, ::distortLine, outputCorners);\
		
		for (int i=0; i<outputCorners.size(); i++){
			circle(frame, outputCorners.at(i), 5, (0,0,0));
		}
		projectPoints(extraPoints, rvec, tvec, ::camMat, ::distortLine, outputExtra);            
		projectPoints(pyramidPoints, rvec, tvec, ::camMat, ::distortLine, outputCorners);

		//draw all the points
		for (int i=0; i<outputCorners.size()-1; i++){
            line(frame, outputCorners.at(i), outputCorners.at(i+1), Scalar({255,0,0}), 4);
		}
		line(frame, outputCorners.at(0), outputCorners.at(3), Scalar({255,0,0}), 4);

		for (int i=2;i<6;i++){
			line(frame,outputExtra.at(1), outputExtra.at(i), Scalar({0,0,255}), 4);
		}
		line(frame, outputExtra.at(0), outputExtra.at(1), Scalar({0,0,255}), 4);

		
		for (int i=0; i<outputCorners.size()-1; i++){
            line(frame, outputCorners.at(i), outputCorners.at(4), Scalar({255,0,0}), 4);
		}

	}
    return frame;
}



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
    printf("Image must be taken with calibrated camera\n");
	readFile("writeout.txt");
    std::cout << "camMat\n" << ::camMat << "\n";
    
    Mat frame;
    char filename[256];
    // usage
	if(argc < 2) {
		printf("Usage %s <image filename>\n", argv[0]);
		exit(-1);
	}
	strcpy(filename, argv[1]);

	// read the image
	frame = cv::imread(filename);

	// test if the read was successful
	if(frame.data == NULL) {
		printf("Unable to read image %s\n", filename);
		exit(-1);
	}
    namedWindow("Start",WINDOW_NORMAL);
    imshow("Start",frame);
    resizeWindow("Start", 800, 800);
	
    Mat draw = drawPyramid(frame);
    namedWindow("Image",WINDOW_NORMAL);
    resizeWindow("Image", 800, 800);
	
    imshow("Image",draw);
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
	strcpy(label, argv[1]);
           
	for(;!quit;) {

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
	// delete capdev;

	return(0);
}
