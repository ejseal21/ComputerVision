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

int main(int argc, char *argv[]) {
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
    cv::Mat frame;
    cv::Mat gray;
	std::vector<Point2f> corners;
	std::vector<std::vector<Point2f>> cornerList;
	std::vector<Point3f> pointSet;
	std::vector<std::vector<Point3f>> pointList; 

    Size patternSize = Size(9,6);
	for (int y=0; y>-6; y--){
		for (int x=0; x<9; x++){
			pointSet.push_back(Point3f(x,y,0));
		}
	}
		
	for(;!quit;) {
		*capdev >> frame; // get a new frame from the camera, treat as a stream

		if( frame.empty() ) {
		  printf("frame is empty\n");
		  break;
		}

		cvtColor(frame, gray, CV_BGR2GRAY);

        //copied a lot of this from documentation
		bool patternFound = findChessboardCorners(gray, patternSize, corners);

		if (patternFound){
		    cornerSubPix(gray,corners,Size(11,11),Size(-1,-1),
		            TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		}
		Mat frameCopy= frame.clone();
		drawChessboardCorners(frame, patternSize, Mat(corners), patternFound);
		imshow("Video", frame);
		int key = cv::waitKey(10);
		
		switch(key) {
			case 'q':
				quit = 1;
				break;
				
			case 'c': // capture a photo if the user hits c
				sprintf(buffer, "../data/%s.%03d.png", label, frameid++);
				cv::imwrite(buffer, frameCopy, pars);
				printf("Image written: %s\n", buffer);
				break;


			//this case creates the camera calibration matrix
			case 's':
				{
					if (corners.size() != 54)
						break;
					cornerList.push_back(corners);
					pointList.push_back(pointSet);

					sprintf(buffer, "../data/Calibrate.%s.%03d.png", label, frameid++);
					cv::imwrite(buffer, frame, pars);
					printf("Calibration Image written: %s\n", buffer);

					std::vector<double> distCoeffs;
					std::vector<Mat> rvecs;
					std::vector<Mat> tvecs;

					//need five corners in order to create the matrix
					if (cornerList.size() >= 5){
						float data[3][3] = {{1,0,frame.cols/2},{0,1,frame.rows/2},{0,0,1}};
						Mat cameraMatrix = Mat( 3,3 , CV_32FC1, data);
						Size imageSize = Size(frame.cols, frame.rows);

						//what a convenient function to calibrate the camera
						double error = calibrateCamera( pointList, cornerList, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_FIX_ASPECT_RATIO);
						std::cout << "cameraMatrix: " << cameraMatrix << "\n" << "distortion coefficients: ";
						for (int i=0; i<distCoeffs.size(); i++){
							std::cout << distCoeffs.at(i) << ", ";
						}
						std::cout << "\nerror: " << error << "\n";
					}
					break;
				}

			//this case writes the matrix and the distortion coeffecients to the file
			case 'w':
				{
				std::vector<double> distCoeffs;
				std::vector<Mat> rvecs;
				std::vector<Mat> tvecs;

				//only happens if there are at least 5 corners
				if (cornerList.size() >= 5){
					float data[3][3] = {{1,0,frame.cols/2},{0,1,frame.rows/2},{0,0,1}};
					Mat cameraMatrix = Mat( 3,3 , CV_32FC1, data);
					Size imageSize = Size(frame.cols, frame.rows);
					double error = calibrateCamera( pointList, cornerList, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_FIX_ASPECT_RATIO);
					std::cout << "cameraMatrix: " << cameraMatrix << "\n" << "distortion coefficients: ";
					
					for (int i=0; i<distCoeffs.size(); i++){
						std::cout << distCoeffs.at(i) << ", ";
					}
					
					std::cout << "\nerror: " << error << "\n";
					std::string filename = "writeout.txt";
					std::ofstream myfile;
					myfile.open (filename);
					
					//write cameraMatrix to file
					myfile << cameraMatrix;
					myfile << "\n";

					//write distortion coefficients to file
					for (int  i=0;i<distCoeffs.size(); i++){
						myfile <<  distCoeffs.at(i);
						if (i!=distCoeffs.size()-1){
							myfile << ", ";
						}
					}
					myfile << "\n";
					myfile.close();
					
				}  	
				break;
			}
		}
	}

	// terminate the video capture
	printf("Terminating\n");
	delete capdev;
	return(0);
}
