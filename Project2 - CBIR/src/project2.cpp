/*
	Henry Doud and Ethan Seal
	Content-Based Image Retrieval
*/
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <cstring>
#include <string>
#include "opencv/highgui.h"
#include "opencv2/opencv.hpp"
using namespace cv;


double sad(Mat srcImg, Mat testImg){
	//initialize the matrices
	Mat A = srcImg.clone();
	Mat B = testImg.clone();

	int x1 = A.rows/2-2;
	int y1 = A.cols/2-2;
	int x2 = B.rows/2-2;
	int y2 = B.cols/2-2;

	//go over 5x5 centers of each image at the centers
	double sums = 0;
	for (int i = 0; i < 5; i++){
		for (int j = 0; j < 5; j++){	
			for (int k=0; k<3; k++){
				int t = (A.at<Vec3b>(x1+i,y1+j)[k] - B.at<Vec3b>(x2+i,y2+j)[k]);
				sums += t*t;
			}
		}
	}
	return sums/4876875; // division normalizes the value
}

double hist(Mat srcImg, Mat testImg){
	Mat A = srcImg.clone();
	Mat B = testImg.clone();
	
	int rbins = 8;
    int histSize[] = {rbins, rbins};

    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float branges[] = { 0, 256 };
    const float* ranges[] = { branges, branges, branges };
    MatND histA;
	MatND histB;

    int channels[] = {0, 1,2};

    calcHist( &A, 1, channels, Mat(), // do not use mask
             histA, 2, histSize, ranges,
             true, // the histogram is uniform
             false );
	
	calcHist( &B, 1, channels, Mat(), // do not use mask
             histB, 2, histSize, ranges,
             true, // the histogram is uniform
             false );

	//normalize the histograms
	double factor = 1;
	normalize(histA, histA, factor, 0, NORM_L1);
	normalize(histB, histB, factor, 0, NORM_L1);
	
	double hdiff = compareHist(histA, histB, CV_COMP_INTERSECT);
	return hdiff;
}

double multHist(Mat srcImg, Mat testImg){
	Mat A = srcImg.clone();
	Mat B = testImg.clone();

	int x1 = A.rows/2;
	int x2 = B.rows/2;

	Rect roiA1 = Rect(0,0,A.cols,x1);
	Rect roiA2 = Rect(0,x1,A.cols,x1);

	Rect roiB1 = Rect(0,0,B.cols,x2);
	Rect roiB2 = Rect(0,x2,B.cols,x2);

	Mat subA1;
	Mat subA2;
	Mat subB1;
	Mat subB2;

	//the compiler yelled at us when we didn't use these if statements, so we use them and it works
	if ((0 <= roiA1.x && 0 <= roiA1.width && roiA1.x + roiA1.width <= A.cols && 0 <= roiA1.y && 0 <= roiA1.height && roiA1.y + roiA1.height <= A.rows)){subA1 = Mat(A,roiA1);}
	if ((0 <= roiA2.x && 0 <= roiA2.width && roiA2.x + roiA2.width <= A.cols && 0 <= roiA2.y && 0 <= roiA2.height && roiA2.y + roiA2.height <= A.rows)){subA2 = Mat(A,roiA2);}
	if ((0 <= roiB1.x && 0 <= roiB1.width && roiB1.x + roiB1.width <= B.cols && 0 <= roiB1.y && 0 <= roiB1.height && roiB1.y + roiB1.height <= B.rows)){subB1 = Mat(B,roiB1);}
	if ((0 <= roiB2.x && 0 <= roiB2.width && roiB2.x + roiB2.width <= B.cols && 0 <= roiB2.y && 0 <= roiB2.height && roiB2.y + roiB2.height <= B.rows)){subB2 = Mat(B,roiB2);}

	double hdiff1 = hist(subA1, subB2);
	double hdiff2 = hist(subA2, subB2);
	
	if (hdiff1 + hdiff2 == 0){
		return 0;
	}

	double hdiff = (hdiff1*hdiff2)/(hdiff1+hdiff2);
	return 2 * hdiff; //multiplication normalizes the value
}

double texture(Mat srcImg, Mat testImg){
	Mat A = srcImg.clone();
	Mat B = testImg.clone();
	
	int rbins = 8;
    int histSize[] = {rbins, rbins};

    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float branges[] = { 0, 256 };
    const float* ranges[] = { branges, branges, branges };
    MatND histA;
	MatND histB;

    int channels[] = {0, 1,2};

	//make histograms from the images
    calcHist( &A, 1, channels, Mat(), // do not use mask
             histA, 2, histSize, ranges,
             true, // the histogram is uniform
             false );
	calcHist( &B, 1, channels, Mat(), // do not use mask
             histB, 2, histSize, ranges,
             true, // the histogram is uniform
             false );
	
	//normalize the histograms
	double factor = 1;
	normalize(histA, histA, factor, 0, NORM_L1);
	normalize(histB, histB, factor, 0, NORM_L1);
	double cor = compareHist(histA, histB, CV_COMP_INTERSECT);
	double tex = compareHist(histA, histB, CV_COMP_CHISQR_ALT);

	return cor * tex;
}

double randssd(Mat srcImg, Mat testImg){
	
	//initialize the matrices
	Mat A = srcImg.clone();
	Mat B = testImg.clone();
	
	int rows;
	int cols;
	
	if (A.rows > B.rows){
		rows = B.rows;
	}
	else {
		rows = A.rows - 1; // tried subtracting 1 on a whim and it got rid of a seg fault
	}
	
	if (A.cols > B.cols){
		cols = B.cols;
	}
	else{
		cols = A.cols;
	}
	int x;
	int y;
	double sums = 0;
	
	//go over 100 random values for i and j
	for (int i = 0; i < 100; i++){
		//random values for x and y
		x = rand() % rows;
		y = rand() % cols;	
		for (int k=0; k<3; k++){
			int t = (A.at<Vec3b>(x,y)[k] - B.at<Vec3b>(x,y)[k]);
			sums += t*t;
		}
	}
	return sums/19507500; //division normalizes the value
}



//Create a GUI that cycles through the images based on user input
//Passing in the name of the method being used to be the title of the window
void createGUI(std::vector<std::string> min, std::vector<double> scores, std::string name){
	// std::string name = "Interactive photo ";  
	//Use WINDOW_NORMAL and resizeWindow to resize the window from a gigantic things
	namedWindow(name, WINDOW_NORMAL);
	resizeWindow(name, 800, 800);
	Mat src = imread(min[0]);

	std::string toprint = "Score: ";
	std::string result = toprint + std::to_string(scores[0]);
	std::cout << scores[-1] << "\n\n";
	
	//draw text on the image so the user can see what score the image got
	putText(src, result, Point(0,src.rows), FONT_HERSHEY_PLAIN, src.cols/300, Scalar({255,0,255}),src.cols/300 );
	

	imshow(name, src);
	
	int index = 0;
	int key = waitKey(0);
	std::string score;

	//cycle through the images using k and l by incrementing or decrementing the index	
	while (key != 'q'){
		if (key == 'l'){
			index += 1;
			if (index >= min.size()){
				index = 0;
			}
		}
		else if (key == 'k'){
			index -= 1;
			if (index < 0){
				index = min.size()-1;
			}
		}
		src = imread(min[index]);
		result = toprint + std::to_string(scores[index]);
		putText(src, result, Point(0,src.rows), FONT_HERSHEY_PLAIN, src.cols/300, Scalar({255,0,255}),src.cols/300 );
		imshow(name, src);
		key = waitKey(0);	
	}
	// get rid of the window
	destroyWindow(name);
}

int main(int argc, char *argv[]) {
	char dirname[256];
	DIR *dirp;
	struct dirent *dp;

	// by default, look at the current directory
	strcpy(dirname, ".");

	// if the user provided a directory path, use it
	if(argc > 1) {
		strcpy(dirname, argv[1]);
	}
	printf("Accessing directory %s\n\n", dirname);

	// open the directory
	dirp = opendir( dirname );
	if( dirp == NULL ) {
		printf("Cannot open directory %s\n", dirname);
		exit(-1);
	}

	char imgFilename[256];
	char dirFilename[256];
	Mat img;
	Mat dirImg;

	strcpy(imgFilename, argv[2]);
	img = imread(imgFilename);

	std::string name = "Original Image";
	namedWindow(name, WINDOW_NORMAL);
	resizeWindow(name, 800, 800);
	imshow(name, img);
	std::cout << "\nPress 1 for SSD\nPress 2 for Histogram\nPress 3 for Multiple Histograms\nPress 4 for Texture\nPress 5 for RandomSSD\n";
	while (true){
		std::string caption;
		int key = waitKey(0);
		std::vector<double> scores;
		std::vector<std::string> names;

		if (key == '1' || key == '2' || key == '3' || key == '4' || key == '5'){
			// loop over the contents of the directory, looking for images
			while( (dp = readdir(dirp)) != NULL ) {
				if( strstr(dp->d_name, ".JPG") ||
					strstr(dp->d_name, ".jpg") ||
						strstr(dp->d_name, ".PNG") ||
						strstr(dp->d_name, ".png") ||
						strstr(dp->d_name, ".PPM") ||
						strstr(dp->d_name, ".ppm") ||
						strstr(dp->d_name, ".TIF") ||
						strstr(dp->d_name, ".tif") ) {
					
					
					char filestart[256];
					strcpy(filestart, argv[1]);
					strcat(filestart, dp->d_name);
					
					dirImg = imread(filestart);
					double diff = 0;
					
					//lets the user select which comparison they want to do by hitting 1-5
					//sets caption to the proper caption so that the window can have a proper name
					if (key == '1'){
						diff = sad(img, dirImg);
						caption = "Sum Squared Difference of Center 5x5 Matrix";
					}
					else if (key == '2'){
						diff = hist(img, dirImg);
						caption = "Histogram Comparison";
					}
					else if (key == '3'){
						diff = multHist(img, dirImg);
						caption = "Multiple Histogram Regions";
					}
					else if (key == '4'){
						diff = texture(img, dirImg);
						caption = "Texture Comparison Histograms";
					}
					else if (key == '5'){
						diff = randssd(img, dirImg);
						caption = "Sum Squared Difference of 100 Random Pixels";
					}

					printf("image file: %s\n", dp->d_name);
					scores.push_back(diff);
					std::string fileString(filestart);
					names.push_back(fileString);
				}
			}

			//find minimum score
			std::vector<std::string> min;
			std::vector<double> minScores;

			for (int i = 0; i < 10; i++){
				int size = scores.size();
				int index = distance(scores.begin(), std::min_element(scores.begin(), scores.end()));
				min.push_back(names[index]);
				minScores.push_back(scores[index]);
				names.erase(names.begin()+index);
				scores.erase(scores.begin()+index);
			}

			for (int i=0; i<min.size(); i++){
				std::cout << i << "\tMinimum: " << min[i] << "\n";
			}
			createGUI(min, minScores, caption);
		}
		dirp = opendir( dirname );
		std::cout << "\nPress l and k to cycle through images\n";
		if (key == 'q'){
			break;
		}
	}
	// close the directory
	
	closedir(dirp);
		
	printf("\nTerminating\n");

	return(0);
}