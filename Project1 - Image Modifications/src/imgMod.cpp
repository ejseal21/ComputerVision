/*
	Bruce A. Maxwell
	J16 
	Simple example of reading, manipulating, displaying, and writing an image

	Compile command

	clang++ -o imod -I /opt/local/include imgMod.cpp -L /opt/local/lib -lopencv_core -lopencv_highgui 

*/
#include <cstdio>
#include <cstring>
#include "opencv2/opencv.hpp"
using namespace cv;

void trackbar(int, void*){
	return;
}

int main(int argc, char *argv[]) {
        cv::Mat src;
	char filename[256];

	// usage
	if(argc < 2) {
		printf("Usage %s <image filename>\n", argv[0]);
		exit(-1);
	}
	strcpy(filename, argv[1]);

	// read the image
	src = cv::imread(filename);

	// test if the read was successful
	if(src.data == NULL) {
		printf("Unable to read image %s\n", filename);
		exit(-1);
	}

	// print out information about the image
	printf("filename:         %s\n", filename);
	printf("Image size:       %d rows x %d columns\n", (int)src.size().height, (int)src.size().width);
	printf("Image dimensions: %d\n", (int)src.channels());
	printf("Image depth:      %d bytes/channel\n", (int)src.elemSize()/src.channels());

	// create a window
	cv::namedWindow(filename, 1);

	

	// edit the source image here
	

	int val;
	createTrackbar("trackbar", filename, &val, 100, trackbar);
	
	//show the image at the start
	imshow(filename, src);
	
	//initialize matrices
	
	Mat img = src;
	Mat blur;
	
	int i = 1;
	int lastval = 0;
	while (1)
	{
		int key = waitKey(10);
		if (key == 'q')
			break;
		//have a separate value for i because it always has to be an odd number
		if (lastval != val)
		{
			i +=2;// (val/2) * 2 + 1; // make this val but odd
			GaussianBlur(img,blur,Size(i,i),0,0);
			imshow(filename, blur);
			lastval = val;
		
			if (val == 0)
			{
				imshow(filename, img);
			}
		}
	}	
	

	// //code to perform a Gaussian blur on the image
	// //for this to work smoothly, comment out the waitKey below
	// //command+f for "comment this out" to find it easily
	
	// //show the image at the start
	// imshow(filename, src);
	
	// //initialize matrices
	// Mat img = src;
	// Mat blur;

	// //put the keypress in a variable
	// char key = waitKey(-1);

	// //as long as the user doesn't quit, execute this
	// while (key != 'q')
	// {
	// 	//blur the image if the user presses b
	// 	if (key == '1')
	// 	{
	// 		for ( int i = 1; i < 20; i = i + 2 )
	// 		{
	// 			//this function is nice because it does all the work for me
	// 			GaussianBlur(img, blur, Size( i, i ), 0, 0 );
	// 		}
	// 		//show the image after blurring
	// 		imshow(filename, blur);
	// 	}
	// 	else if (key == '2')
	// 	{
	// 		for ( int i = 1; i < 40; i = i + 2 )
	// 		{
	// 			//this function is nice because it does all the work for me
	// 			GaussianBlur(img, blur, Size( i, i ), 0, 0 );
	// 		}
	// 		//show the image after blurring
	// 		imshow(filename, blur);
	// 	}
	// 	else if (key == '3')
	// 	{
	// 		for ( int i = 1; i < 60; i = i + 2 )
	// 		{
	// 			//this function is nice because it does all the work for me
	// 			GaussianBlur(img, blur, Size( i, i ), 0, 0 );
	// 		}
	// 		//show the image after blurring
	// 		imshow(filename, blur);
	// 	}
	// 	//if the user doesn't want to blur the image		
	// 	else if (key == '0')
	// 	{
	// 		//show the original image		
	// 		imshow(filename, src);
	// 	}
	// 	key = waitKey(-1);
	// }
	



	// //code using the threshold function
	// //declaring variables
	// Mat img = src;
	// Mat output; 
	// Mat gray;
	
	// //convert the image to grayscale
	// cvtColor(img, gray, CV_BGR2GRAY);

	// //select grayscale values between 128 and 130
	// threshold(gray, output, 128, 130 , THRESH_BINARY);
	
	// //write the grayscale image and the binary image to files
	// imwrite("../data/gray.jpg", gray);
	// imwrite("../data/thresh.jpg", output);



	// //code to turn the image blue
	// Mat img = src;
	// //loop through each row and column
	// for(int i = 0; i < img.rows; i++){
	//   for(int j = 0; j < img.cols; j++){
	// 		//a 3-byte vector called color gets the pixel at the current point
	//     cv::Vec3b color = img.at<cv::Vec3b>(cv::Point(i,j));
	// 		//add 100 to the first value in the vector
	//     color.val[0] = 100;
	// 		//update the pixel's color
	//     img.at<cv::Vec3b>(cv::Point(i,j)) = color;
	// 	}
	// }
	// // write the image to a new file
	// imwrite("../data/blue.jpg", img);

	

	// //code to flip the image
	// //for this to work as smoothly as possible, comment out the waitKey below
	// //command+f "comment this out" to find it easily 

	// //start off with the image in the window
	// cv::imshow(filename, src);

	// //initialize matrices
	// Mat img = src;
	// Mat flipped;
	
	// //assign to key any key the user presses
	// char key = waitKey(-1);
	// //as long as the user hasn't pressed q, execute this while loop
	// while (key != 'q')
	// {
	// 	//seeing if the user ever presses h
	// 	if (key == 'h')
	// 	{
	// 		//flip the image over a vertical axis
	// 		cv::flip(img, flipped, 1);
	// 		cv::imshow(filename,flipped);
	// 	}
	// 	//seeing if the user ever presses v
	// 	else if (key == 'v')
	// 	{
	// 		//flip the image over a horizontal axis
	// 		cv::flip(img, flipped, 0);
	// 		cv::imshow(filename,flipped);
	// 	}
	// 	else
	// 	{	
	// 		//if the user presses anything else, show the original image
	// 		cv::imshow(filename, src);
	// 	}
	// key = waitKey(-1);

	// }	
	
	
	// wait for a key press (indefinitely)
	//comment this out
	cv::waitKey(0);

	// get rid of the window
	cv::destroyWindow(filename);

	// terminate the program
	printf("Terminating\n");

	return(0);
}
