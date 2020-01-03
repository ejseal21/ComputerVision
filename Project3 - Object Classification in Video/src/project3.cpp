/*
	Henry Doud and Ethan Seal

	Compile command

	clang++ -o imod -I /opt/local/include imgMod.cpp -L /opt/local/lib -lopencv_core -lopencv_highgui 

*/
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
using namespace cv;

std::vector<std::vector<float>> featureVector;
std::vector<std::string> labelVector;

void trackbar(int, void*){
	return;
}





std::vector<float> calcFeatures(Mat openMat, Mat connMat, int regions, int label=-1){
	//see if they gave a valid label 
	if (label > regions || label < -1)
		label = 1;

	std::vector<std::vector<cv::Point>> points;
	for (int i=0; i < regions; i++){
		points.push_back(std::vector<Point>());
	}
	// Loop over each pixel and create a point
	for (int x = 0; x < connMat.rows; x++){
	    for (int y = 0; y < connMat.cols; y++){
			if (regions != 0){
				for (int i=0 ; i < regions; i++){
					if (connMat.at<int>(x,y) == i)
			        	points[i].push_back(Point(y, x));
	    		}
	    	}
	    	else
				points[0].push_back(Point(x, y));
		 }
	}
	//if the user didn't pass in a label, give it a label
	if (label == -1){
		label = 0;
		//checks the middle quarter of the image first and gives label the first thing it finds
		for (int i = (connMat.cols/2) - (connMat.cols/4); i < (connMat.cols/2) + (connMat.cols/4); i++){
			for (int j = (connMat.rows/2) - (connMat.rows/4); j < (connMat.rows/2) + (connMat.rows/4); j++){
				if (connMat.at<int>(j,i) != 0){
					label = connMat.at<int>(j,i);
					goto outside;
				}
			}
		} 
	}
	outside:
	RotatedRect rect = minAreaRect(points[0]);
	if (regions != 0){
		rect = minAreaRect(points[label]);
	}
    // matrices we'll use
    Mat M, rotated, cropped;
    // get angle and size from the bounding box
    float angle = rect.angle;
    Size rect_size = rect.size;
    // thanks to http://felix.abecassis.me/2011/10/opencv-rotation-deskewing/
    if (rect.angle < -45.) {
        angle += 90.0;
        swap(rect_size.width, rect_size.height);
    }
    // get the rotation matrix
    M = getRotationMatrix2D(rect.center, angle, 1.0);
    // perform the affine transformation
    warpAffine(openMat, rotated, M, connMat.size(), INTER_CUBIC);
    // crop the resulting image
    getRectSubPix(rotated, rect_size, rect.center, cropped);
    // imshow("cropped", cropped);

	Moments ourMoments = moments(cropped, false);
	float hc = ourMoments.mu02/(ourMoments.m00 * ourMoments.m00);//prevents nonzero values
	float vc = ourMoments.mu20/(ourMoments.m00 * ourMoments.m00);
	float hwr; 
	if (rect.size.height > rect.size.width){
		
		hwr = rect.size.width/rect.size.height;
	}
	else{
		hwr = rect.size.height/rect.size.width;
	}

	if (vc > hc){
		vc = ourMoments.mu02/(ourMoments.m00 * ourMoments.m00);//prevents nonzero values
		hc = ourMoments.mu20/(ourMoments.m00 * ourMoments.m00);
	}
			
	float fill = ourMoments.m00/((rect.size.width * rect.size.height)+ 0.0001);

	return {hc, vc, hwr, 1-fill};
}

std::vector<cv::Point> findRegionPoints(Mat connMat, int regions){
	std::vector<cv::Point> points;
	// Loop over each pixel and create a point
	for (int i=1 ; i < regions; i++){
		for (int x = 0; x < connMat.rows; x++){
		    for (int y = 0; y < connMat.cols; y++){
				if (connMat.at<int>(x,y) == i){
					points.push_back(Point(y, x));
					goto skip;
				}	
			}
		}	
	skip:
	;}
	
	return points;
}

std::vector<std::vector<float>> calcMeanStdDev(){
	std::vector<float> means;
	std::vector<float> stddev;
	std::vector<float> feature;
	for (int i = 0; i < ::featureVector.at(0).size()-1; i++){
		means.push_back(0);
		stddev.push_back(0);
	}

	for (int j = 0; j < ::featureVector.size(); j++){
		feature = ::featureVector.at(j);
		for (int i = 0; i < feature.size()-1; i++){
			means.at(i) += feature.at(i+1)/::featureVector.size();
		}
	}

	for (int j = 0; j < ::featureVector.size(); j++){
		feature = ::featureVector.at(j);
		for (int i = 0; i < feature.size()-1; i++){
			stddev.at(i) += ((feature.at(i+1)-means.at(i)) * (feature.at(i+1)-means.at(i)))/::featureVector.size();
		}
	}

	for (int i=0; i < stddev.size(); i++){
		stddev.at(i) = sqrt(stddev.at(i));
		}

	std::vector<std::vector<float>> together;
	together.push_back(means);
	together.push_back(stddev);

	return together;
}

std::vector<std::vector<std::vector<float>>> calcMeanStdDevLabels(){
	std::vector<std::vector<float>> means;
	std::vector<std::vector<float>> stddev;
	std::vector<std::vector<std::vector<float>>> labelFeatures;
	std::vector<float> feature;
	std::vector<std::vector<float>> labelf;
	float maxInd = 0;

	// put data base features into label boxes
	// make stuff right size
	for (int i = 0; i < ::featureVector.size(); i++){ // find the number of label objects
		if (::featureVector.at(i).at(0) > maxInd){
			maxInd = ::featureVector.at(i).at(0);
		}
	}

	for (int i = 0; i <= int(maxInd); i++){ // make the number of lists for the labels
		labelFeatures.push_back({{{0}}});
		means.push_back({}); // put empty lists for means
		stddev.push_back({});
		for (int j=0; j< featureVector.at(0).size()-1; j++){ // put zeros into the means and stdevs
			means.at(i).push_back(0);
			stddev.at(i).push_back(0);
		}	
	}

	// put stuff into labels
	for (int i = 0; i < ::featureVector.size(); i++){
		int index = ::featureVector.at(i).at(0);
		labelFeatures.at(index).push_back( ::featureVector.at(i) ) ;	
	}

	// calc mean
	for (int j = 0; j < labelFeatures.size(); j++){ // loop through labels
		labelf = labelFeatures.at(j);
		for (int k=0; k< labelf.size(); k++){ // loop through each vector in the label
			feature = labelf.at(k);
			for (int i = 0; i < feature.size()-1; i++){ // loop through the features of the vector
				means.at(feature.at(0)).at(i) += feature.at(i+1)/labelFeatures.at(j).size(); // mean at the label at the feature += the feature/#points
			}
		}
	}

	// calc stddev
	for (int j = 0; j < labelFeatures.size(); j++){ // loop through labels
		labelf = labelFeatures.at(j);
		for (int k=0; k< labelf.size(); k++){ // loop through each vector in label
			feature = labelf.at(k);
			for (int i = 0; i < feature.size()-1; i++){ // loop through the features
				// calc stdDev?
				stddev.at(feature.at(0)).at(i) += ((feature.at(i+1)-means.at(feature.at(0)).at(i)) * (feature.at(i+1)-means.at(feature.at(0)).at(i)))/labelf.size();
			}
		}
	}

	for (int i=0 ; i < stddev.size(); i++){
		for (int j=0 ; j < stddev.at(i).size(); j++){
			stddev.at(i).at(j) = sqrt(stddev.at(i).at(j));		}
	}

	std::vector<std::vector<std::vector<float>>> together;
	together.push_back(means);
	together.push_back(stddev);
	return together;
}


std::string compare(std::vector<float> features){
	std::vector<std::vector<float>> meanDev = calcMeanStdDev();
	std::vector<float> mean = meanDev.at(0);
	std::vector<float> stddev = meanDev.at(1);
	std::vector<float> diff;

	for (int i = 0; i < ::featureVector.size(); i++){
		diff.push_back(0);
	}

	for (int i = 0; i < ::featureVector.size(); i++){
		for (int j = 0; j < features.size(); j++){
			diff.at(i) += (features.at(j)-::featureVector.at(i).at(j+1))*(features.at(j)-::featureVector.at(i).at(j+1))/(stddev.at(j));
		}
	}
	int index = distance(diff.begin(), std::min_element(diff.begin(), diff.end()));
	float minLabelIndex = ::featureVector.at(index).at(0);
	std::string minLabel = ::labelVector.at(minLabelIndex);
	return minLabel;
}

void writeFeatures(std::vector<float> features, std::string filename, std::string object){
  	std::ofstream myfile;
  	myfile.open (filename, std::ios_base::app);
  	myfile << object << ", ";
	for (int i = 0; i < features.size(); i++){
  		myfile << features[i];


  		if (i != features.size() - 1)
  			myfile << ", ";
	}  	
	myfile << "\n";

  	myfile.close();
}

void readDatabase(std::string filename){
	std::string line;
	std::ifstream nameFileout;

	nameFileout.open(filename);
	if (!nameFileout){
		std::cout << "unable to open\n";
		exit(1);
	}

	while (std::getline(nameFileout, line))
	{
		std::stringstream ss(line);
		float feature;
		int index;

		std::string label;
		std::vector<float> featureLine;
		ss >> label;

		std::ptrdiff_t  i = std::find(::labelVector.begin(), ::labelVector.end(), label) - ::labelVector.begin();
		int next = 0;
		
		if (i == labelVector.size()){
			::labelVector.push_back(label);
		}
		
		index = i;

		featureLine.push_back(index);
		if (ss.peek() == ',' || ss.peek() == ' ')
			ss.ignore();

		while (ss >> feature){
			featureLine.push_back(feature);
			if (ss.peek() == ',' || ss.peek() == ' ')
				ss.ignore();
		}
		::featureVector.push_back(featureLine);

	    
	}
	nameFileout.close();
}

std::string calcProb(std::vector<float> features){
	std::vector<std::vector<std::vector<float>>> labelMeansStddevs = calcMeanStdDevLabels();
	std::vector<std::vector<float>> labelMeans = labelMeansStddevs[0];
	std::vector<std::vector<float>> labelDevs = labelMeansStddevs[1];
	std::vector<std::vector<float>> zscores;
	
	//put a bunch of empty vectors into zscores
	for (int labelI=0; labelI < labelMeans.size(); labelI++){
		zscores.push_back({});
	}

	//push back some math equation into each element in zscores
	for (int labelI=0; labelI < labelMeans.size(); labelI++){
		for (int featureI=0; featureI < features.size(); featureI++){
			// zscore = feature - mean / stddev
			zscores.at(labelI).push_back( std::abs((features.at(featureI) - labelMeans.at(labelI).at(featureI)) / labelDevs.at(labelI).at(featureI)));
		}
	}

	//average the zscores
	std::vector<float> avgZscores;
	for (int i=0; i < zscores.size(); i++){
		avgZscores.push_back(0);
		for (int j=0; j < zscores.at(i).size(); j++){
			avgZscores.at(i) += zscores.at(i).at(j) / zscores.at(i).size();
		}
	}
	// find min element of zscore
	int index = distance(avgZscores.begin(), std::min_element(avgZscores.begin(), avgZscores.end()));
	float zscore = avgZscores.at(index);
	std::string label = ::labelVector.at(index);
	std::cout << "zscore: " << zscore << "\n";
	if (zscore > 3){
		// if zscore > 3 not any label
		label = "?????";
	}


	return label;
	
}



int main(int argc, char *argv[]) {
	readDatabase("../data/data.txt");
    cv::Mat frame;
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

	// print out information about the image
	printf("filename:         %s\n", filename);
	printf("Image size:       %d rows x %d columns\n", (int)frame.size().height, (int)frame.size().width);
	printf("Image dimensions: %d\n", (int)frame.channels());
	printf("Image depth:      %d bytes/channel\n", (int)frame.elemSize()/frame.channels());

	cv::namedWindow("Image", 1); // identifies a window?

	// create a window
	namedWindow("Connected Components",1);
	//declaring variables
	Mat img = frame;
	Mat output,	gray, blur, threshout, openMat, connMat, normedMat, subMat;
	
	//convert the image to grayscale
	cvtColor(img, gray, CV_BGR2GRAY);
	GaussianBlur(gray,blur,Size(7,7),0,0);
	
	//select grayscale values above 87 and turn them white
	threshold(blur, threshout, 113, 255, THRESH_BINARY_INV);
	morphologyEx(threshout, openMat, MORPH_CLOSE, getStructuringElement(MORPH_RECT,Size(20,20)));
	int regions = connectedComponents(openMat, connMat, 8);

	std::vector<Point> points = findRegionPoints(connMat, regions);
	// find points to putText
	std::vector<float> features;

	if (regions != 0){
		normalize(connMat, normedMat, 0, 255, NORM_MINMAX, CV_8UC1);
		for (int regionI=1; regionI < regions; regionI++){
			features = calcFeatures(openMat, connMat, regions, regionI);
			float hc=features[0];
			float vc=features[1];
			float hwr=features[2];
			float fill=features[3];
			std::string label = compare(features);
			std::string probLabel = calcProb(features);
			label = label.substr(0, label.size() - 1);
			probLabel = probLabel.substr(0, probLabel.size() - 1);
			std::cout << "Region: " << regionI << "\n";
			std::cout << "Label: " << label << "\nProbLabel: " << probLabel << "\n";
			std::cout << "Height:Width Ratio: " << hwr << "\n";
			std::cout << hc << "\n" << vc << "\n" << fill << "\n";

			// std::string result = "Horizontal Moment" + std::to_string(hc) + "   Vertical Moment" + std::to_string(vc);
			// putText(frame, result, Point(0,frame.rows), FONT_HERSHEY_PLAIN, frame.cols/500, Scalar({255,0,255}),frame.cols/500 );

			putText(frame, probLabel+"|"+label, points.at(regionI-1), FONT_HERSHEY_PLAIN, frame.cols/200, Scalar({255,0,255}),frame.cols/300 );
		}
		
	}
	
	while(true){
		//write the grayscale image and the binary image to files
		imshow("Connected Components", normedMat);
		cv::imshow("Image", frame);

		int key = cv::waitKey(10);
		
		switch(key) {
		case 'q':
		    break;


		case 'n'://write to features text file if the user hits n
			{
				Mat textFrame = frame.clone();
				std::string object;
				std::string character;
				while (key != 13 && key != 10){ // enter or carriage return
					key = waitKey(0);
					character = char(key);
					if (key > 0 && key < 127 && key != 13 && key != 10 && key != 8 && key != 32 && key != 27){
						object.append(character);
						putText(textFrame, object, Point(frame.cols/25,frame.rows/2), FONT_HERSHEY_PLAIN, frame.cols/200, Scalar({255,0,255}),frame.cols/200 );
						imshow("Video", textFrame);
						std::cout << object << "\n";
					}
					else if (key == 8){ // backspace
						textFrame = frame.clone();
						//imshow("Video", textFrame);
						object = object.substr(0, object.size() - 1);
						putText(textFrame, object, Point(frame.cols/25,frame.rows/2), FONT_HERSHEY_PLAIN, frame.cols/200, Scalar({255,0,255}),frame.cols/200 );
						imshow("Video", textFrame);
						std::cout << "Backspace: " << object << "\n";
					}
					else if (key == 27){ // escape
						goto escape;
					}
				}
				std::cout << object << "\n";
				writeFeatures(features, filename, object);	
				escape:			
				break;
			}
		default:
		    break;
		}

		}

		// terminate the video capture
		printf("Terminating\n");

		return(0);
	}