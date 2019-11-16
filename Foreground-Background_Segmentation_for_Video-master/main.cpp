#include "SCGMM_JT.h"


cv::Mat origImg, segImg, gtImg;

cv::Mat fgdSamples, bgdSamples;

int main() {
	vector<cv::Mat> origImgs, gtImgs;

	int numOfImages = 5;
	string origImgsPath = "/media/akhil/Code1/Studies/MA20057-Sem-9/MTP/Data/L2V1D1R1Copy/trainingData/editedResults/original";
	string gtImgsPath = "/media/akhil/Code1/Studies/MA20057-Sem-9/MTP/Data/L2V1D1R1Copy/trainingData/editedResults/maskRCNNResultsBW";
	string outputPath = "/media/akhil/Code1/Studies/MA20057-Sem-9/MTP/Data/L2V1D1R1Copy/trainingData/editedResults/modelsResultsBW";

	cout << "Reading " << numOfImages << " original and ground truth images... ";
	getImgSeqFromDir(origImgsPath.c_str(), gtImgsPath.c_str(), (numOfImages), origImgs, gtImgs);
	cout << "Done!" << endl;
	origImg = origImgs.front();
	gtImg = gtImgs.front(); 
	//cout<<(origImg.size);
	//return 0;
	//cv::namedWindow("original");
	//cv::namedWindow("ground truth");

	//cv::imshow("original", origImg);
	//cv::imshow("ground truth", gtImg);
	/*
	 * Use the whole image to be our training samples for initialization.
	 */
	//cout<<"sssssssss";
	for (int i = 0; i < gtImg.rows; i++) {
		for (int j = 0; j < gtImg.cols; j++) {
			cv::Mat sample = (cv::Mat_<double>(1, 5) << j, i, origImg.at<cv::Vec3b>(i, j)[0], origImg.at<cv::Vec3b>(i, j)[1], origImg.at<cv::Vec3b>(i, j)[2]);
			sample.convertTo(sample, CV_32F);
			//cout<<gtImg.at<cv::Vec3b>(i, j);
			if (gtImg.at<cv::Vec3b>(i, j) == cv::Vec3b(255, 255, 255))
				fgdSamples.push_back(sample);
			else
				bgdSamples.push_back(sample);
		}
	}

	while (1) {
		int key = 's';//cv::waitKey(1);
		if (key == 27)
			break;
		else if (key == 's') {
			clock_t start, end;
			SCGMM_JT model;

			cout << "Initializing SCGMM Joint Tracking... ";
			start = clock();
			//cout<<fgdSamples.type();
			model.init(origImg, fgdSamples, bgdSamples);
			end = clock();
			cout << "Done! (" << (end - start) / (double)CLOCKS_PER_SEC << "s)" << endl;

			for (int imgIndex = 0; imgIndex < origImgs.size(); imgIndex++) {
				cout << "Processing frame " << imgIndex << "... ";
				start = clock();
				model.run(origImgs.at(imgIndex), segImg);
				end = clock();
				cout << "Done! (" << (end - start) / (double)CLOCKS_PER_SEC << "s)" << endl;
				cout << "Accuracy: " << calcAccuracy(segImg, gtImgs.at(imgIndex)) << endl;

				stringstream ss;
				ss << outputPath <<"/"<< imgIndex << ".png";
				cout<<ss.str();
				cv::imwrite(ss.str(), segImg);
			}
		}
		break;
	}

	return 0;
}