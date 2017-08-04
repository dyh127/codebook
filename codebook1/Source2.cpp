

//opencv
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/features2d.hpp>
//#include <features2d.hpp>
#include <features2d/features2d.hpp>
#include <algorithm>
//C
#include <stdio.h>
//C++
#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;

#define CHANNELS 3
typedef struct ce {
	uchar learnHigh[CHANNELS];
	//学习上限
	uchar learnLow[CHANNELS];
	//学习下限
	uchar min[CHANNELS];
	//属于此码元各通道的最小值
	uchar max[CHANNELS];
	//属于此码元各通道的最大值
	int t_last_update;
	//此码元最后一次更新的时间
	int t_create;
	//此码元产生的时间
	int stale;
	//最长不更新的时间
}code_element;

typedef struct code_book {
	code_element **cb;
	//指向码元数组的指针
	int numEntries;
	//此码本中码元的数量
	int t;
	//此码本现在的时间
}codeBook;

int cvupdateCodeBook(uchar *p, codeBook &c, unsigned *cbBounds, int numChannels)
{
	int n;
	unsigned int high[3], low[3];
	for (n = 0; n < numChannels; n++)
	{
		high[n] = *(p + n) + *(cbBounds + n);
		//high值为当前值加上弹性范围的数
		if (high[n] > 255 || high[n]<0) high[n] = 255;
		low[n] = *(p + n) - *(cbBounds + n);
		//同理，low值为当前值加上弹性范围的数
		if (low[n] < 0 || low[n]>255)low[n] = 0;
	}
	int matchChannel;
	int i;
	for (i = 0; i < c.numEntries; i++)
	{//当一个点的像素值来时，判断是在原有的codebook中还是需要创立一个新的codebook
		matchChannel = 0;
		for (n = 0; n < numChannels; n++)
		{
			if (((c.cb[i]->learnLow[n]) <= *(p + n)) && (*(p + n) <= (c.cb[i]->learnHigh[n])))
			{
				matchChannel++;
			}
		}
		if (matchChannel == numChannels)
		{//如果三个通道都满足，那么说明在某个codeelement中
			c.cb[i]->t_last_update = c.t;//更新更新的时间
			for (n = 0; n < numChannels; n++)
			{
				if (c.cb[i]->max[n] < *(p + n))
					c.cb[i]->max[n] = *(p + n);
				else if (c.cb[i]->min[n] > *(p + n))
					c.cb[i]->min[n] = *(p + n);
			}
			break;
		}
	}
	if (i == c.numEntries)
	{//如果循环中没有找到匹配的codeelement，则需要重新创建一个codeelement
		code_element **foo = new code_element*[c.numEntries + 1];
		for (int ii = 0; ii < c.numEntries; ii++)
			foo[ii] = c.cb[ii];
		foo[c.numEntries] = new code_element;
		if (c.numEntries)delete[] c.cb;
		c.cb = foo;
		for (n = 0; n < numChannels; n++)
		{
			c.cb[c.numEntries]->learnHigh[n] = high[n];
			c.cb[c.numEntries]->learnLow[n] = low[n];
			c.cb[c.numEntries]->max[n] = *(p + n);
			c.cb[c.numEntries]->min[n] = *(p + n);
		}
		c.cb[c.numEntries]->t_last_update = c.t;
		c.cb[c.numEntries]->stale = 0;
		c.cb[c.numEntries]->t_create = c.t;
		c.numEntries += 1;
	}
	for (int s = 0; s < c.numEntries; s++)
	{//对每个像素点的codeelement进行一个stale的更新
		int negRun = c.t - c.cb[s]->t_last_update;
		if (c.cb[s]->stale < negRun)
			c.cb[s]->stale = negRun;
	}
	for (n = 0; n < numChannels; n++)
	{//自动调节学习范围
		if (c.cb[i]->learnHigh[n] < high[n])
			c.cb[i]->learnHigh[n] += 1;
		if (c.cb[i]->learnLow[n] > low[n])
			c.cb[i]->learnLow[n] -= 1;
	}
	return(i);
}
//每来一帧的图像的一个像素点的信息，更新该点的codebook的相关值，（创建和更新），返回当前像素点所在的codeelement的编号
uchar cvbackgroundDiff(uchar *p, codeBook &c, int numChannels, int *minMod, int *maxMod)
{
	int matchChannel;
	int i;
	int t[3];
	for (i = 0; i < c.numEntries; i++)
	{
		matchChannel = 0;
		for (int k = 0; k < 3; k++)
		{
			t[k] = 0;
		}
		for (int n = 0; n < numChannels; n++)
		{
			if (((c.cb[i]->min[n] - minMod[n]) <= *(p + n)) && (*(p + n) <= (c.cb[i]->max[n] + maxMod[n])))
			{
				matchChannel++;
				t[n]++;
			}
			else
				break;
		}
		if (t[0] == 0 && t[1] == 1 && t[2] == 1)
		{
			matchChannel = 3;
		}
		if (matchChannel == numChannels)//如果发现匹配，则跳出循环
			break;
	}
	if (i == c.numEntries)//如果没有找到匹配的，则说明是前景
		return(255);
	else
		return(0);//找到了，则说明是背景
}
//没来一帧图像的一个像素点的信息，返回0/255，来标定这是前景还是背景。
int cvclearStaleEntries(codeBook &c)
{
	int staleThresh = 50;
	//标定过期时间
	int *keep = new int[c.numEntries];
	int keepCnt = 0;
	if (!c.numEntries)
		return 0;
	for (int i = 0; i < c.numEntries; i++)
	{
		if (c.cb[i]->stale > staleThresh)
			keep[i] = 0;
		else
		{
			keep[i] = 1;
			keepCnt += 1;
		}
	}
	c.t = 0;
	code_element **foo = new code_element*[keepCnt];
	int k = 0;
	for (int ii = 0; ii < c.numEntries; ii++)
	{
		if (keep[ii])
		{
			foo[k] = c.cb[ii];
			//
			foo[k]->stale = c.cb[ii]->stale;
			foo[k]->t_last_update = c.cb[ii]->t_last_update;
			foo[k]->t_create = c.cb[ii]->t_create;
			//
			k++;
		}
	}
	delete[] keep;
	delete[] c.cb;
	c.cb = foo;
	int numCleared = c.numEntries - keepCnt;
	c.numEntries = keepCnt;
	return(numCleared);
}
//对每个codebook进行codeelement的删除，返回删除的数值
#define DP_EPSILON_DENOMINATOR 20.0
#define CVCLOSE_ITR 1

//void findConnectedComponents(
//	cv::Mat& mask,//输入的图像，对该图像进行修改
//	int poly1_hull0,
//	float perimScale,
//	vector<cv::Rect>& bbs,
//	vector<cv::Point>& centers
//) {
//	cv::morphologyEx(
//		mask, mask, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1, -1), CVCLOSE_ITR
//	);
//	cv::morphologyEx(
//		mask, mask, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), CVCLOSE_ITR
//	);//利用形态学去除噪声
//	vector<vector<cv::Point>>contours_all;//所有发现的轮廓
//	vector<vector<cv::Point>>contours;//我们需要的那个轮廓
//	cv::findContours(mask, contours_all, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
//	//寻找轮廓
//	for (
//		vector<vector<cv::Point>>::iterator c = contours_all.begin();
//		c != contours.end();
//		++c
//		) {//找出最佳的轮廓，并用多边形近似其余的部分
//		int len = cv::arcLength(*c, true);//这个轮廓的长度
//		double q = (mask.rows + mask.cols) / DP_EPSILON_DENOMINATOR;
//		if (len >= q) {
//			vector<cv::Point> c_new;
//			if (poly1_hull0)
//			{
//				cv::approxPolyDP(*c, c_new, len / 20.0, true);
//			}
//			else {
//				cv::convexHull(*c, c_new);
//			}
//			contours.push_back(c_new);
//		}
//	}
//	const cv::Scalar CVX_WHITE = cv::Vec3b(0xff, 0xff, 0xff);
//	const cv::Scalar CVX_BLACK = cv::Vec3b(0X00, 0X00, 0X00);
//	int idx = 0;
//	cv::Moments moments;
//	cv::Mat scratch = mask.clone();
//	for (
//		vector<vector<cv::Point>>::iterator c = contours.begin();
//		c != contours.end;
//		c++, idx++
//		) {
//		cv::drawContours(scratch, contours, idx, CVX_WHITE, CV_FILLED);
//
//		moments = cv::moments(scratch, true);
//		cv::Point p;
//		p.x = (int)(moments.m10 / moments.m00);
//		p.y = (int)(moments.m01 / moments.m00);
//		centers.push_back(p);
//
//		bbs.push_back(cv::boundingRect(contours.size()));
//
//		scratch.setTo(0);
//	}
//	mask.setTo(0);
//	cv::drawContours(mask, contours, -1, CVX_WHITE);
//}

#define CVCONTOUR_APPROX_LEVEL 2
void find_connected_components(
	IplImage* mask,
	int poly_hull0,
	float perimScales,
	CvRect* bbs,
	CvPoint* centers
)
{
	static CvMemStorage* mem_storage = NULL;
	static CvSeq* contours = NULL;
	//CLEAN UP RAW MASK
	//
	cvMorphologyEx(mask, mask, 0, 0, CV_MOP_OPEN, CVCLOSE_ITR);
	cvMorphologyEx(mask, mask, 0, 0, CV_MOP_CLOSE, CVCLOSE_ITR);


	//FIND CONTOURS AROUND ONLY BIGGER REGIONS
	//
	if (mem_storage == NULL)
	{
		mem_storage = cvCreateMemStorage(0);
	}
	else
	{
		cvClearMemStorage(mem_storage);
	}


	CvContourScanner scanner = cvStartFindContours(mask, mem_storage, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);


	CvSeq* c;
	int numCont = 0;
	while ((c = cvFindNextContour(scanner)) != NULL)
	{
		double len = cvContourPerimeter(c);


		//calculate perimeter len threshold;
		//
		double q = (mask->height + mask->width) / perimScales / 4;


		//Get rid of blob if its perimeter is too small:
		//
		if (len < q)
		{
			cvSubstituteContour(scanner, NULL);
		}
		else
		{
			//Smooth its edges if its large enough
			//
			CvSeq* c_new;
			if (poly_hull0)
			{
				//Polygonal approximation
				//
				c_new = cvApproxPoly(c, sizeof(CvContour), mem_storage, CV_POLY_APPROX_DP, CVCONTOUR_APPROX_LEVEL);
			}
			else
			{
				//Convex hull of the segmentation
				//
				c_new = cvConvexHull2(c, mem_storage, CV_CLOCKWISE, 1);
			}
			cvSubstituteContour(scanner, c_new);//新处理的图像序列代替原序列
			numCont++;
		}
	}
	contours = cvEndFindContours(&scanner);


	//Just some convenience variables
	const CvScalar CVX_WHITE = CV_RGB(0xff, 0xff, 0xff);
	const CvScalar CVX_BLACE = CV_RGB(0x00, 0x00, 0x00);
	//PAINT THE FOUND REGIONS BACK INTO THE IMAGE
	//
	cvZero(mask);
	IplImage *maskTemp;
	//CALC CENTER OF MASS AND/OR BOUNDING RECTANGLE
	//
	//User wants to collect statistics
	int i = 0;
	CvMoments moments;
	double M00, M01, M10;
	maskTemp = cvCloneImage(mask);
	for (i = 0, c = contours; c != NULL; c = c->h_next, i++)
	{
		//only process up to *num of them
		//
		cvDrawContours(maskTemp, c, CVX_WHITE, CVX_WHITE, -1, CV_FILLED, 8);


		//Find the center of each contour
		//
		if (centers != NULL)
		{
			cvMoments(maskTemp, &moments, 1);
			M00 = cvGetSpatialMoment(&moments, 0, 0);
			M10 = cvGetSpatialMoment(&moments, 1, 0);
			M01 = cvGetSpatialMoment(&moments, 0, 1);
			centers[i].x = (int)(M10 / M00);
			centers[i].y = (int)(M01 / M00);
		}


		//Bounding rectangles around blobs
		//
		if (bbs != NULL)
		{
			bbs[i] = cvBoundingRect(c);
		}
		cvZero(maskTemp);


		//Draw filled contours into mask
		//
		cvDrawContours(mask, c, CVX_WHITE, CVX_WHITE, -1, CV_FILLED, 8);
	}
	cvReleaseImage(&maskTemp);
}
void blob(
	IplImage* mask
)
{
	using namespace cv;
	Mat im = cv::cvarrToMat(mask, false);

	// Set up the detector with default parameters.
	SimpleBlobDetector::Params params;
	params.minThreshold = 0;
	params.maxThreshold = 255;
	params.thresholdStep = 255;
	params.filterByArea = 1;
	params.minArea = 100;
	params.maxArea = 8000;
	params.filterByConvexity = 1;
	params.minConvexity = .05f;
	params.filterByInertia = 1;
	params.minInertiaRatio = .05f;

	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	// Detect blobs.
	std::vector<KeyPoint> keypoints;
	detector->detect(im, keypoints);

	// Draw detected blobs as red circles.
	// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
	Mat im_with_keypoints;
	drawKeypoints(im, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	// Show blobs
	//imshow("keypoints", keypoints);

}

int main()
{
	int a;
	Mat mask;
	Mat mask2;
	Mat raw;
	CvCapture* capture;
	IplImage* rawImage;
	IplImage* yuvImage;
	IplImage* ImaskCodeBook;
	IplImage* mask1;
	codeBook* cB;
	codeBook* adp;
	unsigned cbBounds[CHANNELS];
	uchar* pColor;
	int imageLen;
	int nChannels = CHANNELS;
	int minMod[CHANNELS];
	int maxMod[CHANNELS];
	int he;
	int wi;
	int existTime;
	CvScalar s;
	int matchChannel;
	uchar* s1 = new uchar[3];

	/*cvNamedWindow("Raw");
	cvNamedWindow("CodeBook");
	cvNamedWindow("CodeBook1");
	cvNamedWindow("blob");*/

	pColor = new uchar[480000];
	capture = cvCreateFileCapture("camera6.avi");
	if (!capture)
	{
		printf("Couldn't open the video!");
		return -1;
	}
	rawImage = cvQueryFrame(capture);
	yuvImage = cvCreateImage(cvGetSize(rawImage), 8, 3);
	//给yuvimage分配一个与rawimage尺寸相同，8位3通道图像
	ImaskCodeBook = cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_8U, 1);
	//给imaskcodebook分配一个rawimage尺寸相同，8位单通道图像
	cvSet(ImaskCodeBook, cvScalar(255));
	//初始化IMaskcodebook，为白色图像

	imageLen = rawImage->width*rawImage->height;
	//得到每张图像的像素点
	cB = new codeBook[imageLen];
	//初始化与像素点数量相同的codebook
	for (int i = 0; i < imageLen; i++)
		cB[i].numEntries = 0;
	for (int i = 0; i < nChannels; i++) {
		cbBounds[i] = 5;//用于确定各通道的阈值
		minMod[i] = 10;
		maxMod[i] = 10;
	}
	cbBounds[0] = 10;
	minMod[0] = 10;
	maxMod[0] = 10;
	cbBounds[1] = 5;
	minMod[1] = 5;
	maxMod[1] = 5;
	cbBounds[2] = 5;
	minMod[2] = 5;
	maxMod[2] = 5;

	//选取纯背景进行学习（手动操作）
	/*int codeelementnum;
	int* is_inside;
	is_inside = new int[480000];
	for (int i = 0; i < 480000; i++)
	{
	is_inside[i] = 1;
	}*/
	for (int i = 0;; i++)
	{
		if (i <= 1140 && i >= 600)
		{
			cvCvtColor(rawImage, yuvImage, CV_BGR2YCrCb);
			pColor = (uchar*)yuvImage->imageData;
			//色彩空间转换，将rawimage转换到yuv色彩空间中，输出yuvimage

			for (int c = 0; c < imageLen; c++) {
				/*if (is_inside[c] == 1)
				{
				codeelementnum = cvupdateCodeBook(pColor, cB[c], cbBounds, nChannels);
				if (codeelementnum >= 2)
				{
				is_inside[c] = 0;
				}
				}*/
				cB[c].t = i;
				cvupdateCodeBook(pColor, cB[c], cbBounds, nChannels);
				pColor += 3;
			}
			if (i % 100 == 0)
			{
				for (int c = 0; c < imageLen; c++)
					cvclearStaleEntries(cB[c]);
			}
		}
		if (!(rawImage = cvQueryFrame(capture)))
			break;
	}
	adp = new codeBook[imageLen];
	for (int c = 0; c < imageLen; c++)
	{
		adp[c].numEntries = 0;
	}
	//
	//进行前背景分离
	capture = cvCreateFileCapture("camera6.avi");
	rawImage = cvQueryFrame(capture);
	for (int i = 0;; i++)
	{
		if (1)
		{
			cvCvtColor(rawImage, yuvImage, CV_BGR2YCrCb);
			uchar maskPixelCodeBook;
			pColor = (uchar*)((yuvImage)->imageData);
			uchar *pMask = (uchar *)((ImaskCodeBook)->imageData);
			for (int c = 0; c < imageLen; c++)
			{
				for (int m = 0; m < cB[c].numEntries; m++)
				{
					matchChannel = 0;
					for (int n = 0; n < CHANNELS; n++)
					{
						if (((cB[c].cb[m]->learnLow[n]) <= *(pColor + n)) && (*(pColor + n) <= (cB[c].cb[m]->learnHigh[n])))
						{
							matchChannel++;
						}
					}
					if (matchChannel == CHANNELS)
					{
						cB[c].t = i;
						cvupdateCodeBook(pColor, cB[c], cbBounds, nChannels);
					}
					else
					{
						adp[c].t = i;
						cvupdateCodeBook(pColor, adp[c], cbBounds, nChannels);
					}
				}
				pColor += 3;
			}
			if (i % 25 == 0)
			{
				for (int c = 0; c < imageLen; c++)
				{
					cvclearStaleEntries(cB[c]);
					cvclearStaleEntries(adp[c]);
				}
			}
			if (i % 100 == 0)
			{
				for (int c = 0; c < imageLen; c++)
				{
					for (int ii = 0; ii < adp[c].numEntries; ii++)
					{
						existTime = i - adp[c].cb[ii]->t_create;
						if (existTime > 200)
						{
							code_element **foo = new code_element*[cB[c].numEntries + 1];
							for (int ii = 0; ii < cB[c].numEntries; ii++)
								foo[ii] = cB[c].cb[ii];
							foo[cB[c].numEntries] = new code_element;
							if (cB[c].numEntries)
								delete[] cB[c].cb;
							cB[c].cb = foo;
							for (int n = 0; n < CHANNELS; n++)
							{
								cB[c].cb[cB[c].numEntries]->learnHigh[n] = adp[c].cb[ii]->learnHigh[n];
								cB[c].cb[cB[c].numEntries]->learnLow[n] = adp[c].cb[ii]->learnLow[n];
								cB[c].cb[cB[c].numEntries]->max[n] = adp[c].cb[ii]->max[n];
								cB[c].cb[cB[c].numEntries]->min[n] = adp[c].cb[ii]->min[n];
							}
							cB[c].cb[cB[c].numEntries]->t_last_update = adp[c].cb[ii]->t_last_update;
							cB[c].cb[cB[c].numEntries]->stale = 0;
							cB[c].cb[cB[c].numEntries]->t_create = adp[c].cb[ii]->t_create;
							cB[c].numEntries += 1;
						}
					}
				}
			}
			pColor = (uchar*)((yuvImage)->imageData);
			for (int c = 0; c < imageLen; c++)
			{
				/*if (is_inside[c] == 1)
				{
				maskPixelCodeBook = cvbackgroundDiff(pColor, cB[c], nChannels, minMod, maxMod);
				*(pMask++) = maskPixelCodeBook;
				}
				else
				{
				*(pMask++) = 0;
				}*/
				he = c / rawImage->width;
				wi = c % rawImage->width;
				if ((he >= 175) && (he >= ((wi - 605) * 14 / 30 + 175)) && (he <= ((wi - 420)*-53 / 300 + 520))&& (he <= ((wi - 150)*-33 / 270 + 553)))
				{
					maskPixelCodeBook = cvbackgroundDiff(pColor, cB[c], nChannels, minMod, maxMod);
					*(pMask++) = maskPixelCodeBook;
				}
				else
				{
					*(pMask++) = 0;
				}
				pColor += 3;
			}
			//联通
			mask1 = cvCloneImage(ImaskCodeBook);
			CvRect bbs[100];
			CvPoint center1[100];
			find_connected_components(mask1, 0, 4, bbs, center1);
			//
			//筛选
			//blob(mask1);
			//
			//cvShowImage("CodeBook1", mask1);
			//
			//加框
			vector<vector<Point>> contours;
			Mat im = cv::cvarrToMat(mask1, false);
			findContours(im, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

			vector<vector<Point> > contours_poly(contours.size());
			vector<Rect> boundRect(contours.size());
			vector<Point2f>center(contours.size());
			vector<float>radius(contours.size());

			for (int i = 0; i < contours.size(); i++)
			{
				approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
				boundRect[i] = boundingRect(Mat(contours_poly[i]));
			}
			//Mat drawing = Mat::zeros(im.size(), CV_8UC3);

			for (int i = 0; i < contours.size(); i++)
			{
				Scalar color = Scalar(255, 0, 255);
				raw = cv::cvarrToMat(rawImage, false);
				drawContours(raw, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
				rectangle(raw, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
			}
			//
			//在原图像中显示帧数
			raw = cv::cvarrToMat(rawImage, false);
			rectangle(raw, cv::Point(10, 2), cv::Point(100, 20),
				cv::Scalar(255, 255, 255), -1);
			string frameNumberString = to_string(i);
			CvFont* font;
			putText(raw, &frameNumberString[0], cv::Point(15, 15), FONT_HERSHEY_SCRIPT_COMPLEX, .5f, (0, 0, 0), 1, 8, false);
			//

			/*		CvScalar s;
			s.val[0] = 255;
			s.val[1] = 0;
			s.val[2] = 0;
			for (he = 550; he < 555; he++)
			{
			for (wi = 0; wi < 800; wi++)
			{
			cvSet2D(rawImage, he, wi, s);
			}
			}*/
			//cvShowImage("rawimage", rawImage);
			imshow("Raw", raw);
			//cvShowImage("yuvimage", yuvImage);
			CvScalar s;
			s.val[0] = 255;
			s.val[1] = 255;
			s.val[2] = 255;
			for (he = 0; he < 600; he++)
			{
				for (wi = 0; wi < 800; wi++)
				{
					if ((he == 175) || (he == ((wi - 605) * 14 / 30 + 175)) || (he == ((wi - 420)*-53 / 300 + 520)) || (he == ((wi - 150)*-33 / 270 + 553)))
					{
						cvSet2D(ImaskCodeBook, he, wi, s);
					}
				}
			}
			cvShowImage("CodeBook", ImaskCodeBook);


			//存储结果
			/*if (i > 1200)
			{
			mask = cv::cvarrToMat(ImaskCodeBook, false);
			std::string savingName1 = "E:\\summer research\\data\\output\\" + std::to_string(i) + ".jpg";
			cv::imwrite(savingName1, mask);

			mask2 = cv::cvarrToMat(mask1, false);
			std::string savingName2 = "E:\\summer research\\data\\output\\liantong" + std::to_string(i) + ".jpg";
			cv::imwrite(savingName2, mask2);
			}*/

			/*if (i <= 500 )
			{
			std::string savingName1 = "E:\\summer research\\data\\output\\codebook_" + std::to_string(i) + ".jpg";
			cv::imwrite(savingName1, raw);
			}*/

			if (cvWaitKey(30) == 27)
				break;
			/*	if ( i == 200 )
			cvWaitKey();*/
		}
		if (!(rawImage = cvQueryFrame(capture)))
			break;
	}
	cvReleaseCapture(&capture);
	if (yuvImage)
		cvReleaseImage(&yuvImage);
	if (ImaskCodeBook)
		cvReleaseImage(&ImaskCodeBook);
	cvDestroyAllWindows();
	delete[] cB;

	return 0;
}

