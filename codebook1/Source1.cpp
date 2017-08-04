////opencv
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/imgproc.hpp"
//#include "opencv2/videoio.hpp"
//#include <opencv2/highgui.hpp>
//#include <opencv2/video.hpp>
////#include <opencv2/features2d/features2d.hpp>
////#include <opencv2/features2d.hpp>
////#include <features2d.hpp>
//#include <features2d/features2d.hpp>
//#include <algorithm>
////C
//#include <stdio.h>
////C++
//#include <iostream>
//#include <sstream>
//
//using namespace cv;
//using namespace std;
//
//int main()
//{
//	int he;
//	int wi;
//	CvCapture* capture;
//	IplImage* rawImage;
//	capture = cvCreateFileCapture("camera6.avi");
//	rawImage = cvQueryFrame(capture);
//	for (int i = 0;; i++)
//	{
//		CvScalar s;
//		s.val[0] = 255;
//		s.val[1] = 255;
//		s.val[2] = 255;
//		for (he = 0; he < 600; he++)
//		{
//			for (wi = 0; wi < 800; wi++)
//			{
//				if(he==553)
//				{ 
//					cvSet2D(rawImage, he, wi, s);
//				}
//				if (wi == 150)
//				{
//					cvSet2D(rawImage, he, wi, s);
//				}
//				if (he == ((wi - 150) * -33 / 270 + 553))
//				{
//					cvSet2D(rawImage, he, wi, s);
//				}
//				if (he == ((wi - 420) * -53 / 300 + 520))
//				{
//					cvSet2D(rawImage, he, wi, s);
//				}
//			}
//		}
//		cin.get();
//		cvShowImage("rawimage", rawImage);
//		if (cvWaitKey(30) == 27)
//			break;
//		if (!(rawImage = cvQueryFrame(capture)))
//			break;
//	}
//}