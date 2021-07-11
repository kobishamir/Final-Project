// Calibration.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
using namespace cv;
using namespace std;

bool first_click = true;
bool new_line = false;
Point first;
Point second;
Mat frame;
stringstream coord;
static void onMouse(int event, int x, int y, int, void*)
{

    if (event != EVENT_LBUTTONDOWN)
        return;

    if (first_click)
    {
        first_click = false;
        first = Point(x, y);
        new_line = false;
        coord.clear();
    }
    else
    {
        first_click = true;
        second = Point(x, y);
        new_line = true;
        coord.clear();
        coord << "(" << first.x << ", " << first.y << "), " << "(" << second.x << ", " << second.y << ")";
        cout << "(" << first.x << ", " << first.y << "), " << "(" << second.x << ", " << second.y << ")" << endl;

        ofstream MyFile("Calibration_coordinate.txt");
        MyFile << (first.x + second.x) / 2 << " " << (first.y + second.y) / 2 << endl;
        MyFile.close();
    }
}


int main()
{
    //VideoCapture cap(0);
    VideoCapture cap("C:\\Open-CV\\Test\\Test\\Test\\Eytan_1.avi");
    namedWindow("image", 0);
    setMouseCallback("image", onMouse, 0);
    while (1) { // choose between 0-COCO, 1-MPI or 2-HAND
        cap >> frame;
        // If the frame is empty, break immediately
        if (frame.empty())
            break;
        if (new_line)
        {
            line(frame, first, second, Scalar(0, 200, 0), 2);
            cv::putText(frame, //target image
                coord.str(), //text
                cv::Point(10, frame.rows / 2), //top-left position
                cv::FONT_HERSHEY_DUPLEX,
                1.0,
                CV_RGB(118, 185, 0), //font color
                2);
            coord.clear();
        }
        imshow("image", frame);

        // Press  ESC on keyboard to exit
        char c = (char)waitKey(60); // use 1 for USB camera, use 60 for videoclip
        if (c == 27)
            break;
        
        //printf("%d\n", first.y);
    }

    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    destroyAllWindows();
    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
