//
//  this sample demonstrates the use of pretrained openpose networks with opencv's dnn module.
//
//  it can be used for body pose detection, using either the COCO model(18 parts):
//  http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
//  https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/openpose_pose_coco.prototxt
//
//  or the MPI model(16 parts):
//  http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel
//  https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/openpose_pose_mpi_faster_4_stages.prototxt
//
//  (to simplify this sample, the body models are restricted to a single person.)
//
//
//  you can also try the hand pose model:
//  http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/pose_iter_102000.caffemodel
//  https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/hand/pose_deploy.prototxt
//
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/core/core.hpp>

#include <cmath>
#include <typeinfo>
using namespace cv;
using namespace cv::dnn;
#include <iostream>
using namespace std;
// connection table, in the format [model_id][pair_id][from/to]
// please look at the nice explanation at the bottom of:
// https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
//
const int POSE_PAIRS[3][20][2] = {
{   // COCO body
    {1,2}, {1,5}, {2,3},
    {3,4}, {5,6}, {6,7},
    {1,8}, {8,9}, {9,10},
    {1,11}, {11,12}, {12,13},
    {1,0}, {0,14},
    {14,16}, {0,15}, {15,17}
},
{   // MPI body
    {0,1}, {1,2}, {2,3},                    //{head,neck},{neck,Rshoulder},{Rshoulder,Relbow},
    {3,4}, {1,5}, {5,6},                    //{elbow,Rhand},{neck,Lshoulder},{Lshoulder,Lelbow},
    {6,7}, {1,14}, {14,8}, {8,9},           //{Lelbow,Lhand},{neck,torso},{torso,Rhip},{Rhip,Rknee},
    {9,10}, {14,11}, {11,12}, {12,13}       //{Rknee,Rabkle},{torso,Lhip},{Lhip,Lknee},{Lknee,Lankle},
},
{   // hand
    {0,1}, {1,2}, {2,3}, {3,4},         // thumb
    {0,5}, {5,6}, {6,7}, {7,8},         // pinkie
    {0,9}, {9,10}, {10,11}, {11,12},    // middle
    {0,13}, {13,14}, {14,15}, {15,16},  // ring
    {0,17}, {17,18}, {18,19}, {19,20}   // small
} };


/////////////////////////////// my edit:

double distance(double x1, double x2, double y1, double y2)
{
    double power, dis;

    power = pow( x1 - x2, 2) + pow(y1 - y2, 2);
    dis = sqrt(power);
    
    return dis;
}

double degree(double x1, double x2, double y1, double y2)
{
    double deg, pi = 3.141592;
    deg = atan2(y2 - y1, x2 - x1) * (180 / pi);

    return deg;
}

/////////////////////////////// end myedit


int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv,
        "{ h help           | false     | print this help message }"
        "{ p proto          |           | (required) model configuration, e.g. hand/pose.prototxt }"
        "{ m model          |           | (required) model weights, e.g. hand/pose_iter_102000.caffemodel }"
        "{ i image          |           | (required) path to image file (containing a single person, or hand) }"
        "{ d dataset        |           | specify what kind of model was trained. It could be (COCO, MPI, HAND) depends on dataset. }"
        "{ width            |  368      | Preprocess input image by resizing to a specific width. }"
        "{ height           |  368      | Preprocess input image by resizing to a specific height. }"
        "{ t threshold      |  0.1      | threshold or confidence value for the heatmap }"
        "{ s scale          |  0.003922 | scale for blob }"
    );

    String modelTxt = samples::findFile(parser.get<string>("proto"));
    String modelBin = samples::findFile(parser.get<string>("model"));
    String imageFile = samples::findFile(parser.get<String>("image"));
    String dataset = parser.get<String>("dataset");
    int W_in = parser.get<int>("width");
    int H_in = parser.get<int>("height");
    float thresh = parser.get<float>("threshold");
    float scale = parser.get<float>("scale");
    if (parser.get<bool>("help") || modelTxt.empty() || modelBin.empty() || imageFile.empty())
    {
        cout << "A sample app to demonstrate human or hand pose detection with a pretrained OpenPose dnn." << endl;
        parser.printMessage();
        return 0;
    }
    int midx, npairs, nparts;
    if (!dataset.compare("COCO")) { midx = 0; npairs = 17; nparts = 18; }
    else if (!dataset.compare("MPI")) { midx = 1; npairs = 14; nparts = 16; }
    else if (!dataset.compare("HAND")) { midx = 2; npairs = 20; nparts = 22; }
    else
    {
        std::cerr << "Can't interpret dataset parameter: " << dataset << std::endl;
        exit(-1);
    }
    // read the network model
    Net net = readNet(modelBin, modelTxt);
    // and the image


    //VideoCapture cap("http://192.168.1.255:8080/video?x.mjpeg");  // capture from streaming
    VideoCapture cap(0); // capture from usb camera
    //VideoCapture cap("Videoclip3.avi");
    // Check if camera opened successfully
    if (!cap.isOpened()) {
        //printf("hello");
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    
    while (1) { // choose between 0-COCO, 1-MPI or 2-HAND
        Mat frame;
        // Capture frame-by-frame
        cap >> frame;
        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        // send it through the network
        Mat inputBlob = blobFromImage(frame, scale, Size(W_in, H_in), Scalar(0, 0, 0), false, false);
        net.setInput(inputBlob);
        Mat result = net.forward();
        // the result is an array of "heatmaps", the probability of a body part being in location x,y
        int H = result.size[2];
        int W = result.size[3];
        // find the position of the body parts
        vector<Point> points(22);

        int flag = 0;

        for (int n = 0; n < nparts; n++)
        {
            // Slice heatmap of corresponding body's part.
            Mat heatMap(H, W, CV_32F, result.ptr(0, n));
            // 1 maximum per heatmap
            Point p(-1, -1), pm;
            double conf;
            minMaxLoc(heatMap, 0, &conf, 0, &pm);
            if (conf > thresh)
                p = pm;
            points[n] = p;
        }
        // connect body parts and draw it !
        float SX = float(frame.cols) / W;
        float SY = float(frame.rows) / H;
        for (int n = 0; n < npairs; n++)
        {
            // lookup 2 connected body/hand parts
            Point2f a = points[POSE_PAIRS[midx][n][0]];
            Point2f b = points[POSE_PAIRS[midx][n][1]];
            // we did not find enough confidence before
            if (a.x <= 0 || a.y <= 0 || b.x <= 0 || b.y <= 0)
                continue;
            // scale to image size
            a.x *= SX; a.y *= SY;
            b.x *= SX; b.y *= SY;
            line(frame, a, b, Scalar(0, 200, 0), 2);
            circle(frame, a, 3, Scalar(0, 0, 200), -1);
            circle(frame, b, 3, Scalar(0, 0, 200), -1);

            ///////////////////////////////// my edit:
            
            // cuda implamentation??
            // net.setPreferableBackend(DNN_BACKEND_CUDA);
            // net.setPreferableTarget(DNN_TARGET_CUDA);
            

            //Angles define:
            double lknee_torso_deg, rknee_torso_deg, legs_deg, rknee_rhip_deg, lknee_lhip_deg;


            //Distance define:
            double dist;


            // Body parts defintion
            Point2f head = points[POSE_PAIRS[midx][0][0]];          //0
            Point2f neck = points[POSE_PAIRS[midx][0][1]];         //1
            Point2f rshoulder = points[POSE_PAIRS[midx][1][1]];     //2
            Point2f lshoulder = points[POSE_PAIRS[midx][4][1]];     //5
            Point2f relbow = points[POSE_PAIRS[midx][2][1]];        //3
            Point2f lelbow = points[POSE_PAIRS[midx][5][1]];        //6
            Point2f rpalm = points[POSE_PAIRS[midx][3][1]];         //4
            Point2f lpalm = points[POSE_PAIRS[midx][6][1]];         //7
            Point2f torso = points[POSE_PAIRS[midx][7][1]];         //14
            Point2f rhip = points[POSE_PAIRS[midx][8][1]];          //8
            Point2f lhip = points[POSE_PAIRS[midx][11][1]];         //11
            Point2f rknee = points[POSE_PAIRS[midx][9][1]];         //9
            Point2f lknee = points[POSE_PAIRS[midx][12][1]];        //12
            Point2f rankle = points[POSE_PAIRS[midx][10][1]];       //10
            Point2f lankle = points[POSE_PAIRS[midx][13][1]];       //13

            // scale to image size // float
            head.x *= SX; head.y *= SY;
            neck.x *= SX; neck.y *= SY;
            rshoulder.x *= SX; rshoulder.y *= SY;
            lshoulder.x *= SX; lshoulder.y *= SY;
            relbow.x *= SX; relbow.y *= SY;
            lelbow.x *= SX; lelbow.y *= SY;
            rpalm.x *= SX; rpalm.y *= SY;
            lpalm.x *= SX; lpalm.y *= SY;
            torso.x *= SX; torso.y *= SY;
            rhip.x *= SX; rhip.y *= SY;
            lhip.x *= SX; lhip.y *= SY;
            rknee.x *= SX; rknee.y *= SY;
            lknee.x *= SX; lknee.y *= SY;
            rankle.x *= SX; rankle.y *= SY;
            lankle.x *= SX; lankle.y *= SY;
            
            // Distance:
            
            dist = distance(rpalm.x, lpalm.x, rpalm.y, lpalm.y);
            //putText(frame, %lf , Point(10, 25), FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(200, 10, 10), 2);
      
            // Degree:

            rknee_torso_deg = degree(torso.x, rknee.x, torso.y, rknee.y);
            lknee_torso_deg = degree(torso.x, lknee.x, torso.y, lknee.y);
            legs_deg = rknee_torso_deg - lknee_torso_deg;

            rknee_rhip_deg = degree(rhip.x, rknee.x, rhip.y, rknee.y);
            lknee_lhip_deg = degree(lhip.x, lknee.x, lhip.y, lknee.y);
            

            if (legs_deg > 45) {
                if (lankle.y > rankle.y && lknee.y > rknee.y) {
                    if (lknee_lhip_deg > 75) {
                        // Right raised leg
                        flag = 1;
                        putText(frame, "right leg raised", Point(10, 25), FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(200, 10, 10), 2);
                    }
                                        
                    if (lpalm.y >= lankle.y || rpalm.y >= lankle.y) {
                        if (lknee_lhip_deg > 75) {
                           // Right leg raised to danger zone
                           flag = 2;
                           putText(frame, "Right leg raised to danger zone", Point(10, 25), FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(200, 10, 10), 2);
                        }
                        
                    }

                }
                if (rankle.y > lknee.y && rknee.y > lknee.y) {
                    if (rknee_rhip_deg > 75) {
                        // Left raised leg
                        flag = 3;
                        putText(frame, "left leg raised", Point(10, 25), FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(200, 10, 10), 2);
                    }
                    
                    if (rpalm.y >= rankle.y || lpalm.y >= rankle.y) {
                        if (rknee_rhip_deg > 75) {
                            // Left leg raised to danger zone
                            flag = 4;
                            putText(frame, "left leg raised to danger zone", Point(10, 25), FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(200, 10, 10), 2);
                        }

                    }
                }

            }

            // places of dots by (x,y) position only:

            if (head.y < rshoulder.y || head.y < lshoulder.y) {
                if (rshoulder.y < torso.y && lshoulder.y < torso.y) {
                    if (torso.y < rhip.y || torso.y < lhip.y) {
                        //putText(frame, "STANDING", Point(10, 25), FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(200, 10, 10), 2);

                    }
                }
                


            }           
            //printf("%lf\n", legs_deg);

            ///////////////////////////////// end myedit
        }
        // end for loop

        ///////////////////////////////// my edit:
        
     /*   switch(flag) {
            case 1:
                putText(frame, "right leg raised", Point(10, 25), FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(200, 10, 10), 2);
                break;
            case 2:
                putText(frame, "Right leg raised to danger zone", Point(10, 25), FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(200, 10, 10), 2);
                break;
            case 3:
                putText(frame, "left leg raised", Point(10, 25), FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(200, 10, 10), 2);
                break;
            case 4:
                putText(frame, "left leg raised to danger zone", Point(10, 25), FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(200, 10, 10), 2);
                break;       
        }*/
        //printf("%d", flag);
        

        ///////////////////////////////// end myedit
        imshow("OpenPose", frame);

        // Press  ESC on keyboard to exit
        char c = (char)waitKey(1);
        if (c == 27)
            break;
    }

    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    destroyAllWindows();
    return 0;
}

