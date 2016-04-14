/**
 * gentrain.cpp
 * 
 * @author Simon Teiwes (simalexteiwes [at] gmail.com)
 *
 * Description: Tool for cropping positives and negatives for HoG-SVM detector learning.
 * Usage: first create directories origin/, pos/ and neg/ in the directory of
 * gentrain. Configure the parameters in this file, fitting your desired HoG-Descriptors
 * parameters.
 * Put your raw training data pictures in the origin/ directory,
 * then start gentrain by typing "./gentrain".
 * Left-click and drag the red box around the desired positive training area,
 * right-click and drag to resize box.
 * Press "e" to save image from box to pos/ directory and randomly crop and save the
 * automatically generated negatives to neg/ directory.
 * To select next image from origin/ dir, press "n". For previous, press "p".
 * To quit, press q two times.
 *
 */

// TODO: fix crash when trying to export upscaled rects, that cross the image border
// TODO: fix missing image names in csv if images were never opened (switch positiveNames with imageNames)

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <ctype.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>

using namespace cv;
using namespace std;

//-----------------------------------------------
// CONFIGURABLE PARAMETERS
//-----------------------------------------------

// directory names
//static string originDir = "origin/";
static string originDir = "../origin/";
static string positiveDir = "../pos/";
static string negativeDir = "../neg/";
static string csvFileName = "../positions.csv";

// define how many negatives will be generated
static int numNegatives = 10;

// all images will be saved with this dimensions
// the dimensions also define the aspect ratio
// dimensions must fit hogdescriptor dimensions
static int windowWidth = 40;
static int windowHeight = 72;

// cellsize of hogdescriptor, for visualizing only
Size cellSize(8,8);

// generate csv-file containing positions and sizes of positives
static bool generateCSV = true;

// generate images? may be useful for csv labeling only
static bool generateImages = true;

// mirror positives to double number
static bool generateMirrored = true;

// limit downscaling of crop-window
static int minSizeDiv = 3;

//-----------------------------------------------


// for displaying purposes
static int fileIndex = 1;
static int numFiles = 0;

// actual image
Mat img;

// actual selection for positive
Rect selection;

// some helping variables
Rect origin;
Point mouseOrigin;
bool rButtonPressed = false;
bool lButtonPressed = false;

static void redraw();

static void onMouse( int event, int x, int y, int, void* )
{
	// scaling of selection window
	if (rButtonPressed && !lButtonPressed) {
		//cout << "setting dimensions.." << endl;
		if ((origin.height - ((float)windowHeight/(float)windowWidth)*(mouseOrigin.y - y)) >= windowHeight/minSizeDiv) {
			selection.height = origin.height - ((float)windowHeight/(float)windowWidth)*(mouseOrigin.y - y);
		} else {
			selection.height = windowHeight/minSizeDiv;
		}
		selection.width = selection.height*windowWidth/windowHeight;
		/*if ((origin.width - (mouseOrigin.y - y)) >= windowWidth/minSizeDiv) {
			selection.width = origin.width - (mouseOrigin.y - y);
		} else {
			selection.width = windowWidth/minSizeDiv;
		}*/
		// center window
		selection.x = origin.x - (selection.width - origin.width)/2;
		selection.y = origin.y - (selection.height - origin.height)/2;

        // redraw image
		redraw();
	}

	// move window center to mouse location while left mouse button is pressed
	if (lButtonPressed && !rButtonPressed) {
        // move roi relative to mouse origin
        int dx = x - mouseOrigin.x;
        int dy = y - mouseOrigin.y;
        
        // check for image boundaries
		if (((origin.x + dx) >= 0) && ((origin.x + dx + origin.width) < img.cols)) {
			selection.x = origin.x + dx;
		} else if (!((origin.x + dx + origin.width) < img.cols)) {
			selection.x = img.cols - selection.width;
		} else {
			selection.x = 0;
		}
		if (((origin.y + dy) >= 0) && ((origin.y + dy + origin.height) < img.rows)) {
			selection.y = origin.y + dy;
		} else if (!((dy + selection.height/2) < img.rows)) {
			selection.y = img.rows - selection.height;
		} else {
			selection.y = 0;
		}
        // redraw image
        redraw();
	}
    switch(event)
    {
    case EVENT_LBUTTONDOWN:
        // save mouse position
        mouseOrigin.x = x;
        mouseOrigin.y = y;
        origin.x = selection.x;
        origin.y = selection.y;
        origin.width = selection.width;
        origin.height = selection.height;
        lButtonPressed = true;
        break;
    case EVENT_LBUTTONUP:
        lButtonPressed = false;
        break;
    case EVENT_RBUTTONDOWN:
    	// anfaenglichen Punkt merken fuer Skalierug mit Mausbewegung
    	mouseOrigin.x = x;
    	mouseOrigin.y = y;
        origin.x = selection.x;
        origin.y = selection.y;
        origin.width = selection.width;
        origin.height = selection.height;
        rButtonPressed = true;
        break;
    case EVENT_RBUTTONUP:
        rButtonPressed = false;
    }
}

static void redraw() {
	Mat drawImg = img.clone();
    //Point offset(selection.width/4, selection.height/((float)(windowHeight/windowWidth)*4));
	rectangle(drawImg, selection.tl(), selection.br(), Scalar(15, 10, 255, 0.5), 2);
    //rectangle(drawImg, selection.tl()+offset, selection.br()-offset, Scalar(15, 10, 255, 0.5), 1);
    int cellCountX, cellCountY;
    cellCountX = windowWidth/cellSize.width;
    cellCountY = windowHeight/cellSize.height;
    for (int i = 0; i < cellCountX; i++) {
        for (int j = 0; j < cellCountY; j++) {
            Point topLeft, rightBottom;
            topLeft = Point(i*selection.width/cellCountX,j*selection.height/cellCountY) + selection.tl();
            rightBottom = Point((i+1)*selection.width/cellCountX,(j+1)*selection.height/cellCountY) + selection.tl();
            rectangle(drawImg, topLeft, rightBottom, Scalar(15, 10, 255, 0.5), 1);
        }
    }
	imshow("GenTrain", drawImg);
}

/*
 * crops roi, scales to defined dimension and saves to new image with given filename
 */
static void saveToFile(Mat &image, Rect &roi, string fileName) {
	Size size(windowWidth,windowHeight);
	Mat finalImg(image, roi);
	resize(finalImg,finalImg,size);
	if (!imwrite(fileName, finalImg)) {
        cout << "could not write image" << endl;
    }
}

/*
 * crops and saves possibly overlapping regions that don't overlap with the roi
 */
static void generateAndSaveNegatives(Mat &image, Rect &roi, string fileName) {
	int i = 0, j = 0, k = 0, randX = 0, randY = 0;
	vector<Rect> rects;
	// try only 3 times the wanted number of negatives
	while ((i < numNegatives) && (j < 3*numNegatives)) {
		Rect randRect, overlap;
		bool alreadyRect = false;
		randX = rand() % (image.cols - selection.width);
		randY = rand() % (image.rows - selection.height);
		randRect = Rect(randX,randY,selection.width,selection.height);
		overlap = randRect & roi;
		if (overlap.width == 0 && overlap.height == 0) {
			// check if rectangle is exactly the same as a previous one
			for (vector<Rect>::iterator it = rects.begin(); it != rects.end(); it++) {
				if (*it == randRect) {
					alreadyRect = true;
					break;
				}
			}
			if (!alreadyRect) {
				rects.push_back(randRect);
				i++;
			}
		}
        j++;
	}

	Mat drawImg = img.clone();

    // show roi
    rectangle(drawImg, roi.tl(), roi.br(), Scalar(15, 10, 250), 3);
    
	// save generated negatives to images
    // cut file extension
    string sub = fileName.substr(0, fileName.size()-4);
    // save extension
    string extension = fileName.substr(fileName.size()-4, fileName.size());
    //cout << "subfilename: " << sub << endl;
    //cout << "extension: " << extension << endl;
	for (vector<Rect>::iterator it = rects.begin(); it != rects.end(); it++) {
    	rectangle(drawImg, it->tl(), it->br(), Scalar(15, 240, 12), 3);
		stringstream fn;
        // build filename
		fn << sub << "-" << k << extension;
		saveToFile(image, *it, fn.str());
		k++;
	}
    imshow("GenTrain", drawImg);
	//cout << i << " negatives in " << j << " iterations." << endl;
}

static string toLowerCase(const string& in) {
    string t;
    for (string::const_iterator i = in.begin(); i != in.end(); ++i) {
        t += tolower(*i);
    }
    return t;
}


static bool compNames(string a, string b) {
    return a.compare(b) < 0;
}

/**
 * For unixoid systems only: Lists all files in a given directory and returns a vector of path+name in string format
 * @param dirName
 * @param fileNames found file names in specified directory
 * @param validExtensions containing the valid file extensions for collection in lower case
 */
static void getFilesInDirectory(const string& dirName, vector<string>& fileNames, const vector<string>& validExtensions) {
    struct dirent* ep;
    size_t extensionLocation;
    DIR* dp = opendir(dirName.c_str());
    if (dp != NULL) {
        while ((ep = readdir(dp))) {
            // Ignore (sub-)directories like . , .. , .svn, etc.
            if (ep->d_type & DT_DIR) {
                continue;
            }
            extensionLocation = string(ep->d_name).find_last_of("."); // Assume the last point marks beginning of extension like file.ext
            // Check if extension is matching the wanted ones
            string tempExt = toLowerCase(string(ep->d_name).substr(extensionLocation + 1));
            if (find(validExtensions.begin(), validExtensions.end(), tempExt) != validExtensions.end()) {
                fileNames.push_back((string)ep->d_name);
            } else {
                printf("Found file does not match required file type, skipping: '%s'\n", ep->d_name);
            }
        }
        (void) closedir(dp);
        
        // sort names alphabetically
        sort(fileNames.begin(), fileNames.end(), compNames);
    } else {
        printf("Error opening directory '%s'!\n", dirName.c_str());
    }
    return;
}

static void mirrorAndSave(string fileName, Mat& img, Rect sel) {
    Mat temp, flipped;
    Size size(windowWidth,windowHeight);
    temp = Mat(img, sel);
    resize(temp,temp,size);
    cv::flip(temp, flipped, 1);
    string name = fileName.substr(0, fileName.size()-4);
    string extension = fileName.substr(fileName.size()-4, fileName.size());
    stringstream s;
    s << name << "_mirror" << extension;
    if (!imwrite(s.str(), flipped)) {
        cout << "could not write image" << endl;
    }
}

static void clrLine() {
    cout << '\r';
    for (int i = 0; i < 120; i++) {
        cout << " ";
    }
    cout << '\r' << flush;
}

static void generateAndSaveCSV(vector<Rect> positions, vector<string> names, vector<bool> positiveExported) {
    // create file
    ofstream o(csvFileName.c_str());
    
    // delimiter
    char d = ';';
    
    // first line
    o << "name" << d << "x" << d << "y" << d << "width" << d << "height" << d << endl;
    
    for (int i = 0; i < numFiles; i++) {
        if (positiveExported[i]) {
            int x = positions[i].x + positions[i].width/2;
            int y = positions[i].y + positions[i].height/2;
            o << names[i] << d << x << d << y << d << positions[i].width << d << positions[i].height << d << endl;
        } else {
            o << names[i] << d << "-" << d << "-" << d << "-" << d << "-" << d << endl;
        }
    }
    o.close();
}

int main( int argc, const char** argv )
{
	vector<string> imageNames;
	vector<string> validExtensions;
	validExtensions.push_back("jpg");
	validExtensions.push_back("png");
	validExtensions.push_back("ppm");

	getFilesInDirectory(originDir, imageNames, validExtensions);

    namedWindow("GenTrain", 0);
    resizeWindow("GenTrain", 1440,800);
    
    numFiles = imageNames.size();
    vector<string>::iterator it = imageNames.begin();
    
    vector<Rect> positivePositions;
    vector<string> positiveNames;
    vector<bool> positiveExported;
    positivePositions.resize(numFiles);
    positiveNames.resize(numFiles);
    positiveExported.resize(numFiles, false);
    
    clrLine();
	cout << "loading image " << fileIndex << "/" << numFiles << flush;
    
    // append filename to path
	string inFile = originDir + *it;
	string outFile;
    img = imread(inFile, CV_LOAD_IMAGE_COLOR);
    selection = Rect(0,0,windowWidth,windowHeight);
    
    // enable mouse callback
    setMouseCallback("GenTrain", onMouse, 0);

	redraw();
    
	while (1) {
		char key = (char)waitKey(10);

		switch(key) {
		case 'e':
            
            // store name and rect
            positivePositions[it-imageNames.begin()] = selection;
            positiveNames[it-imageNames.begin()] = *it;
            positiveExported[it-imageNames.begin()] = true;

            if (generateImages) {
                //export image
                clrLine();
                cout << "exporting image from selection with dimension " << windowWidth << "x" << windowHeight << ". " << flush;
                outFile = positiveDir + *it;
                saveToFile(img,selection,outFile);
                if (generateMirrored) {
                    cout << "additionally exporting mirrored image." << flush;
                    mirrorAndSave(outFile,img,selection);
                }
                outFile = negativeDir + *it;
                generateAndSaveNegatives(img, selection, outFile);
            }
			break;
		case 'n':
			// load next image if it is available
			it++;
			if (it != imageNames.end()) {
                fileIndex++;
                clrLine();
				cout << "loading image " << fileIndex << "/" << numFiles << flush;
				inFile = originDir + *it;
                positiveNames[it-imageNames.begin()] = *it;
				img = imread(inFile, CV_LOAD_IMAGE_COLOR);
                redraw();
			} else {
                clrLine();
				cout << "already at file " << fileIndex << "/" << numFiles << " . Press q if you want to quit!" << flush;
				it--;
			}
			break;
		case 'p':
			// load previous image if it is available
			it--;
			if (it >= imageNames.begin()) {
                fileIndex--;
                clrLine();
				cout << "loading image " << fileIndex << "/" << numFiles << flush;
				inFile = originDir + *it;
				img = imread(inFile, CV_LOAD_IMAGE_COLOR);
        		redraw();
			} else {
                clrLine();
				cout << "already at file " << fileIndex << "/" << numFiles << " !" << flush;
				it++;
			}
			break;
		case 'q':
            clrLine();
            cout << "Press q again to really quit! Any other key to continue." << flush;
            // wait for input forever
            char key1 = (char)waitKey(0);
            if (key1 == 'q') {
                generateAndSaveCSV(positivePositions, positiveNames, positiveExported);
                clrLine();
                cout <<  "GenTrain 1.0 by Simon, thank me later, bye!" << endl;
                return 0;
            } else {
                clrLine();
                cout << "Ok, not quitting for now" << flush;
                break;
            }
            break;
		}
	}

    return 0;
}
