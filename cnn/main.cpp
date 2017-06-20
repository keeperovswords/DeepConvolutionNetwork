// DeepConvolutionNeuralNetwork.cpp : Defines the entry point for the console application.
//
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/objdetect/objdetect.hpp>

#include <string>
#include <fstream>
#include "dcnn.h"
#include <vector>
#include "Utilities.h"

using namespace cv;
using namespace std;
using namespace DeepLearning;


cv::Mat wc, wh, bc, bh;
int main()
{
	// image dimension (width and height)
	int imageDim = 28;

	// number of classes(MNIST images fall into 10 classes)
	int numClasses = 10;

	// Filter size for conv layer
	int filterDim = 9;

	// number of filters for conv layer
	int numFilters = 20;

	// pooling dimension, (should divide imageDim - filterDim + 1)
	int poolDim = 2;


#define LoadModel	0	
#define MODEL_FILE	"cnn.xml"
	try
	{

		Utilities util;
		if(LoadModel)
		{
			Ptr<dcnnImpl> cnn = ml::StatModel::load<dcnnImpl>( MODEL_FILE );
			Mat testImages, testResponse;
			util.loadMNISTImages( "t10k-images.idx3-ubyte", testImages );
			util.loadMNISTLabels( "t10k-labels.idx1-ubyte", testResponse );
			Ptr<ml::TrainData> tdata = ml::TrainData::create( testImages, ml::ROW_SAMPLE, testResponse );
			cnn->modelTest( tdata );
			cnn.release();
		}
		else
		{
			Mat images, response;
			util.loadMNISTImages( "train-images.idx3-ubyte", images );
			util.loadMNISTLabels( "train-labels.idx1-ubyte", response );
			//util.writeLocally();

			Ptr<ml::TrainData>_tdata = ml::TrainData::create( images.rowRange(0, 10), ml::ROW_SAMPLE, response.rowRange( 0, 10 ) );
			Ptr<ml::TrainData>tdata = ml::TrainData::create( images, ml::ROW_SAMPLE, response );
			Ptr<dcnnImpl> cnn = dcnnImpl::create();
			if (cnn->gradientCheck( _tdata ))
			{
				//ModelParams model( 20, 9, 2, imageDim,/* response.rows,*/ 10 );
				ModelParams model( numFilters, filterDim, poolDim, imageDim,/* response.rows,*/ 10 );
				cnn->setModelParameter( model );
				LearnParams m;
				cnn->setLearningParameter( m );
				if (cnn->train( tdata ))
				{
					cout << "trained successed!" << '\a' << endl;

					Mat testImages, testResponse;
					util.loadMNISTImages( "t10k-images.idx3-ubyte", testImages );
					util.loadMNISTLabels( "t10k-labels.idx1-ubyte", testResponse );
					Ptr<ml::TrainData> tdata = ml::TrainData::create( testImages, ml::ROW_SAMPLE, testResponse );
					cnn->modelTest( tdata );

					cnn->save( MODEL_FILE );
				}
			}
			cnn.release();
		}
	}
	catch (Exception e)
	{
		cout << e.msg << endl;
	}

	return 0;
}

