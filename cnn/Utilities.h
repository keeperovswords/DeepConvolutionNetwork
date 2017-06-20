#pragma once
#include "opencv2/highgui/highgui.hpp"
#include <istream>
#include <fstream>
#include <iostream>
#include <sstream>
#include <time.h>

#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

//double( *pfCost )( std::vector<double> theta, const Mat images, const Mat labels, int numClasses, int filteDim, int filterNum, int poolDim, std::vector<double>& grad );

#define _DEBUG 0
#define CURRENT_FUNCTION		\
if(_DEBUG)						\
	std::cout << __FUNCTION__ << endl


//#define DUMMY_MODE
#define CLASSES_NUM		10
#define BUFFER_LENGTH	64
#define TEST_SAMPLES	13000
#define WIN_SIZE_SIZE	Size(32, 32)
typedef enum {
	RET_INIT = 0,
	RET_SUCCESS,
	RET_FAILED,
	RET_MODEL_INIT_FAILED,
	RET_LOAD_SRC_NG
}RET_STATUS;

typedef vector<double>	DVec;
typedef vector<int>		IVec;
typedef vector<Mat>		MVec;
typedef vector<MVec>	Tensor;
enum enMeanMask {
	/* calculate the mean in column order*/
	MeanMask_Column,
	/* calculate the mean in row order*/
	MeanMask_Row
};

enum ConvolutionType {
	/* Return the full convolution, including border */
	CONVOLUTION_FULL,

	/* Return only the part that corresponds to the original image */
	CONVOLUTION_SAME,

	/* Return only the submatrix containing elements that were not influenced by the border */
	CONVOLUTION_VALID
};


template <typename T> string ToString(T val)
{
	stringstream ss;
	ss << val;
	return ss.str();
}
class Timer {
public:
	Timer();
	void start();
	void stop();
	void reset();
	void getTakenTime();
	double getSpendingSec();
	string getCurrentTime();

private:
	clock_t begin;
	clock_t end;
};
class Utilities
{
public:
	Utilities();
	~Utilities();

	void loadMNISTImages( const string path, Mat& image /*MVec& images*/ );
	void loadMNISTLabels( const string path, Mat& responses /*IVec& responses*/ );

	// compare the given mats
	int compareMat( const Mat src1, const Mat src2 );

	// debug helper functions
	void writeMatWithPath( const string filePath, const Mat mat );

	void conv2( const Mat &img, const Mat& kernel, ConvolutionType type, Mat& dest );

	// shuffle the trainint data randomly
	void shuffleData( const Mat _input, Mat& _output, Mat& _sf_inputs, Mat& _sf_outputs );

	template <typename T>
	static inline void readVectorOrMat(const FileNode &node, std::vector<T> &v)
	{
		if(node.type() == FileNode::MAP)
		{
			Mat m;
			node >> m;
			m.copyTo(v);
		}
		else if(node.type() == FileNode::SEQ)
			node >> v;
	}

	int loadImagesWithPath( const string &prefix, vector<int> &labels, vector<Mat>&imageList, vector<string>&fileList );
	Mat normalizeImage( Mat input );
	int preprocessTrainingData( const vector<Mat> &imageSets, Mat &matData );
	RET_STATUS loadImageWithLablesInMixture( const string &prefix, const string &filename, vector<int> &labels, vector<Mat>&imageList, vector<string>&fileList );
	void shuffleData( Mat& inputs, Mat& responses, Mat& _shuffle );
	void shuffleData( MVec& inputs, IVec& responses, MVec& _shuffle );
	void getConfusionMatrix( const int matrix[][CLASSES_NUM], const int* sampleCounter );
	void getPredictionTimeInfo( int samples );

	void tic() { timer.start(); }
	void tac() { timer.stop(); }
	Mat deskew( Mat& src );
	/** Calcuates mean with given order privilidge
	*/
	inline Mat mean( const Mat src, enMeanMask mask )
	{
		Mat dst( 1, src.cols, CV_64F );
		if (mask == MeanMask_Column)
		{
			for (size_t c = 0; c < src.cols; c++)
			{
				Scalar m = cv::mean( src.col( c ) );
				dst.at<double>( c ) = m[0];
			}
		}
		else
		{
			for (size_t r = 0; r < src.rows; r++)
			{
				Scalar m = cv::mean( src.row( r ) );
				dst.at<double>( r ) = m[0];
			}
		}
		return dst;
	}

	/** Calcautes mean of matrix, than once again of the mean
	*/
	inline double mean( const Mat src )
	{
		/* if just use mean(....) error occurs here */
		Mat dst = this->mean( src, MeanMask_Column );
		Scalar m = cv::mean( dst );
		return m[0];
	}

	/*rotates array A counterclockwise by k*90 degrees, where k is an integer.*/
	inline Mat rot90( Mat src, int roateFactor )
	{
		Mat dst( src.rows, src.cols, src.type() );
		flip( src, dst, roateFactor );

		return dst;
	}
	
	inline 	int reverseInt(int i){
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = ( i >> 8 ) & 255, c3 = ( i >> 16 ) & 255, c4 = ( i >> 24 ) & 255;
		return ( ( int ) c1 << 24 ) + ( ( int ) c2 << 16 ) + ( ( int ) c3 << 8 ) + c4;
	}
 
	void writeLocally();
	void shuffleData(MVec& inputs, IVec& responses);
	RET_STATUS loadImageAndMerge(const string &prefix, vector<int> &labels, vector<Mat>&imageList);
private:
	void _loadMNISTImages( const string path, vector< vector<double> >& pixels );
	void _loadMNISTLabels( const string path, vector<int> &lables );
	
	MVec trainingData;
	vector<int> trainingLables;
	Timer timer;
};

