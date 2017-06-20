#include "Utilities.h"
#include <sstream>
#include <ctime>
using namespace std;


Utilities::Utilities()
{
}


Utilities::~Utilities()
{
}

void Utilities::loadMNISTImages( const string path, Mat& images /*MVec& images*/ )
{
	CURRENT_FUNCTION;
	vector< vector<double> > pixels;
	_loadMNISTImages( path, pixels );

	vector<double> pixel = pixels[0];
	size_t cols = pixel.size();
	size_t rows = pixels.size();
	cout << "rows " << rows << " cols " << cols << endl;
	images.create( rows, cols, CV_32F );
	for (size_t i = 0; i < pixels.size(); i++)
	{
		vector<double> pixel = pixels[i];
		Mat mat;
		Mat( pixel ).copyTo( mat );
		mat.convertTo( mat, CV_64F, 1.0 / 255 );
		if (mat.cols == 1)
		{
			transpose( mat, mat );
		}
		mat.copyTo( images.row( i ) );
	}
}

void Utilities::_loadMNISTImages( const string path, vector< vector<double> > &pixels )
{
	ifstream fs( path.c_str(), ios::binary );

	//	auto reverseInt = [](int i) {
	//		unsigned char c1, c2, c3, c4;
	//		c1 = i & 255, c2 = ( i >> 8 ) & 255, c3 = ( i >> 16 ) & 255, c4 = ( i >> 24 ) & 255;
	//		return ( ( int ) c1 << 24 ) + ( ( int ) c2 << 16 ) + ( ( int ) c3 << 8 ) + c4;
	//	};


	typedef unsigned char uchar;

	if (fs.is_open()) {
		int magic_number = 0, number_of_images, n_rows = 0, n_cols = 0;

		fs.read( ( char * ) &magic_number, sizeof( magic_number ) );
		magic_number = reverseInt( magic_number );

		if (magic_number != 2051) throw runtime_error( "Invalid MNIST image file!" );

		fs.read( ( char * ) &number_of_images, sizeof( number_of_images ) ), number_of_images = reverseInt( number_of_images );
		fs.read( ( char * ) &n_rows, sizeof( n_rows ) ), n_rows = reverseInt( n_rows );
		fs.read( ( char * ) &n_cols, sizeof( n_cols ) ), n_cols = reverseInt( n_cols );

		size_t image_size = n_rows * n_cols;

		//number_of_images = 10000;
		// resize the double vector object to the same size as array, // step-a and b
		pixels.resize( number_of_images ); // step-a
		for (int i = 0; i < number_of_images; ++i)
		{
			pixels[i].resize( image_size ); // step-b
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					fs.read( ( char* ) &temp, sizeof( temp ) );
					int row = i, col = ( n_rows*r ) + c;
					pixels[i][( n_rows*r ) + c] = ( double ) temp;
				}
			}
			{
				Mat tmp(28, 28, CV_64F);

				int index = 0;
				for (size_t c = 0; c < 28; c++)
				{
					for (size_t j = 0; j < 28; j++)
					{
						tmp.at<double>( c, j ) = pixels[i][index++];
					}
				}
				//trainingData.push_back(tmp);	
			}
		}
	}
	else {
		throw runtime_error( "Cannot open file `" + path + "`!" );
	}
}

void Utilities::loadMNISTLabels( const string path, Mat& responses /*IVec& responses*/ )
{
	vector<int> lables;
	_loadMNISTLabels( path, lables );
	//trainingLables.assign(lables.begin(), lables.end());
	Mat( lables ).copyTo( responses );
	// 	responses.assign(lables.begin(), lables.end());
}

void Utilities::_loadMNISTLabels( const string path, vector<int> &lables )
{
	// 	auto reverseInt = []( int i ) {
	// 		unsigned char c1, c2, c3, c4;
	// 		c1 = i & 255, c2 = ( i >> 8 ) & 255, c3 = ( i >> 16 ) & 255, c4 = ( i >> 24 ) & 255;
	// 		return ( ( int ) c1 << 24 ) + ( ( int ) c2 << 16 ) + ( ( int ) c3 << 8 ) + c4;
	// 	};

	typedef unsigned char uchar;

	ifstream fs( path.c_str(), ios::binary );

	if (fs.is_open()) {
		int magic_number = 0, number_of_labels = 0;
		fs.read( ( char * ) &magic_number, sizeof( magic_number ) );
		magic_number = reverseInt( magic_number );

		if (magic_number != 2049) throw runtime_error( "Invalid MNIST label file!" );

		fs.read( ( char * ) &number_of_labels, sizeof( number_of_labels ) ), number_of_labels = reverseInt( number_of_labels );
		//number_of_labels = 10000;
		//uchar* _dataset = new uchar[number_of_labels];
		lables.resize( number_of_labels );
		for (int i = 0; i < number_of_labels; i++) {
			fs.read( ( char* ) &lables[i], 1 );
		}
	}
	else {
		throw runtime_error( "Unable to open file `" + path + "`!" );
	}
}


// debug helper functions
void Utilities::writeMatWithPath( const string filePath, const Mat mat )
{
	fstream fs( filePath.c_str() );

	if (fs.is_open())
	{
		for (int i = 0; i < 9; i++)
		{
			for (int j = 0; j < 9; j++)
			{
				for (int k = 0; k < 20; k++)
				{
					std::cout << mat.at<double>( i, j, k ) << "\t";
					fs << mat.at<double>( i, j, k ) << "\t";
					//cout << input[i];
				}
				fs << endl;
				cout << endl;
			}
			fs << endl;
			cout << endl;
		}
		fs.close();
	}
	// these two accesses are the same way to get the content of matrix
	double *input = ( double * ) ( mat.data );
	//double *input = wc.ptr<double>();
}


int Utilities::compareMat( const Mat src1, const Mat src2 )
{
	if (src1.empty() && src2.empty())
	{
		return true;
	}

	if (src1.cols != src2.cols || src1.rows != src2.rows || src1.dims != src2.dims)
	{
		return false;
	}

	Mat ret;
	compare( src1, src2, ret, CMP_NE );
	return countNonZero( ret );
}

void Utilities::conv2( const Mat &img, const Mat& kernel, ConvolutionType type, Mat& dest ) {
	Mat source = img;
	if (CONVOLUTION_FULL == type)
	{
		source = Mat();
		const int additionalRows = kernel.rows - 1, additionalCols = kernel.cols - 1;
		copyMakeBorder( img, source, ( additionalRows + 1 ) / 2, additionalRows / 2, ( additionalCols + 1 ) / 2, additionalCols / 2, BORDER_CONSTANT, Scalar( 0 ) );
	}

	Point anchor( kernel.cols - kernel.cols / 2 - 1, kernel.rows - kernel.rows / 2 - 1 );
	int borderMode = BORDER_CONSTANT;
	flip( kernel, kernel, -1 );
	filter2D( source, dest, img.depth(), kernel, Point( -1, -1 ), 0, borderMode );

	if (CONVOLUTION_VALID == type)
	{
		Mat tmp = dest.clone();
		int cs = 0, ce = 0, rs = 0, re = 0;
		if (kernel.cols % 2)
		{
			cs = ( kernel.cols - 1 ) / 2;
			ce = dest.cols - kernel.cols / 2;

			rs = ( kernel.rows - 1 ) / 2;
			re = dest.rows - kernel.rows / 2;
		}
		else
		{
			cs = ( kernel.cols ) / 2;
			ce = dest.cols - kernel.cols / 2 + 1;

			rs = ( kernel.rows ) / 2;
			re = dest.rows - kernel.rows / 2 + 1;
		}
		dest = dest.colRange( cs, ce ).rowRange( rs, re );
	}
}

void Utilities::shuffleData( const Mat _input, Mat& _output, Mat& _sf_inputs, Mat& _sf_outputs )
{
	RNG rng( 65536 );

	randShuffle( _output, 1, &rng );
	_sf_inputs.create( _input.rows, _input.cols, CV_64F );
	for (size_t i = 0; i < _input.rows; i++)
	{
		int rowIndex = _output.row( i ).at<int>( 0 );
		const double *src = _input.ptr<double>( rowIndex );
		_input.row( rowIndex ).copyTo( _sf_inputs.row( i ) );
	}
}

int Utilities::loadImagesWithPath( const string &prefix, vector<int> &labels, vector<Mat>&imageList, vector<string>&fileList )
{
	int ret = 1;
	assert( prefix.length() );
	int cnt = 0;

	for (int i = 0; i < CLASSES_NUM; i++)
	{
		if (_DEBUG)
			cout << "Number: [" << i << "]'s testing sample is loading" << endl;

		stringstream fileName( stringstream::in | stringstream::out );
		char buf[BUFFER_LENGTH] = { 0 };
		sprintf( buf, "Num%d\\Num%d.lst", i, i );
		fileName << buf;

		// Load images from separted folders
		string line;
		ifstream file;

		string fullpath = prefix + fileName.str();
		const char* path = fullpath.c_str();
		file.open( path );
		if (!file.is_open())
		{
			cerr << "Unable to open the list of images from " << fullpath << " filename." << endl;
			return RET_LOAD_SRC_NG;
		}

		bool end_of_parsing = false;
		while (!end_of_parsing)
		{
			getline( file, line );
			if (line == "") // no more file to read
			{
				end_of_parsing = true;
				break;
			}

			fileList.push_back( ToString( i ) + line );

			Mat img = imread( ( prefix + "Num" + ToString( i ) + "\\" + line ).c_str(), 0 ); // load the image
			if (img.empty()) // invalid image, just skip it.
				continue;
			Mat tmp;
			cv::GaussianBlur( img, tmp, cv::Size( 7, 7 ), 0 );
			//thresholding to get a binary image
			cv::threshold( tmp, tmp, 40, 255, THRESH_BINARY + CV_THRESH_OTSU );
			Mat out = normalizeImage( tmp );
			imageList.push_back( out );
			labels.push_back( i );
			if (++cnt >= TEST_SAMPLES)
			{
				break;
			}
		}
		cout << "Loaded " << cnt << " Samples successed!" << endl;
		cnt = 0;
	}

	return ret;
}

Mat Utilities::normalizeImage( Mat input ) {
	int m = max( input.rows, input.cols );

	Mat transformMat = Mat::eye( 2, 3, CV_64F );
	transformMat.at<double>( 0, 2 ) = ( float ) m / 2 - input.cols / 2;
	transformMat.at<double>( 1, 2 ) = ( float ) m / 2 - input.rows / 2;

	Mat warpImage( m, m, input.type() );
	warpAffine( input, warpImage, transformMat, warpImage.size(), CV_INTER_LINEAR, BORDER_CONSTANT, Scalar( 0 ) );
	Mat out;
	resize( warpImage, out, WIN_SIZE_SIZE );

	// for noraml matrix matching
	out.convertTo( out, CV_64F, 1.0 / 255 );

	// deskew image
	Mat dsk = deskew( out );
	dsk = dsk.reshape( 0, 1 );

	return dsk;
}

int Utilities::preprocessTrainingData( const vector<Mat> &imageSets, Mat &matData )
{

	int ret = 1;

	//--Convert data
	const int rows = ( int ) imageSets.size();
	const int cols = ( int ) max( imageSets[0].cols, imageSets[0].rows );
	cv::Mat tmp; //< used for transposition if needed
	matData = cv::Mat( rows, cols, CV_32F );
	vector< Mat >::const_iterator itr = imageSets.begin();
	vector< Mat >::const_iterator end = imageSets.end();
	for (int i = 0; itr != end; ++itr, ++i)
	{
		CV_Assert( itr->cols == 1 ||
				itr->rows == 1 );
		if (itr->cols == 1)
		{
			transpose( *( itr ), tmp );
			tmp.copyTo( matData.row( i ) );
		}
		else if (itr->rows == 1)
		{
			itr->copyTo( matData.row( i ) );
		}
	}

	return ret;
}

RET_STATUS Utilities::loadImageWithLablesInMixture( const string &prefix, const string &filename, vector<int> &labels, vector<Mat>&imageList, vector<string>&fileList )
{
	CURRENT_FUNCTION;

	RET_STATUS ret = RET_SUCCESS;

	assert( prefix.length() != 0 );
	assert( filename.length() != 0 );
	string line;
	ifstream file;

	string fullpath = prefix + filename;
	const char* path = fullpath.c_str();
	file.open( path );
	if (!file.is_open())
	{
		cerr << "Unable to open the list of images from " << filename << " filename." << endl;
		return RET_LOAD_SRC_NG;
	}

	bool end_of_parsing = false;

	while (!end_of_parsing)
	{
		getline( file, line );
		if (line == "") // no more file to read
		{
			end_of_parsing = true;
			break;
		}

		fileList.push_back( line );
		Mat img = imread( ( prefix + line ).c_str(), IMREAD_GRAYSCALE ); // load the image
		if (img.empty()) // invalid image, just skip it.
			continue;
		Mat tmp;
		//Applying gaussian blur to remove any noise
		cv::GaussianBlur( img, tmp, cv::Size( 7, 7 ), 0 );
		if (img.channels() > 1)
		{
			cvtColor( tmp, tmp, COLOR_BGR2GRAY );
		}

		//thresholding to get a binary image
		cv::threshold( tmp, tmp, 40, 255, CV_THRESH_BINARY + CV_THRESH_OTSU );

		// normalize image
		Mat out = normalizeImage( tmp );

		imageList.push_back( out );
		labels.push_back( ( int ) line.at( 0 ) - 48 );
	}
	return ret;
}


void Utilities::writeLocally()
{
	for( int idx = 0; idx < trainingData.size(); idx++)
	{
		char buf[32] = {0};
		stringstream ss(stringstream::in | stringstream::out);
		Mat img = trainingData[idx];
		int tmpIdx = trainingLables[idx];
		sprintf( buf, "TrainingData/InUsing/Num%d/%05d", tmpIdx, idx);
		ss << buf;
		string name = ss.str();
		//imshow(name + ".png", img);
		//waitKey(120);
		imwrite(name + ".png", img);
	}
}

void Utilities::shuffleData(MVec& inputs, IVec& responses)
{
	std::srand( unsigned( std::time(0) ) );
	for( size_t idx = 0; idx < responses.size(); idx++)
	{
		int r = idx + rand() % (responses.size() - idx );
		std::swap( responses[idx], responses[r] );
		std::swap( inputs[idx], inputs[r] );
	}
}

#if 1
RET_STATUS Utilities::loadImageAndMerge(const string &prefix, vector<int> &labels, vector<Mat>&imageList)
{
	CURRENT_FUNCTION;
	RET_STATUS ret = RET_INIT;
	assert( prefix.length() );
	int cnt = 0;

	for (int i = 0; i < CLASSES_NUM; i++)
	{
		if (_DEBUG)
			cout << "Number: [" << i << "]'s testing sample is loading" << endl;

		stringstream fileName( stringstream::in | stringstream::out );
		char buf[BUFFER_LENGTH] = { 0 };
		sprintf( buf, "Num%d/Num%d.lst", i, i );
		fileName << buf;

		// Load images from separted folders
		string line;
		ifstream file;

		string fullpath = prefix + fileName.str();
		const char* path = fullpath.c_str();
		file.open( path );
		if (!file.is_open())
		{
			cerr << "Unable to open the list of images from " << fullpath << " filename." << endl;
			return RET_LOAD_SRC_NG;
		}

		bool end_of_parsing = false;
		while (!end_of_parsing)
		{
			getline( file, line );
			if (line == "") // no more file to read
			{
				end_of_parsing = true;
				break;
			}

			Mat img = imread( ( prefix + "Num" + ToString( i ) + "/" + line ).c_str(), 0 ); // load the image
			if (img.empty()) // invalid image, just skip it.
			{
				cout << "Invalid image " << endl;
				continue;
			}
			Mat tmp;
			cv::GaussianBlur( img, tmp, cv::Size( 7, 7 ), 0 );
			//thresholding to get a binary image
			cv::threshold( tmp, tmp, 40, 255, THRESH_BINARY + CV_THRESH_OTSU );
			Mat out = normalizeImage( tmp );
			imageList.push_back( out );
			labels.push_back( i );
			if (++cnt >= TEST_SAMPLES)
			{
				break;
			}
		}
		cout << "Loaded " << cnt << " Samples successed!" << endl;
		cnt = 0;
	}

	return ret;
}
#endif

Mat Utilities::deskew( Mat& src )
{
	Moments m = moments( src );
	if (abs( m.m02 )< 1e-2)
	{
		return src.clone();
	}

	double skew = m.mu11 / m.mu02;

	Mat warp = ( Mat_<double>( 2, 3 ) << 1, skew, -0.5 * 60 * skew, 0, 1, 0 );
	Mat dst = Mat::zeros( src.rows, src.cols, src.type() );
	warpAffine( src, dst, warp, dst.size(), CV_INTER_LINEAR );
	return dst;
}

void Utilities::getConfusionMatrix( const int matrix[][CLASSES_NUM], const int* sampleCounter )
{
	CURRENT_FUNCTION;

	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	/*	Calculate True/False Positive Rate, Precision and Recall
	* TPR = TP/(TP+TN) = (number of true postives/ number of true positves + nuber of false negatives) 27/1002
	* FPR = FP/N = FP/(FP+TN)
	* Precision(i) = M(i,i)/sum_j(M(j,i))
	* Recall(i) = M(i,i)/sum_j(M(i,j))
	* F-Measure = 2 * Precision * Recall/Precision + Recall
	*/
	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// True positve samples
	int TP[CLASSES_NUM] = { 0 };

	// True positve rate
	float TPR[CLASSES_NUM] = { 0 };

	// False positve samples
	int FP[CLASSES_NUM] = { 0 };

	// False positve rate
	float FPR[CLASSES_NUM] = { 0 };

	// False negative samples
	int FN[CLASSES_NUM] = { 0 };

	// True negative samples
	int TN[CLASSES_NUM] = { 0 };

	// Precision
	float PR[CLASSES_NUM] = { 0 };

	// Recall
	float RE[CLASSES_NUM] = { 0 };

	// F-Measure
	float fMeasure[CLASSES_NUM] = { 0 };

	// Number of correctly classified samples
	int numCorrect = 0;

	// Number of incorrectly classified samples
	int numInCorrect = 0;

	// Number of all training samples
	int numSampels = 0;

	//const int* ptr = sampleCounter;
	for (int i = 0; i < CLASSES_NUM; i++)
	{
		numSampels += sampleCounter[i];
	}

	// Get total training samples and  the necessary variables here
	for (int i = 0; i < CLASSES_NUM; i++)
	{
		//numSampels += sampleCounter[i];

		// True positve samples
		TP[i] = matrix[i][i];
		//numCorrect += TP[i];

		for (int j = 0; j < CLASSES_NUM; j++)
		{
			// False negative samples
			FN[i] += matrix[i][j];
		}
		FN[i] -= matrix[i][i];

		// False positve samples
		for (int j = 0; j < CLASSES_NUM; j++)
		{
			// False positve samples
			FP[i] += matrix[j][i];
			//numInCorrect += FP[j];
		}
		FP[i] -= matrix[i][i];

		// True negative samples
		TN[i] = numSampels - TP[i] - FN[i] - FP[i];
	}


	// Calculate  true/False positive rate, precision and recall
	for (int i = 0; i < CLASSES_NUM; i++)
	{
		// True positive rate
		TPR[i] = static_cast< float >( TP[i] ) / static_cast< float >( ( TP[i] + FN[i] ) );

		// False positive rate
		FPR[i] = static_cast< float >( FP[i] ) / static_cast< float >( ( FP[i] + TN[i] ) );

		// Precision
		PR[i] = static_cast< float >( TP[i] ) / static_cast< float >( ( TP[i] + FP[i] ) );

		// Recall
		RE[i] = static_cast< float >( TP[i] ) / static_cast< float >( ( TP[i] + FN[i] ) );

		// F-Measure
		fMeasure[i] = 2 * PR[i] * RE[i] / ( PR[i] + RE[i] );

		// Correctly classified samples
		numCorrect += TP[i];

		// Incorrectly classified samples
		numInCorrect += FP[i];
	}

	cout << "[Correctly Classified Instances:\t" << numCorrect << "\t" << ( double ) numCorrect / ( double ) numSampels << "]" << endl;
	cout << "[Incorrectly Classified Instances:\t" << numInCorrect << "\t" << ( double ) numInCorrect / ( double ) numSampels << "]" << endl;
	cout << "[Total Number of Instances:\t" << numSampels << "]" << "\n" << endl;
	cout << "====Detailed Accuracy By Class====" << endl;
	cout << "\tClass\t\t" << "TP Rate\t\t" << "FP Rate\t\t" << "Precision\t\t" << "Recall\t\t" << "F-Measure\t\t" << "ROC Area\t\t" << endl;
	const char * format = "\t\t%d \t\t%.3f \t\t%.3f \t\t%.3f \t\t%.3f \t\t%.3f\n";// "\t\t%d \t %.3f \t\t%.3f \t\t%.3f \t\t%.3f \t\t%.3f \t\t%.3f\n";
	for (int i = 0; i < CLASSES_NUM; i++)
	{
		//cout <<"\t\t" << i << "\t\t" << TPR[i] << "\t\t" << FPR[i] << "\t\t" << PR[i] << "\t\t" << RE[i] << "\t\t" << fMeasure[i] << "\t\t" << endl;
		printf( format, i, TPR[i], FPR[i], PR[i], RE[i], fMeasure[i] );
	}
	//cout << "Weighted Avg." << "?" << "?" << "?" << "?" << "?" << "?" << endl;

	cout << "\n====Confusion Matrix====" << endl;
	cout << "\t\t0\t" << "\t1\t" << "\t2\t" << "\t3\t" << "\t4\t" << "\t5\t" << "\t6\t" << "\t7\t" << "\t8\t" << "\t9\t" << "<------Labeled as" << endl;
	for (int i = 0; i < CLASSES_NUM; i++)
	{
		cout << "| " << i << " |";
		for (int j = 0; j < CLASSES_NUM; j++)
		{
			cout << "\t" << matrix[i][j] << "\t";
		}
		cout << "\t" << TP[i] + FN[i] << endl;
	}
}

void Utilities::getPredictionTimeInfo( int samples )
{
	double seconds = timer.getSpendingSec();
	assert( samples > 0 );

	if (_DEBUG) {
		cout << "\n====Prediction Time====" << endl;
		cout << "[Testing until now has taken " << seconds << " seconds.]" << endl;
		cout << "[Average prediction time: " << seconds / samples << "]" << "\n" << endl;
	}
}
//////////////////////// cTimer implementation ////////////////////////
Timer::Timer() {
	begin = 0;
	end = 0;
}

void Timer::start()
{
	// reset timer
	reset();

	// counter start
	begin = clock();
}

void Timer::stop()
{
	end = clock();
}
void Timer::reset()
{
	begin = end = 0;
}

void Timer::getTakenTime()
{
	double seconds = double( end - begin ) / CLOCKS_PER_SEC;
	int days = int( seconds ) / 60 / 60 / 24;
	int hours = ( int( seconds ) / 60 / 60 ) % 24;
	int minutes = ( int( seconds ) / 60 ) % 60;
	int seconds_left = int( seconds ) % 60;

	if (_DEBUG) {
		cout << "Training until now has taken " << days << " days " << hours << " hours " << minutes << " minutes " << seconds_left << " seconds." << endl;
	}
}

double Timer::getSpendingSec()
{
	return double( end - begin ) / CLOCKS_PER_SEC;
}

string Timer::getCurrentTime()
{
	time_t     now = time( 0 );
	struct tm  tstruct;
	char       buf[80];
	tstruct = *localtime( &now );
	// Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
	// for more information about date/time format
	strftime( buf, sizeof( buf ), "%Y-%m-%d.%X", &tstruct );

	return string( buf );
}
