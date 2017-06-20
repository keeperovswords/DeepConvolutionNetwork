#include "dcnn.h"
#include <vector>

using namespace cv;
using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////
namespace DeepLearning {


	// Model parameters accessors
	//CV_IMPL_PROPERTY( int, FilterNum, mParams.nFilterNum )
	//CV_IMPL_PROPERTY( int, FilterDim, mParams.nFilterDim )
	//CV_IMPL_PROPERTY( int, PoolDim, mParams.nPoolDim )
	//CV_IMPL_PROPERTY( int, ImageDim, mParams.nImageDim )
	//CV_IMPL_PROPERTY( int, ClassesNum, mParams.nClassesNum )
	//CV_IMPL_PROPERTY( int, ConOutDim, mParams.nConOutDim )
	//CV_IMPL_PROPERTY( int, HiddenSize, mParams.nHiddenSize )

	// Learning parameters accessors
	//CV_IMPL_PROPERTY( int, Epochs, lParams.epochs )
	//CV_IMPL_PROPERTY( int, BatchSize, lParams.batchSize )
	//CV_IMPL_PROPERTY( float, Alpha, lParams.alpha )
	//CV_IMPL_PROPERTY( float, Momentum, lParams.momentum )
	//CV_IMPL_PROPERTY( float, Lambda, lParams.lambda )


	////////////////////////////////DCNNIMPL////////////////////////////////////
	void dcnnImpl::write_params( FileStorage& fs ) const
	{
		CURRENT_FUNCTION;
		// basic model parameters
		const char* activ_func_name = mParams.activeFun == IDENTITY ? "IDENTITY" :
			mParams.activeFun == SIGMOID_SYM ? "SIGMOID_SYM" :
			mParams.activeFun == GAUSSIAN ? "GAUSSIAN" : 0;

		if (activ_func_name)
			fs << "activation_function" << activ_func_name;
		else
			fs << "activation_function_id" << mParams.activeFun;

		//if (activ_func != IDENTITY)
		//{
		//	fs << "f_param1" << f_param1;
		//	fs << "f_param2" << f_param2;
		//}

		//fs << "min_val" << min_val << "max_val" << max_val << "min_val1" << min_val1 << "max_val1" << max_val1;

		fs << "model_params" << "{";
		if (mParams.trainMethod == BACKPROP)
		{
			fs << "train_method" << "BACKPROP";
			fs << "dw_scale" << mParams.bpDWScale;
			fs << "moment_scale" << mParams.bpMomentScale;
			fs << "filter_dim" << mParams.nFilterDim;
			fs << "filter_num" << mParams.nFilterNum;
			fs << "pool_dim" << mParams.nPoolDim;
			fs << "image_dim" << mParams.nImageDim;
			fs << "hidden_num" << mParams.nHiddenSize;
			fs << "classes_num" << mParams.nClassesNum;
			fs << "conout_num" << mParams.nConOutDim;
		}
		else if (mParams.trainMethod == RPROP)
		{
			fs << "train_method" << "RPROP";
			//fs << "dw0" << mParams.rpDW0;
			//fs << "dw_plus" << params.rpDWPlus;
			//fs << "dw_minus" << params.rpDWMinus;
			//fs << "dw_min" << params.rpDWMin;
			//fs << "dw_max" << params.rpDWMax;
		}
		else
			CV_Error( CV_StsError, "Unimplemented training method yet" );

		fs << "term_criteria" << "{";
		if (mParams.termCrit.type & TermCriteria::EPS)
			fs << "epsilon" << mParams.termCrit.epsilon;
		if (mParams.termCrit.type & TermCriteria::COUNT)
			fs << "iterations" << mParams.termCrit.maxCount  << "}";

		fs  << "}";

		// learning parameter
		fs << "training_params" << "{";
		fs << "alpha" << lParams.alpha;
		fs << "batchSize" << lParams.batchSize;
		fs << "epochs" << lParams.epochs;
		fs << "lambda" << lParams.lambda;
		fs << "momentum" << lParams.momentum;
		fs  << "}";
	}

	void dcnnImpl::read_params( const FileNode& fn )
	{
		CURRENT_FUNCTION;

		// basic model parameters			
		String activ_func_name = (String)fn["activation_function"];
		if( !activ_func_name.empty() )
		{
			mParams.activeFun = activ_func_name == "SIGMOID_SYM" ? SIGMOID_SYM :
				activ_func_name == "IDENTITY" ? IDENTITY :
				activ_func_name == "GAUSSIAN" ? GAUSSIAN : -1;
			CV_Assert( mParams.activeFun >= 0 );
		}
		else
			mParams.activeFun = (int)fn["activation_function_id"];

		FileNode mpn = fn["model_params"];
		if(! mpn.empty() )
		{
			String tmethod_name = (String)mpn["train_method"];

			if( tmethod_name == "BACKPROP" )
			{
				mParams.trainMethod = BACKPROP;
				mParams.bpDWScale = (double)mpn["dw_scale"];
				mParams.bpMomentScale = (double)mpn["moment_scale"];
				mParams.nFilterDim = (double) mpn["filter_dim"];
				mParams.nFilterNum = (double) mpn["filter_num"];
				mParams.nPoolDim = (double) mpn["pool_dim"];
				mParams.nImageDim = (double) mpn["image_dim"];
				mParams.nHiddenSize = (double) mpn["hidden_num"];
				mParams.nClassesNum = (double) mpn["classes_num"];
				mParams.nConOutDim = (double) mpn["conout_num"];
			}
			else if( tmethod_name == "RPROP" )
			{
				mParams.trainMethod = RPROP;
				//	mParams.rpDW0 = (double)tpn["dw0"];
				// 	mParams.rpDWPlus = (double)tpn["dw_plus"];
				// 	mParams.rpDWMinus = (double)tpn["dw_minus"];
				// 	mParams.rpDWMin = (double)tpn["dw_min"];
				// 	mParams.rpDWMax = (double)tpn["dw_max"];
			}
			else
				CV_Error(CV_StsParseError, "Unknown training method (should be BACKPROP or RPROP)");

			FileNode tcn = mpn["term_criteria"];
			if( !tcn.empty() )
			{
				FileNode tcn_e = tcn["epsilon"];
				FileNode tcn_i = tcn["iterations"];
				mParams.termCrit.type = 0;
				if( !tcn_e.empty() )
				{
					mParams.termCrit.type |= TermCriteria::EPS;
					mParams.termCrit.epsilon = (double)tcn_e;
				}
				if( !tcn_i.empty() )
				{
					mParams.termCrit.type |= TermCriteria::COUNT;
					mParams.termCrit.maxCount = (int)tcn_i;
				}
			}
		}

		FileNode tpn = fn["training_params"];
		if( !tpn.empty())
		{
			lParams.alpha = (double)mpn["alpha"];
			lParams.batchSize = (double)mpn["batchSize"];
			lParams.epochs = (double)mpn["epochs"];
			lParams.lambda = (double)mpn["lambda"];
			lParams.momentum = (double)mpn["momentum"];
		}
	}

	////////////////////////////////////////////INTERFACES////////////////////////////////////////////
	bool dcnnImpl::gradientCheck( Ptr<ml::TrainData>tdata )
	{
		CURRENT_FUNCTION;
		int filterNum = 2;
		int filterDim = 9;
		int poolDim = 5;
		int imageDim = 28;
		int numClasses = 10;
		int outDim = imageDim - filterDim + 1;
		outDim = outDim / poolDim;
		int hiddenSize = outDim * outDim * filterNum;
		mParams.nFilterNum = filterNum;
		mParams.nFilterDim = filterDim;
		mParams.nPoolDim = poolDim;
		mParams.nImageDim = imageDim;
		mParams.nClassesNum = numClasses;
		mParams.nHiddenSize = hiddenSize;
		mParams.nImageNum = 10;

		DVec _grad, _numGrad;
		DVec _theta;
		Mat inputs = tdata->getTrainSamples();
		Mat outputs = tdata->getTrainResponses();

		__initWeightParams( mParams, _theta );
		Mat _inputs = inputs.rowRange(0, 10);
		Mat _outputs = outputs.rowRange(0, 10);	
		double cost = dcnnFireUp( _theta, _inputs, _outputs, /*mParams.nClassesNum, mParams.nFilterDim, mParams.nFilterNum, mParams.nPoolDim,*/ _grad );

		dcnnComputeNumericalGradient( _theta, inputs, outputs, mParams.nClassesNum, mParams.nFilterDim, mParams.nFilterNum, mParams.nPoolDim, _numGrad );

		int sz[] = { 1, _grad.size() };
		Mat gradM( 2, sz, CV_64F, _grad.data() );
		Mat gradNumM( 2, sz, CV_64F, _numGrad.data() );
		double diff = cv::norm( gradNumM - gradM ) / cv::norm( gradNumM + gradM );
		double dd = 1e-9 - FLT_EPSILON; //2.1923058654563575e-10
		CV_Assert( diff < FLT_EPSILON/*, "Difference too large. Check your gradient computation again"*/ );

		return true;
	}

	void dcnnImpl::setModelParameter( ModelParams model )
	{

		//setFilterNum( model.nFilterNum );
		//setFilterDim( model.nFilterDim );
		//setPoolDim( model.nPoolDim );
		//setImageDim( model.nImageDim );
		//setClassesNum( model.nClassesNum );
		//int outDim = model.nImageDim - model.nFilterDim + 1;
		//setConOutDim( model.nImageDim - model.nFilterDim + 1 );

		//outDim = getConOutDim() / model.nPoolDim;

		mParams.nFilterNum = model.nFilterNum;
		mParams.nFilterDim = model.nFilterDim;
		mParams.nPoolDim = model.nPoolDim;
		mParams.nImageDim = model.nImageDim;
		mParams.nClassesNum = model.nClassesNum;
		mParams.nConOutDim = model.nImageDim - model.nFilterDim + 1;
		int nDim = mParams.nConOutDim / model.nPoolDim;
		int hiddenSize = nDim * nDim * mParams.nFilterNum;
		mParams.nHiddenSize = hiddenSize;
		mParams.nImageNum = model.nImageNum;


		__initWeightParams( mParams/*mParams.nImageDim, mParams.nFilterDim, mParams.nFilterNum, mParams.nPoolDim, mParams.nClassesNum*/, theta );
	}

	void dcnnImpl::setLearningParameter( LearnParams model )
	{
		//setEpochs( model.epochs );
		//setBatchSize( model.batchSize );
		//setAlpha( model.alpha );
		//setMomentum( model.momentum );
		//setLambda( model.lambda );
		lParams.epochs = model.epochs;
		lParams.batchSize = model.batchSize;
		lParams.alpha = model.alpha;
		lParams.momentum = model.momentum;
		lParams.lambda = model.lambda;
	}

	void dcnnImpl::modelTest( Ptr<ml::TrainData>tdata )
	{
		CURRENT_FUNCTION;

		Mat inputs = tdata->getTrainSamples();
		Mat responses = tdata->getTrainResponses();

		Mat prediction = Mat::zeros( responses.rows, 1, CV_32S );
		dcnnPredict( inputs, responses, prediction );

		vector<int> pred;
		vector<int> classes;
		pred.assign( ( int* ) prediction.datastart, ( int* ) prediction.dataend );
		classes.assign( ( int* ) responses.datastart, ( int* ) responses.dataend );
//		int cnt = 0;
//		cout << "should : "<< "\t is:" << endl;
//		for (size_t i = 0; i < pred.size(); i++)
//		{
//			if (pred[i] != classes[i])
//			{
//				cout << pred[i] << "\t" << classes[i] << endl;
//				cnt += 1;
//			}
//		}
		int diff = utils.compareMat( responses, prediction );
		int total = responses.rows;
		float acc = float( total - diff ) / total;

		cout << "Prediction accuracy: " << acc * 100 << "%" << " in " << total << "s Samples" << endl;

	}

	////////////////////////////////////////////INTERFACES////////////////////////////////////////////

	int dcnnImpl::__trainlWithBackprop( Mat inputs, Mat outputs, const Mat& _sw, TermCriteria termCrit )
	{

		CURRENT_FUNCTION;
		/** Instead of using normal optimization algorithm, we use here SGD method, cause it needs fewer memory requirement than i.g. BFGS 
		 *   and it'a also easy to debug.  
		 */

		int epoches = lParams.epochs;
		float alpha = lParams.alpha;
		float momentum = 0.5;
		int mom_step = 20;
		int batchSize = lParams.batchSize;
		DVec _theta;
		Mat velocity = Mat::zeros( theta.size(), 1, CV_64F );
		int iter = 0;

		const int sz[] = { theta.size(), 1 };

		for (int i = 0; i < epoches; i++)
		{
			for (int t = 0; t < inputs.rows /*&& iter < inputs.rows / batchSize*/; t+= batchSize)
			{
				if (iter == mom_step)
					momentum = 0.95;

				int step = ( t + batchSize ) <= inputs.rows ? t + batchSize : inputs.rows;
				Mat batchLabels = outputs.rowRange( t, step - 1 );
				Mat batchData = inputs.rowRange(t, step - 1 );

				DVec _grad;
				double cost = dcnnFireUp( theta, batchData, batchLabels, _grad );

				Mat grad( 2, sz, CV_64F, _grad.data() );
				velocity = momentum * velocity + alpha * grad;
				// theta -= velocity;
				Mat _MTheta;
				Mat( theta ).copyTo( _MTheta );
				_MTheta -= velocity;
				theta.clear();
				theta.assign( ( double* ) _MTheta.datastart, ( double* ) _MTheta.dataend );

				iter++;
				cout << "++++++++++++++++++Epoch " << i << " in " << iter << "-th Iteration, Cost: " << cost << "++++++++++++++++++"<< endl;
			}
			iter = 0;
			alpha /= 2;
		}

		optimalTheta.assign( theta.begin(), theta.end() );
		return 1;
	}

	void dcnnImpl::paramMatToVec( /*, const Mat param*/const Mat param1, const Mat param2, const Mat param3, const Mat param4, DVec& vec )
	{

		///! Matlab putting the parameters in columun way, in Opencv is row by row.
		// weight and bias in convolutional layer
		DVec::iterator it;
		int sy[] = { 2, 81 };
		Mat dd = Mat( 2, sy, CV_64F, param1.data + param1.step[0] * 0 );
		vec.assign( ( double* ) param1.datastart, ( double* ) param1.dataend );
		it = vec.end();
		vec.insert( it, ( double* ) param2.datastart, ( double* ) param2.dataend );
		it = vec.end();

		// weight and bias in  subpooling layer
		vec.insert( it, ( double* ) param3.datastart, ( double* ) param3.dataend );
		it = vec.end();
		vec.insert( it, ( double* ) param4.datastart, ( double* ) param4.dataend );
	}

	void dcnnImpl::vecToParamMat( const DVec vec, Mat& tmp_wc, Mat& tmp_bc, Mat& tmp_wh, Mat& tmp_bh )
	{

		DVec::const_iterator it = vec.begin();
		MatIterator_<double> mt_wc, mt_bc, mt_wh, mt_bh;
		const int sz_wc[] = { mParams.nFilterNum, mParams.nFilterDim, mParams.nFilterDim };
		const int sz_bc[] = { mParams.nFilterNum, 1 };
		const int sz_wh[] = { mParams.nClassesNum, mParams.nHiddenSize };
		const int sz_bh[] = { mParams.nClassesNum, 1 };

		tmp_wc.create( 3, sz_wc, CV_64F );
		tmp_bc.create( 2, sz_bc, CV_64F );
		tmp_wh.create( 2, sz_wh, CV_64F );
		tmp_bh.create( 2, sz_bh, CV_64F );
		mt_wc = tmp_wc.begin<double>();
		mt_bc = tmp_bc.begin<double>();
		mt_wh = tmp_wh.begin<double>();
		mt_bh = tmp_bh.begin<double>();

		int index, wc_index = mParams.nFilterDim * mParams.nFilterDim * mParams.nFilterNum;
		int bc_index = wc_index + mParams.nFilterNum;
		int wh_index = bc_index + mParams.nClassesNum * mParams.nHiddenSize;
		int bh_index = wh_index + mParams.nClassesNum;

//		ofstream os;
//		os.open("cnn_weight.txt");
//		ofstream pool;
//		pool.open("pooling_weight.txt");
		for (index = 0; it != vec.end(); ++it)
		{
			if (index < wc_index)
			{
//				os << ( double ) ( *it ) << ",";
				*mt_wc++ = ( double ) ( *it );
			}
			else if (index >= wc_index && index < bc_index)
			{	
//				os.close();
				*mt_bc++ = ( double ) ( *it );
			}
			else if (index >= bc_index && index < wh_index)
			{
//				pool << ( double ) ( *it ) << ",";
				*mt_wh++ = ( double ) ( *it );
			}
			else
			{
//				pool.close();
				*mt_bh++ = ( double ) ( *it );
			}

			index++;
		}

		CV_Assert( index == vec.size()/* "Convert Parameter Failed!" */);
	}

	void dcnnImpl::mat2DtoTensor( const Mat src, int nOutDim, int nImageNum, int nFilterNum, Tensor& ds )
	{
		ds.resize( nImageNum );
		for (size_t i = 0; i < src.cols; i++)
		{
			ds[i].resize( nFilterNum );

			Mat col = src.col( i );
			transpose( col, col );
			double* ptr = ( double* ) col.data;
			for (size_t f = 0; f < nFilterNum; f++)
			{
				Mat blob( nOutDim, nOutDim, CV_64F );
				for (size_t r = 0; r < nOutDim; r++)
				{
					memcpy( blob.data + blob.step[0] * r, ptr + r * nOutDim, sizeof( double ) * nOutDim );
				}
				transpose( blob, blob );
				ds[i][f].push_back( blob );
				ptr = ptr + nOutDim * nOutDim;
			}
		}
	}

	Mat dcnnImpl::tensorTo2DMat( const Tensor src, const int* size )
	{
		DVec dc;
		convertTensorToDvec( src, dc );
		return Mat( 4, size, CV_64F, dc.data() ).clone();
	}

	void dcnnImpl::dcnnUpsampling( const Tensor convolvedMaps, const Tensor ds, Mat& _deltaCon )
	{
		CURRENT_FUNCTION;
		// error in convolution layer
		Tensor dc; 
		int nConDim = getConvolutionDimension();
		int nImageNum = mParams.nImageNum;
		int nFilterNum = mParams.nFilterNum;
		int nPoolDim = mParams.nPoolDim;

		dc.resize( nImageNum );
		double blame = 1.0 / ( nPoolDim * nPoolDim );
		for (size_t i = 0; i < nImageNum; i++)
		{
			dc[i].resize( nFilterNum );
			for (size_t j = 0; j < nFilterNum; j++)
			{
				Mat blob = ds[i][j];
				Mat one = Mat::ones( nPoolDim, nPoolDim, CV_64F );
				Mat conMat = kron( blob, one );
				conMat *= blame;
				dc[i][j].push_back( conMat );
			}
		}


		int sz[] = { nImageNum, nFilterNum, nConDim, nConDim };

		//DVec delta_dc, delta_cm;
		//convertTensorToDvec( dc, delta_dc );
		//convertTensorToDvec( convolvedMaps, delta_cm );
		//Mat conMaps = Mat( 4, sz, CV_64F, delta_cm.data() );
		//Mat deltaCon = Mat( 4, sz, CV_64F, delta_dc.data() );
		Mat conMaps = tensorTo2DMat( convolvedMaps, sz );
		Mat deltaCon = tensorTo2DMat( dc, sz );

		Mat tmp = conMaps.mul( 1 - conMaps );
		deltaCon = deltaCon.mul( tmp );
		if (_DEBUG)
		{
			Mat pDC( nImageNum * nFilterNum, nConDim * nConDim, CV_64F, deltaCon.data + deltaCon.step[0] * 0 );
			Mat pCM( nImageNum * nFilterNum, nConDim * nConDim, CV_64F, conMaps.data + conMaps.step[0] * 0 );
			int dim = deltaCon.dims;
			Mat pDCM( nImageNum * nFilterNum, nConDim * nConDim, CV_64F, deltaCon.data + deltaCon.step[0] * 0 );
			Mat pTM( nImageNum * nFilterNum, nConDim * nConDim, CV_64F, tmp.data + tmp.step[0] * 0 );
		}

		_deltaCon = deltaCon.clone();
	}

	/** Calculates Kronecker tensor product
	 */
	Mat dcnnImpl::kron( const Mat src1, const Mat src2 )
	{
		CV_Assert( src1.channels() == 1 && src2.channels() == 1 );
		Mat ret;
		Mat1d s1d, s2d;
		src1.convertTo( s1d, CV_64F );
		src2.convertTo( s2d, CV_64F );

		Mat1d k1d( s1d.rows * s2d.rows, s1d.cols * s2d.cols, 0.0 );
		for (size_t r = 0; r < s1d.rows; r++)
		{
			for (size_t c = 0; c < s1d.cols; c++)
			{
				k1d( Range( r * s2d.rows, ( r + 1 ) * s2d.rows ), Range( c * s2d.rows, ( c + 1 ) * s2d.rows ) ) = s2d.mul( s1d( r, c ) );
			}
		}

		k1d.convertTo( ret, src1.type() );
		return ret;
	}

	void dcnnImpl::convertTensorToDvec(const Tensor in, DVec& out)
	{
		DVec::iterator it = out.begin();
		bool assign = true;
		int nImageNum = mParams.nImageNum;
		int nFilterNum = mParams.nFilterNum;
		for (size_t i = 0; i < nImageNum; i++)
		{
			for (size_t j = 0; j < nFilterNum; j++)
			{
				Mat m = in[i][j];
				if (assign)
				{
					out.assign( ( double* ) m.datastart, ( double* ) m.dataend );
					assign = false;
				}
				else
				{
					out.insert( it, ( double* ) m.datastart, ( double* ) m.dataend );
				}
				it = out.end();
			}
		}
#if 0
#define Dim0	3	// image num
#define Dim1	2	// filter num
#define Dim2	4	// feature dim
#define Dim3	4	// feature dim

		double data[] = { 0.0, 0, 1 };
		int sz[] = { Dim0, Dim1, Dim2, Dim3 };
		//Mat delta = cv::Mat( 4, sz, CV_64F, data1.data() );
		Tensor test;
		test.resize( Dim0 );

		for (size_t i = 0; i < Dim0; i++)
		{
			test[i].resize( Dim1 );
			for (size_t j = 0; j < Dim1; j++)
			{
				Mat one = Mat::ones( Dim3, Dim3, CV_64F );
				one = one * ( i + 1 ) * ( j + 1 );
				test[i][j].push_back( one );
			}
		}

		DVec errCon;
#endif
	}

	void dcnnImpl::dcnnBiasErrorBackToConvolutionlayer( const Mat src, MVec& vec, Mat& dst )
	{
		CURRENT_FUNCTION;

		int nConDim = getConvolutionDimension();

		//dst = Mat::zeros( mParams.nFilterNum,  mParams.nImageNum * nConDim * nConDim, CV_64F );
		Mat in( mParams.nImageNum * mParams.nFilterNum, nConDim * nConDim, CV_64F, src.data + src.step[0] * 0 );
		int row = 0;

		for (size_t r = 0; r < src.size[1]; r++)
		{
			Mat tmp = Mat::zeros( mParams.nImageNum, nConDim * nConDim, CV_64F );
			for (size_t i = 0; i < src.size[0]; i++)
			{
				in.row( r + src.size[1] * i ).copyTo( tmp.row( row++ ) );
			}
			row = 0;
			vec.push_back( tmp );
			tmp = tmp.reshape(0, 1);
			double s = sum( tmp )[0];
			dst.at<double>( r, 0 ) = s / mParams.nImageNum;
		}
	}


	int dcnnImpl::getConvolutionDimension() const
	{
		return mParams.nImageDim - mParams.nFilterDim + 1;
	}

	int dcnnImpl::getOutDimensionInPoolingLayer() const
	{
		return getConvolutionDimension() / mParams.nPoolDim;
	}

	template <class Tp_> Tp_ logsig( Tp_ a )
	{
		return ( Tp_ ) 1. / ( 1 + exp( -a ) );
	}

	void dcnnImpl::logsig( const Mat src, Mat& dst )
	{
		for (size_t c = 0; c < src.cols; c++)
		{
			for (size_t r = 0; r < src.rows; r++)
			{
				dst.at<double>( c, r ) = ( double ) 1. / ( 1 + exp( -src.at<double>( c, r ) ) );
			}
		}
	}

	void dcnnImpl::reshapeRawImage( const Mat src, const int nDimRow, const int nDimCol, Mat& dst )
	{
		int index = 0;
		for (size_t i = 0; i < nDimRow; i++)
		{
			for (size_t j = 0; j < nDimCol; j++)
			{
				dst.at<double>( i, j ) = src.at<float>( index++ );
			}
		}
	}


	////////////////////////////////DCNNIMPL////////////////////////////////////

	void dcnnImpl::setTrainMethod( int method, double param1, double param2 ) {}



	/** @brief Initializes the weights of neurons

	  @param imageDim the dimension of images
	  @param filterDim dimension of convolutional filter
	  @param numFilters number of convolutional filters
	  @param poolDim dimension of pooling area
	  @param numClasses number of classes to predict

	  The method initializes the parameter for a single layer convolutional neural network folled by a softmax layer.
	 */
	void dcnnImpl::__initWeightParams( ModelParams mp,/*int _imageDim, int _filterDim, int _filterNums, int _poolDim, int _numClasses,*/ DVec& _theta )
	{
		CURRENT_FUNCTION;

		CV_Assert( mp.nFilterDim  < mp.nImageDim/*, "filterDim must be less that imageDim"*/ );

//		cout << "mParams.trainMethod " << mParams.trainMethod << endl;
//		cout << "mParams.bpDWScale " << mParams.bpDWScale << endl;
//		cout << "mParams.bpMomentScale " << mParams.bpMomentScale << endl;
//		cout << "mParams.nFilterDim " << mParams.nFilterDim << endl;
//		cout << "mParams.nPoolDim " << mParams.nPoolDim << endl;
//		cout << "mParams.nImageDim " << mParams.nImageDim << endl;
//		cout << "mParams.nHiddenSize " << mParams.nHiddenSize << endl;
//		cout << "mParams.nClassesNum " << mParams.nClassesNum << endl;
//		cout << "mParams.nConOutDim " << mParams.nConOutDim << endl;

		//  dimension of convolved image
		int outDim = mp.nImageDim - mp.nFilterDim + 1;

		// assume outDim is multiple of poolDim
		CV_Assert( outDim% mp.nPoolDim == 0/*, "poolDim must divide imageDim - filterDim + 1"*/ );

		double mean = 0.0;
		double stddev = 1;
		const int sz_wc[] = { mp.nFilterNum, mp.nFilterDim, mp.nFilterDim };

		wc = cv::Mat( 3, sz_wc, CV_64F );
		randn( wc, Scalar( mean ), Scalar( stddev ) );
		wc = 0.1f * wc;

		const int sz_bc[] = { mp.nFilterNum, 1 };
		bc = Mat::zeros( 2, sz_bc, CV_64F );

		outDim = outDim / mp.nPoolDim;
		int hiddenSize = outDim * outDim * mp.nFilterNum;

		const int sz_wh[] = { mp.nClassesNum, hiddenSize };
		wh.create( 2, sz_wh, CV_64F );
		//Mat means = Mat::zeros( 1, 1, CV_64F );
		//Mat sigma = Mat::ones( 1, 1, CV_64F );
		randu( wh, Scalar( mean ), Scalar( stddev ) );

		//we'll choose weights uniformly from the interval [-r, r]
		float r = sqrt( 6 ) / sqrt( mp.nClassesNum + hiddenSize + 1 );
		wh *= 2 * r;
		wh -= r;

		const int sz_bh[] = { mp.nClassesNum, 1 };
		bh = Mat::zeros( 2, sz_bh, CV_64F );
		//#define DUMMY_MODE
#if defined DUMMY_MODE
		usingDummyData();
#endif

		lParams.lambda = 3e-4;

		// convert to vector 
		paramMatToVec( wc, bc, wh, bh, _theta );
		cout << "Parameter initial successed!" << endl;
	}

	void dcnnImpl::preprocessTrain( const Mat& inputs, const Mat& outputs,
			Mat& sample_weights, int flags )
	{
		if (wc.empty() || wh.empty() || bc.empty() || bh.empty())
			CV_Error( CV_StsError,
					"The network has not been created. Use method create or the appropriate constructor" );

		if (( inputs.type() != CV_32F && inputs.type() != CV_64F ))
			CV_Error( CV_StsBadArg,
					"input training data should be a floating-point matrix with "
					"the number of rows equal to the number of training samples and "
					"the number of columns equal to the size of 0-th (input) layer" );

		//if (( outputs.type() != CV_32F && outputs.type() != CV_64F ) )
		//	CV_Error( CV_StsBadArg,
		//		"output training data should be a floating-point matrix with "
		//		"the number of rows equal to the number of training samples and "
		//		"the number of columns equal to the size of last (output) layer" );

		//if (inputs.rows != outputs.rows)
		//	CV_Error( CV_StsUnmatchedSizes, "The numbers of input and output samples do not match" );

		Mat temp;
		double s = sum( sample_weights )[0];
		sample_weights.convertTo( temp, CV_64F, 1. / s );
		sample_weights = temp;
	}

	double dcnnImpl::dcnnFireUp( DVec theta, const Mat inputs, const Mat labels, /*int numClasses, int filterDim, int filterNum, int poolDim,*/ DVec& _grad )
	{
		CURRENT_FUNCTION;

		mParams.nImageNum = labels.rows;
		int nConDim = getConvolutionDimension();
		int nPoolOutDim = getOutDimensionInPoolingLayer();
		double cost = 0.f;
		// numImages  images.rows, convDim,convDim,numFilters
		const int sz_am[] = { mParams.nImageNum, mParams.nFilterNum, nConDim, nConDim };
		//Mat activationMaps = Mat::zeros( 4, sz_am, CV_64F );
		//MVec activationFeatures;
		Tensor convolvedMaps;

		vecToParamMat( theta, wc, bc, wh, bh );

		// forward step
		dcnnConvolution( inputs, wc, bc, convolvedMaps );

		// TODO:Warp data structure
		// images.rows, mParams.nFilterNum, nPoolOutDim, nPoolOutDim
		//const int sz_ap[] = { 10, mParams.nFilterNum, nPoolOutDim, nPoolOutDim };
		//Mat activationsPooled = Mat::zeros(4, sz_ap, CV_64F );
		Tensor activationsPooled;

		Mat sm = Mat::zeros( mParams.nHiddenSize, mParams.nImageNum, CV_64F );
		// subpooling actived features response from convolutional layer
		dcnnSubpooling( mParams.nPoolDim, convolvedMaps, activationsPooled, sm );

		// feed the pooled blob in to softmax layer
		// probs stores the probabilities of each image belongs to which class
		Mat probs = Mat::zeros( mParams.nClassesNum, mParams.nImageNum, CV_64F );
		dcnnSoftmax( wh, bh, sm, inputs, probs );

		/**STEP b: Calcuate the cost of objective
		 * the convolutional und sumpooling layers to the softmax layer.
		 */
		// build ground truth at first
		Mat groundTrue = Mat::zeros( mParams.nClassesNum, mParams.nImageNum, CV_64F );
		for (size_t i = 0; i < mParams.nImageNum; i++)
		{
			int response = labels.at<int>( i );
			//response = response ? response - 1 : 9; // TODO: remove it after testing
			groundTrue.at<double>( response, i ) = 1.0f;
		}

		// error in output layer
		cost = dcnnCost( groundTrue, wc, wh, probs );

		// backpropagate step
		dcnnBackPropagation( wc, wh, inputs, convolvedMaps, groundTrue, probs, sm, _grad );

		return cost;
	}

	/** @brief Executes the convolution filtering operation.

	  @param features actived mapping result in convolution operation
	  @param images in which convolution will be done
	  @param wc convolution filter
	  @param bc bias of convolution filter

	  The method filters the images in single layer convolutional layer.
	 */
	void dcnnImpl::dcnnConvolution( const Mat inputs, const Mat _wc, const Mat _bc, /*MVec*/Tensor &features )
	{
		CURRENT_FUNCTION;
		int nConDim = getConvolutionDimension();
		//int nImageNum = inputs.rows;

		features.resize( mParams.nImageNum );
		for (size_t i = 0; i < mParams.nImageNum; i++)
		{
			features[i].resize( mParams.nFilterNum );
			for (size_t j = 0; j < mParams.nFilterNum; j++)
			{
				Mat convolvedImage = Mat::zeros( nConDim, nConDim, CV_64F );
				Mat filter( mParams.nFilterDim, mParams.nFilterDim, CV_64F, wc.data + wc.step[0] * j );
				filter = utils.rot90( filter, -1 );
				DVec dv;
				paramMatToVec( wc, bc, wh, bh, dv );
				Mat img = inputs.row( i );
				Mat src( mParams.nImageDim, mParams.nImageDim, CV_64F );
				reshapeRawImage( img, mParams.nImageDim, mParams.nImageDim, src );
				if (src.type() == CV_32F)
					src.convertTo( src, CV_64F, 1.0 / 255 );

				// TODO: remove test
				//src = _dummyInputs[i];
				utils.conv2( src, filter, CONVOLUTION_VALID, convolvedImage );

				Mat tmpSum;
				reduce( convolvedImage, tmpSum, 0, CV_REDUCE_SUM, CV_64F );
				// after convolution than add bias
				convolvedImage = convolvedImage + bc.row( j );

				logsig( convolvedImage, convolvedImage );

				// TODO:Warp data structure
				// convolvedFeatures(:, :, filterNum, imageNum) = convolvedImage;
				//features.push_back( convolvedImage );
				features[i][j].push_back( convolvedImage );
			}
		}
	}

	void dcnnImpl::dcnnSubpooling( int nPoolDim, Tensor convolvedFeatures, Tensor& activationsPooled/*maybe useless, reserved row*/, Mat& sm )
	{
		CURRENT_FUNCTION;

		// 10, mParams.nFilterNum, nConDim, nConDim
		int nImageNum = convolvedFeatures.size(); // convolvedFeatures.size[0];
		int nFilterNum = mParams.nFilterNum;
		int convolvedDim = getConvolutionDimension();
		const int sz[] = { nImageNum, nFilterNum, convolvedDim / nPoolDim, convolvedDim / nPoolDim };
		Mat pooledFeatures = Mat::zeros( 4, sz, CV_64F );

		int poolLen = floor( convolvedDim / nPoolDim );
		activationsPooled.resize( mParams.nImageNum );
		for (size_t i = 0; i < mParams.nImageNum; i++)
		{
			activationsPooled[i].resize( nFilterNum );

			////////////////SOFTMAX/////////////////////
			int sz[] = { nFilterNum, poolLen * poolLen };
			Mat dst = Mat::zeros( 2, sz, CV_64F );
			double* ptr = ( double* ) dst.data;
			////////////////////////////////////////////


			for (size_t j = 0; j < nFilterNum; j++)
			{
				Mat feature = convolvedFeatures[i][j];
				Mat blob = Mat::zeros( poolLen, poolLen, CV_64F );
				for (size_t r = 1; r <= poolLen; r++)
				{
					for (size_t c = 1; c <= poolLen; c++)
					{
						int rs = nPoolDim * ( r - 1 );
						int re = nPoolDim * r;
						int cs = nPoolDim * ( c - 1 );
						int ce = nPoolDim * c;
						//cout << rs << re << cs << ce << endl;
						Range _rowRange( rs, re );
						Range _colRange( cs, ce );
						//blockFeatures = convolvedFeatures( i, j, rb : re, cb : ce );
						Mat blockFeatures = feature( _rowRange, _colRange );
						double mean = utils.mean( blockFeatures );

						blob.at<double>( r - 1, c - 1 ) = utils.mean( blockFeatures );
						//pooledFeatures( r, c, numFilters, nImageNum ) = mean( mean( convolvedFeatures( rs : re, cs : ce, nFilterNum, nImageNum ) ) );
					}
				}
				activationsPooled[i][j].push_back( blob );

				////////////////SOFTMAX/////////////////////
				transpose( blob, blob );
				Mat rp = blob.reshape( 0, 1 );
				memcpy( ptr + j * poolLen * poolLen, rp.data, sizeof( double ) * max( rp.rows, rp.cols ) );
				////////////////SOFTMAX/////////////////////
			}

			////////////////SOFTMAX/////////////////////
			Mat rp = dst.reshape( 0, 1 );
			transpose( rp, rp );
			rp.copyTo( sm.col( i ) );
			////////////////SOFTMAX/////////////////////
		}
	}

	void dcnnImpl::dcnnSoftmax( const Mat _wh, const Mat _bh, Mat activationsPooled, const Mat inputs, Mat& entropy )
	{
		CURRENT_FUNCTION;
		//Mat bias = Mat::zeros( 10, 10, CV_64F );
		Mat bias = repeat( _bh, 1, mParams.nImageNum );
		Mat dst = Mat::zeros( mParams.nClassesNum, mParams.nImageNum, CV_64F );
#if 0
		Mat activationsPooled = ( Mat_<double>( 32, 10 ) <<
				0.4872, 0.4887, 0.4857, 0.5000, 0.4807, 0.4926, 0.5043, 0.4953, 0.5005, 0.5018,
				0.4762, 0.4295, 0.4323, 0.5003, 0.4341, 0.4668, 0.4973, 0.4712, 0.4999, 0.4546,
				0.4819, 0.4007, 0.4346, 0.4865, 0.4702, 0.4020, 0.4999, 0.4307, 0.4999, 0.4729,
				0.4512, 0.4559, 0.4687, 0.4677, 0.4834, 0.4425, 0.5000, 0.4423, 0.5000, 0.4752,
				0.4196, 0.4133, 0.5105, 0.4960, 0.4384, 0.4261, 0.4240, 0.4609, 0.4526, 0.4497,
				0.3835, 0.4385, 0.4816, 0.4215, 0.4056, 0.3740, 0.4080, 0.3662, 0.4348, 0.4158,
				0.3960, 0.3798, 0.4681, 0.3851, 0.3443, 0.4231, 0.4337, 0.3977, 0.4416, 0.3690,
				0.4611, 0.4159, 0.4709, 0.4086, 0.4225, 0.3955, 0.4571, 0.4466, 0.4543, 0.4044,
				0.4942, 0.4308, 0.4754, 0.4151, 0.4188, 0.4512, 0.4586, 0.4130, 0.4919, 0.4734,
				0.3300, 0.3669, 0.4182, 0.4301, 0.3930, 0.3178, 0.3748, 0.3047, 0.4231, 0.3936,
				0.3747, 0.4145, 0.3917, 0.4473, 0.3912, 0.4149, 0.3305, 0.3242, 0.4081, 0.4149,
				0.4346, 0.4730, 0.4538, 0.4660, 0.3950, 0.4122, 0.3717, 0.4536, 0.4271, 0.4157,
				0.5056, 0.4582, 0.4555, 0.4995, 0.5199, 0.5137, 0.5091, 0.4663, 0.5036, 0.4544,
				0.4498, 0.3430, 0.4607, 0.4477, 0.4642, 0.4340, 0.5250, 0.4033, 0.5120, 0.4587,
				0.5149, 0.4445, 0.4420, 0.4830, 0.4889, 0.4203, 0.5320, 0.4399, 0.5200, 0.4648,
				0.4391, 0.4531, 0.4471, 0.4996, 0.5238, 0.4269, 0.5106, 0.4522, 0.5166, 0.4712,
				0.5828, 0.5468, 0.4872, 0.5000, 0.5474, 0.5569, 0.5007, 0.5390, 0.5005, 0.5081,
				0.3801, 0.5414, 0.5605, 0.5076, 0.5598, 0.4719, 0.4889, 0.4940, 0.4989, 0.5849,
				0.5149, 0.4564, 0.5292, 0.5694, 0.3691, 0.5653, 0.4999, 0.4654, 0.4999, 0.4283,
				0.5735, 0.4816, 0.4840, 0.4934, 0.4667, 0.3910, 0.5000, 0.5782, 0.5000, 0.4698,
				0.5827, 0.5889, 0.4909, 0.5386, 0.5837, 0.5737, 0.5627, 0.5402, 0.5691, 0.5412,
				0.4625, 0.3259, 0.5264, 0.5726, 0.4797, 0.4769, 0.4695, 0.4978, 0.4739, 0.5173,
				0.4875, 0.5999, 0.4880, 0.4170, 0.5498, 0.4410, 0.4556, 0.4336, 0.4607, 0.5421,
				0.4018, 0.5520, 0.5108, 0.4314, 0.4679, 0.4804, 0.3736, 0.4939, 0.4322, 0.4010,
				0.4811, 0.5384, 0.5591, 0.5557, 0.5618, 0.5343, 0.5209, 0.6278, 0.4871, 0.5454,
				0.5431, 0.5281, 0.4772, 0.3647, 0.4364, 0.5261, 0.5896, 0.3965, 0.5556, 0.4814,
				0.5706, 0.4668, 0.4929, 0.5234, 0.5268, 0.4810, 0.5993, 0.5278, 0.5633, 0.4750,
				0.4479, 0.4374, 0.4617, 0.5718, 0.5710, 0.4906, 0.5762, 0.4331, 0.5530, 0.5577,
				0.5040, 0.5278, 0.4966, 0.4820, 0.4573, 0.4845, 0.4895, 0.5107, 0.4954, 0.4778,
				0.5147, 0.6138, 0.4965, 0.5813, 0.5636, 0.5511, 0.4891, 0.5717, 0.4924, 0.4880,
				0.4963, 0.4732, 0.5712, 0.5288, 0.5374, 0.6290, 0.4840, 0.5676, 0.4918, 0.5723,
				0.5970, 0.5607, 0.5572, 0.5010, 0.4911, 0.4927, 0.5400, 0.5869, 0.5083, 0.5275 );

		Mat wh = ( Mat_<double>( 10, 32 ) <<
				-0.0222, 0.1549, 0.2618, -0.3532, -0.2294, -0.2698, -0.2122, 0.0628, 0.0485, 0.2931, -0.3367, -0.2578, -0.1635, 0.2818, -0.1742, -0.0453, 0.1549, -0.0753, 0.1845, -0.0141, 0.2238, 0.2273, 0.1803, -0.0920, -0.0895, -0.2392, -0.2810, 0.0593, -0.2641, -0.3645, -0.2471, -0.0269,
				-0.1427, -0.0789, 0.3031, 0.1316, 0.1939, -0.3344, -0.0055, 0.1957, 0.3062, 0.2125, 0.2522, -0.2224, -0.2403, 0.0247, -0.1490, 0.1301, -0.0544, -0.0748, -0.1178, -0.1258, 0.0071, -0.0224, 0.1794, -0.3062, 0.1025, -0.2439, 0.1783, -0.1392, 0.0497, 0.3608, 0.3214, 0.0204,
				-0.2753, -0.3654, -0.0117, 0.1349, 0.1481, 0.1130, 0.3340, 0.2337, -0.2794, -0.0301, -0.0461, -0.2509, 0.0044, -0.0838, -0.1611, -0.3395, -0.0035, 0.0211, 0.2008, 0.0392, -0.0396, -0.3051, 0.0744, 0.3474, 0.1100, 0.2953, -0.3546, 0.2732, 0.0055, -0.3659, 0.2589, -0.2693,
				0.0267, -0.3505, 0.2029, -0.0315, 0.2107, -0.1629, -0.1014, -0.2207, 0.0511, 0.2279, 0.1792, 0.1047, 0.3526, 0.0849, -0.1034, 0.2727, 0.2592, -0.2227, -0.3689, 0.3476, -0.0532, -0.0942, -0.2442, 0.0593, 0.1802, -0.0818, -0.0714, 0.0161, -0.3127, 0.2782, -0.1227, 0.0824,
				-0.1692, -0.2171, 0.0328, 0.2466, 0.2149, -0.1485, -0.1390, 0.0764, -0.3572, -0.0824, 0.1648, -0.2648, -0.2382, 0.2745, 0.3023, -0.1255, -0.2389, 0.0615, -0.2445, 0.1405, 0.2187, 0.0516, -0.1114, -0.1130, 0.2239, 0.3274, -0.1391, -0.1911, -0.3640, -0.1945, -0.1928, 0.0345,
				-0.0977, 0.0547, -0.3094, 0.3012, -0.0362, 0.1726, -0.1539, 0.0520, 0.2340, -0.2248, 0.2198, -0.3067, -0.3094, -0.1020, -0.2419, 0.0226, 0.2376, 0.1361, -0.2781, -0.0535, 0.0346, -0.2009, 0.0289, -0.1269, -0.0265, -0.1290, 0.0478, 0.1756, -0.0791, -0.0973, -0.2758, 0.2247,
				-0.3286, -0.0347, 0.3327, 0.2882, -0.1696, 0.3177, 0.2274, -0.2607, -0.1357, -0.3356, -0.0247, 0.1391, -0.3019, 0.2758, 0.2789, -0.0569, 0.2594, 0.1332, 0.2029, -0.2972, 0.2695, 0.0942, -0.3356, 0.2860, 0.3664, -0.1037, 0.0039, -0.2294, 0.0785, 0.0784, 0.2670, 0.0100,
				-0.1956, 0.1694, 0.2536, 0.1299, 0.3154, -0.0864, -0.1638, 0.1911, -0.1060, -0.0249, -0.1276, 0.0068, 0.0843, -0.3475, -0.0412, -0.3621, 0.0505, -0.0642, -0.0752, -0.3650, 0.3729, 0.1463, 0.1467, 0.0599, 0.2941, -0.2924, -0.2634, 0.3585, -0.2795, 0.0596, 0.0099, -0.1019,
				0.1987, 0.1168, 0.1639, -0.0897, -0.3203, 0.1706, -0.0383, -0.2178, -0.2777, -0.3148, 0.1692, 0.3066, -0.1447, -0.2976, 0.2408, -0.1105, 0.3559, -0.0651, -0.0123, 0.0045, 0.2748, 0.1918, 0.2094, -0.0074, -0.1104, 0.3211, 0.1377, 0.0012, -0.3341, -0.1013, -0.1038, -0.1669,
				0.2799, 0.0554, 0.0220, 0.3090, -0.0819, -0.0968, 0.1711, 0.0796, 0.3710, -0.1954, 0.2899, 0.3603, -0.3493, 0.2013, 0.3588, -0.1098, 0.1547, 0.1271, -0.1236, -0.0801, 0.3465, -0.0425, -0.0902, 0.0188, 0.1275, 0.1317, 0.0165, 0.3453, 0.2610, 0.3200, 0.0625, 0.0916 );

#endif

		gemm( _wh, activationsPooled, 1, noArray(), 1, dst );

		dst += bias;
		Mat maxVal;
		reduce( dst, maxVal, 0, CV_REDUCE_MAX, CV_64F );
		maxVal = repeat( maxVal, mParams.nClassesNum, 1 );
		dst = dst - maxVal;
		exp( dst, dst );
		Mat s;
		reduce( dst, s, 0, CV_REDUCE_SUM, CV_64F );
		Mat rp = repeat( s, mParams.nClassesNum, 1 );
		divide( dst, rp, entropy );
	}

	/** Cost consists of two partys: the error in softmax layer and the penalty cost
	 */
	double dcnnImpl::dcnnCost( const Mat _groundTruth, const Mat _wc, const Mat _wh, Mat _probs )
	{

		double ret = 0.0f;
		Mat logE, gt, pt;
#if 1			
		transpose( _groundTruth, gt );
		Mat gr = gt.reshape( 0, 1 );

		cv::log( _probs, logE );
		transpose( logE, pt );
		Mat lr = pt.reshape( 0, 1 );

		Mat mul = gr.mul( lr );
		double s = sum( mul )[0];
		ret = -1.0 / mParams.nImageNum * s;


		// penalty cost
		Mat pwc, pwh;
		pow( wc, 2, pwc );
		pow( wh, 2, pwh );

		s = sum( pwc )[0];
		double sc = sum( pwh )[0];
		ret += ( lParams.lambda / 2 )*( s + sc );

#else
		transpose( _groundTruth, gt );
		Mat gr = gt.reshape( 0, 1 );

		cv::log( _probs, logE );
		transpose( logE, pt );
		Mat lr = pt.reshape( 0, 1 );

		Mat mul = gr.mul( lr );
		double s = sum( mul )[0];
		ret = -1.0 / mParams.nImageNum * s;


		// penalty cost
		Mat pwc, pwh;
		pow( _wc, 2, pwc );
		pow( _wh, 2, pwh );

		s = sum( pwc )[0];
		double sc = sum( pwh )[0];
		ret += ( lParams.lambda / 2 )*( s + sc );
#endif
		return ret;
	}

	/** Backpropagate the error from softmax layer through convolutional and subpooling layer
	 */

	void dcnnImpl::dcnnBackPropagation( const Mat _wc, const Mat _wh, const Mat inputs, const Tensor convolvedMaps, const Mat _groundTruth, const Mat _probs, const Mat _pooledBlobs, DVec& _grad )
	{
		CURRENT_FUNCTION;

		// error in output layer
		Mat deltaCross = _groundTruth - _probs;
		deltaCross = ( -1 )  * deltaCross;
		// error in subpooling layer
		Mat blobTrans, gradWH, gradBH, sumBH, tmpWH = _wh;
		transpose( _pooledBlobs, blobTrans );
		gemm( deltaCross, blobTrans, 1, noArray(), 1, gradWH );

		// gradient of fully connected layer
		gradWH = ( 1.0 / mParams.nImageNum ) * gradWH + lParams.lambda * tmpWH;

		reduce( deltaCross, sumBH, 1, CV_REDUCE_SUM, CV_64F );
		gradBH = sumBH * ( 1.0 / mParams.nImageNum );

		// error in pooling layer
		Mat whTrans, deltaPool, deltaCon, gradWC, gradBC;
		transpose( wh, whTrans );// TODO:remove Dummy data
		gemm( whTrans, deltaCross, 1, noArray(), 1, deltaPool );

		Tensor ds; //error in pooling layer
		mat2DtoTensor( deltaPool, getOutDimensionInPoolingLayer(), mParams.nImageNum, mParams.nFilterNum, ds );

		// now the error in pooling will be upsampled in convolution layer
		dcnnUpsampling( convolvedMaps, ds, deltaCon );
		if (_DEBUG)
		{
			Mat pDCM( 20, 400, CV_64F, deltaCon.data + deltaCon.step[0] * 0 );
		}

		// gradient of convolution layer 
		dcnnWeightErrorBackToConvolutionlayer( _wc, deltaCon, inputs, gradWC, gradBC );

		paramMatToVec( gradWC, gradBC, gradWH, gradBH, _grad );
	}

	void dcnnImpl::dcnnComputeNumericalGradient( DVec theta, const Mat images, const Mat labels, int numClasses, int filterDim, int filterNum, int poolDim, /*DVec grad,*/ DVec& numGrad )
	{
		CURRENT_FUNCTION;
		double epsilon = 1e-4;

		DVec _theta = theta;
		DVec _grad;
		numGrad.resize( theta.size() );

		for (size_t i = 0; i < theta.size(); i++)
		{
			double oldT = _theta[i];
			_theta[i] = oldT + epsilon;
			double pos = dcnnFireUp( _theta, images, labels, /*numClasses, filterDim, filterNum, poolDim,*/ _grad );
			_theta[i] = oldT - epsilon;
			double neg = dcnnFireUp( _theta, images, labels, /*numClasses, filterDim, filterNum, poolDim,*/ _grad );
			numGrad[i] = ( pos - neg ) / ( 2 * epsilon );
			_theta[i] = oldT;
			cout << "++++++++++++++++++The " << i << "-th gradient: " << numGrad[i] << "++++++++++++++++++" << endl;
		}
	}

	void dcnnImpl::dcnnWeightErrorBackToConvolutionlayer( Mat _wc, const Mat deltaCon, const Mat images, Mat& gradWC, Mat& gradBC )
	{
		CURRENT_FUNCTION;

		int nImageNum = mParams.nImageNum;
		int nFilterNum = mParams.nFilterNum;
		int nFilterDim = mParams.nFilterDim;
		int nConDim = getConvolutionDimension();

		int sz_wc[] = { nFilterNum, nFilterDim, nFilterDim };
		gradWC.create( 3, sz_wc, CV_64F );
		gradBC.create( nFilterNum, 1, CV_64F );
		MVec RFields;

		// calculate bias error in convolution layer
		dcnnBiasErrorBackToConvolutionlayer( deltaCon, RFields, gradBC );
		int sz_d[] = { 2, 81 };
		int sz_c[] = { 20, 400 };
		Mat dd = Mat( 2, sz_d, CV_64F, _wc.data + _wc.step[0] * 0 );
		Mat cc = Mat( 2, sz_c, CV_64F, deltaCon.data + deltaCon.step[0] * 0 );
		for (size_t i = 0; i < nFilterNum; i++)
		{
			// gradient of each filter in all images
			Mat gradwc = Mat::zeros( nFilterDim, nFilterDim, CV_64F );
			Mat Rf = RFields[i];
			double* ptrRf = ( double* ) Rf.data;
			for (size_t j = 0; j < nImageNum; j++)
			{
				Mat img = images.row( j );
				Mat src( mParams.nImageDim, mParams.nImageDim, CV_64F );
				reshapeRawImage( img, mParams.nImageDim, mParams.nImageDim, src );
				//src.convertTo( src, CV_64F, 1.0 / 255 );
				// TODO: remove test
				//src = _dummyInputs[j];

				Mat conM = Mat::zeros( nConDim, nConDim, CV_64F );
				memcpy( conM.data, ptrRf, sizeof( double ) * nConDim * nConDim );
				Mat TconM = utils.rot90( conM, -1 );
				ptrRf += nConDim * nConDim;

				Mat convolvedImage = Mat::zeros( nFilterDim, nFilterDim, CV_64F );
				utils.conv2( src, TconM, CONVOLUTION_VALID, convolvedImage );
				gradwc = gradwc + convolvedImage;
			}

			Mat filter( mParams.nFilterDim, mParams.nFilterDim, CV_64F, _wc.data + _wc.step[0] * i );
			Mat tmp = 1.0 / nImageNum * gradwc + lParams.lambda * filter;
			double s = sum( tmp )[0];
			size_t t = ( tmp.dataend - tmp.datastart );
			memcpy( ( gradWC.data + gradWC.step[0] * i ), ( double* ) tmp.datastart, ( tmp.dataend - tmp.datastart ) );
		}
		// check it ok
		if (_DEBUG)
		{
			Mat pm( nFilterNum, 81, CV_64F, gradWC.data + gradWC.step[0] * 0 );
			int dim = pm.dims;
		}
	}

	void dcnnImpl::dcnnPredict( const Mat inputs, const Mat responses, Mat& prediction )
	{
		CURRENT_FUNCTION;
		CV_Assert( !optimalTheta.empty() );
		double cost = 0.f;
		// numImages  images.rows, convDim,convDim,numFilters
		//int nConDim = getConvolutionDimension();
		//int nPoolOutDim = getOutDimensionInPoolingLayer();
		//const int sz_am[] = { mParams.nImageNum, mParams.nFilterNum, nConDim, nConDim };

		vecToParamMat( optimalTheta, wc, bc, wh, bh );

		// falsch positve
		int preds[CLASSES_NUM][CLASSES_NUM] = { 0 };

		// summe of each number
		int testSum[CLASSES_NUM] = { 0 };

		int batchSize = lParams.batchSize;
		int index = 0;
		int same = 0;
		// Begining prediction
		utils.tic();
		cout << "++++++++++++++>Predicting<++++++++++++++" << endl;
		for (size_t row = 0; row < inputs.rows; row++)//= batchSize
		{
			Tensor convolvedMaps;
			mParams.nImageNum = 1;

			// forward step
			dcnnConvolution( inputs.row( row ), wc, bc, convolvedMaps );

			Tensor activationsPooled;
			Mat sm = Mat::zeros( mParams.nHiddenSize, mParams.nImageNum, CV_64F );
			// subpooling actived features response from convolutional layer
			dcnnSubpooling( mParams.nPoolDim, convolvedMaps, activationsPooled, sm );

			// feed the pooled blob in to softmax layer
			Mat probs = Mat::zeros( mParams.nClassesNum, mParams.nImageNum, CV_64F );
			dcnnSoftmax( wh, bh, sm, inputs.row( row ), probs );


			Mat col = probs.col( 0 );
			int maxIdx[] = { 0, 0 };
			minMaxIdx( col, 0, 0, 0, maxIdx );
			prediction.at<int>( index++ ) = maxIdx[0];

			int trueClasses = responses.at<int>( row );
			testSum[trueClasses] += 1;
			preds[trueClasses][maxIdx[0]]++;

//			{
//				char buf[64] = { 0 };
//				stringstream ss( stringstream::in | stringstream::out );
//				sprintf( buf, "Num%d\\%s", maxIdx[0],  fileList[row].c_str() );
//				ss << buf;
//				Mat img = inputs.row( row );
//				Mat src( mParams.nImageDim, mParams.nImageDim, CV_64F );
//				reshapeRawImage( img, mParams.nImageDim, mParams.nImageDim, src );
//				src = src * 255;
//				imwrite( ss.str(), src );
//			}
		}
		// Ending prediction
		utils.tac();

		// Output confusion matrix
		utils.getConfusionMatrix( preds, testSum );

		// Output prediction time info
		utils.getPredictionTimeInfo( inputs.rows );
	}

	void dcnnImpl::shuffleData( Mat& inputs, Mat& responses, Mat& _shuffle )
	{
		RNG rng( 65536 );

		//Mat outputs = ( Mat_<int>( 10, 1 ) << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 );
		//Mat inputs = ( Mat_<double>( 10, 1 ) << 0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1 );
		randShuffle( responses, 1, &rng );
		//_schuffle.create( inputs.rows, inputs.cols, CV_64F );
		for (size_t i = 0; i < inputs.rows; i++)
		{
			int index = responses.row( i ).at<int>( 0 );
			const double *src = inputs.ptr<double>( index );
			Mat row = inputs.row( index );
			//double *dst = _schuffle.ptr<double>( i );
			//memcpy( dst, src, sizeof( double )* inputs.cols );
			inputs.row( index ).copyTo( _shuffle.row( i ) );
		}
	}

	void dcnnImpl::usingDummyData()
	{

	}


	//dcnn::dcnn()
	//{
	//}

	//dcnn::~dcnnImpl()
	//{
	//}


	//////////////////////////////////////////////////////////////////////////
	/// deep convolution neural network implementation
	Ptr<dcnnImpl> dcnnImpl::create()
	{
		CURRENT_FUNCTION;
		return makePtr<dcnnImpl>();
	}

	Ptr<dcnnImpl> dcnnImpl::load(const String& filepath)
	{
		CURRENT_FUNCTION;
		FileStorage fs;
		fs.open( filepath, FileStorage::READ);

		Ptr<dcnnImpl> dcnn = makePtr<dcnnImpl>();
		dcnn->read(fs.getFirstTopLevelNode());
		return dcnn;
	}
}
