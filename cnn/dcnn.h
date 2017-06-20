#pragma once
#include "opencv2/ml/ml.hpp"
#include <algorithm>
#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/core/matx.hpp"
#include <vector>
#include "Utilities.h"

#include <iterator>
#include <algorithm>

using namespace cv;
using namespace std;
namespace DeepLearning {

	// property implementation macros

#define CV_IMPL_PROPERTY_RO(type, name, member) \
    inline type get##name() const { return member; }

#define CV_HELP_IMPL_PROPERTY(r_type, w_type, name, member) \
    CV_IMPL_PROPERTY_RO(r_type, name, member) \
    inline void set##name(w_type val) { member = val; }

#define CV_HELP_WRAP_PROPERTY(r_type, w_type, name, internal_name, internal_obj) \
    r_type get##name() const { return internal_obj.get##internal_name(); } \
    void set##name(w_type val) { internal_obj.set##internal_name(val); }

#define CV_IMPL_PROPERTY(type, name, member) CV_HELP_IMPL_PROPERTY(type, type, name, member)
#define CV_IMPL_PROPERTY_S(type, name, member) CV_HELP_IMPL_PROPERTY(type, const type &, name, member)

#define CV_WRAP_PROPERTY(type, name, internal_name, internal_obj)  CV_HELP_WRAP_PROPERTY(type, type, name, internal_name, internal_obj)
#define CV_WRAP_PROPERTY_S(type, name, internal_name, internal_obj) CV_HELP_WRAP_PROPERTY(type, const type &, name, internal_name, internal_obj)

#define CV_WRAP_SAME_PROPERTY(type, name, internal_obj) CV_WRAP_PROPERTY(type, name, name, internal_obj)
#define CV_WRAP_SAME_PROPERTY_S(type, name, internal_obj) CV_WRAP_PROPERTY_S(type, name, name, internal_obj)

		enum TrainingMethods {
			BACKPROP = 0, //!< The back-propagation algorithm.
			RPROP = 1 //!< The RPROP algorithm. See @cite RPROP93 for details.
		};
		
		/** possible activation functions */
		enum ActivationFunctions {
			/** Identity function: \f$f(x)=x\f$ */
			IDENTITY = 0,
			/** Symmetrical sigmoid: \f$f(x)=\beta*(1-e^{-\alpha x})/(1+e^{-\alpha x}\f$
			@note
			If you are using the default sigmoid activation function with the default parameter values
			fparam1=0 and fparam2=0 then the function used is y = 1.7159\*tanh(2/3 \* x), so the output
			will range from [-1.7159, 1.7159], instead of [0,1].*/
			SIGMOID_SYM = 1,
			/** Gaussian function: \f$f(x)=\beta e^{-\alpha x*x}\f$ */
			GAUSSIAN = 2,
			/** \f$f(x)= (e^{z} - e^{-z})/(e^{z} + e^{-z})  \f$ */
			TANH = 3,
			RECTIFIED_LINEAR = 4
		};

		/** Train options */
		enum TrainFlags {
			/** Update the network weights, rather than compute them from scratch. In the latter case
			the weights are initialized using the Nguyen-Widrow algorithm. */
			UPDATE_WEIGHTS = 1,
			/** Do not normalize the input vectors. If this flag is not set, the training algorithm
			normalizes each input feature independently, shifting its mean value to 0 and making the
			standard deviation equal to 1. If the network is assumed to be updated frequently, the new
			training data could be much different from original one. In this case, you should take care
			of proper normalization. */
			NO_INPUT_SCALE = 2,
			/** Do not normalize the output vectors. If the flag is not set, the training algorithm
			normalizes each output feature independently, by transforming it to the certain range
			depending on the used activation function. */
			NO_OUTPUT_SCALE = 4
		};
		
		/////////////////////////////////////////////////////////////////////////////////////////////////////
		/*								 Basic parameter of model										   */
		struct ModelParams
		{
			ModelParams()
			{
				termCrit = TermCriteria( TermCriteria::COUNT + TermCriteria::EPS, 1000, 0.01 );
				trainMethod = DeepLearning::BACKPROP;
				bpDWScale = 1, bpMomentScale = 1;
				activeFun = SIGMOID_SYM;
			}

			ModelParams( int _nFilterNum, int _nFilterDim, int _nPoolDim, int _nImageDim, /*int _nImageNum,*/ int _nClassesNum )
				:nFilterNum( _nFilterNum ), nFilterDim( _nFilterDim ), nPoolDim( _nPoolDim ), nImageDim( _nImageDim ), nClassesNum( _nClassesNum )/*, nImageNum(_nImageNum )*/
			{
				termCrit = TermCriteria( TermCriteria::COUNT + TermCriteria::EPS, 1000, 0.01 );
				int outDim = _nImageDim - _nFilterDim + 1;
				nConOutDim = outDim;
				outDim = outDim / _nPoolDim;
				nHiddenSize = outDim * outDim * _nFilterNum;
				trainMethod = DeepLearning::BACKPROP;
				activeFun = SIGMOID_SYM;
				
			}

			TermCriteria termCrit;
			int trainMethod;		// training method
			int activeFun;			// activation function


			double bpDWScale;
			double bpMomentScale;
			int nFilterNum;
			int nFilterDim;
			int nPoolDim;
			int nImageDim;
			int nHiddenSize;
			int nClassesNum;
			int nImageNum;
			int nConOutDim;
		};

		/////////////////////////////////////////////////////////////////////////////////////////////////////
		/*								 Basic parameter for learning			     					   */
		struct LearnParams
		{
			LearnParams()
			{
				epochs = 3;
				batchSize = 256;
				alpha = 0.1f;
				momentum = 0.95;
				lambda = 3e-4;
			}

			int epochs;
			int batchSize;
			float alpha;
			float momentum;
			float lambda;
		};

		

	
	class dcnnImpl :
		public ml::StatModel
	{
	public:
		dcnnImpl()
		{
			clear();
			//setActivationFunction(SIGMOID_SYM, 0, 0);
			//setLayerSizes(Mat());
			//setTrainMethod(RPROP, 0.1, FLT_EPSILON);
		}
		~dcnnImpl() {};
		//cnn();
		//~dcnn();
		/** Available training methods */
		////////////////////////////StatModel Definition////////////////////////////
		/** @brief Returns the number of variables in training samples */
		int getVarCount() const { return layer_sizes.empty() ? 0 : layer_sizes[0]; };

		//CV_WRAP virtual bool empty() const;

		/** @brief Returns true if the model is trained */
		bool isTrained() const { return trained; }

		/** @brief Returns true if the model is classifier */
		bool isClassifier() const { return 1; }

		/** @brief Trains the statistical model

		@param trainData training data that can be loaded from file using TrainData::loadFromCSV or
		created with TrainData::create.
		@param flags optional flags, depending on the model. Some of the models can be updated with the
		new training samples, not completely overwritten (such as NormalBayesClassifier or ANN_MLP).
		*/
		bool train( const Ptr<ml::TrainData>& trainData, int flags = 0 )
		{
			CURRENT_FUNCTION;
			const int MAX_ITER = 1000;
			const double DEFAULT_EPSILON = FLT_EPSILON;

			// initialize training data
			Mat inputs = trainData->getTrainSamples();
			Mat outputs = trainData->getTrainResponses();
			Mat sw = trainData->getTrainSampleWeights();
			preprocessTrain( inputs, outputs, sw, flags );

			// ... and link weights
			if (!( flags & DeepLearning::UPDATE_WEIGHTS ))
			{
				// TODO init_weights
			}

			TermCriteria termcrit;
			termcrit.type = TermCriteria::COUNT + TermCriteria::EPS;
			termcrit.maxCount = max( ( mParams.termCrit.type & CV_TERMCRIT_ITER ? mParams.termCrit.maxCount : MAX_ITER ), 1 );
			termcrit.epsilon = max( ( mParams.termCrit.type & CV_TERMCRIT_EPS ? mParams.termCrit.epsilon : DEFAULT_EPSILON ), DBL_EPSILON );

			int iter = __trainlWithBackprop( inputs, outputs, sw, termcrit );

			trained = iter > 0;
			return trained;
		}


		/** @brief Trains the statistical model

		@param samples training samples
		@param layout See ml::SampleTypes.
		@param responses vector of responses associated with the training samples.
		*/
		bool train( InputArray samples, int layout, InputArray responses ) { return 1; };

		/** @brief Computes error on the training or test dataset

		@param data the training data
		@param test if true, the error is computed over the test subset of the data, otherwise it's
		computed over the training subset of the data. Please note that if you loaded a completely
		different dataset to evaluate already trained classifier, you will probably want not to set
		the test subset at all with TrainData::setTrainTestSplitRatio and specify test=false, so
		that the error is computed for the whole new set. Yes, this sounds a bit confusing.
		@param resp the optional output responses.

		The method uses StatModel::predict to compute the error. For regression models the error is
		computed as RMS, for classifiers - as a percent of missclassified samples (0%-100%).
		*/
		float calcError( const Ptr<ml::TrainData>& data, bool test, OutputArray resp ) const
		{
			return 0.f;
		}

		/** @brief Predicts response(s) for the provided sample(s)

		@param samples The input samples, floating-point matrix
		@param results The optional output matrix of results.
		@param flags The optional flags, model-dependent. See cv::ml::StatModel::Flags.
		*/
		float predict( InputArray samples, OutputArray results = noArray(), int flags = 0 ) const
		{
			Mat response( 1, 10, CV_32F );
			//T.B.D
			return 0.0f;
			//return this->predict(samples, response); 
		}

		void clear()
		{
			// clear weights 
			trained = false;
		}

		int getLayerCount() const { return ( int ) layer_sizes.size(); }
// 		cv::Mat getLayerSizes() const
// 		{
// 			return Mat_<int>( layer_sizes, true );
// 		}
		////////////////////////////StatModel Definition////////////////////////////
		
		
		////////////////////////////Algorithm Definition////////////////////////////
		void write( FileStorage& fs ) const
		{
			CURRENT_FUNCTION;	
			if (theta.size() == 0 || wc.empty() || wh.empty() || bc.empty() || bh.empty())
				CV_Error( CV_StsBadArg, "Adaboost have not been trained" );

			// convolution + subpooling 
			int layer = 2;

			fs << "layer_sizes" << layer;


			write_params( fs );

			size_t esz = wc.elemSize();
			fs << "cov_weight" << "[";
			fs.writeRaw( "d", wc.ptr(), wc.total()*esz );

			esz = bc.elemSize();
			fs << "]" << "con_bias" << "[";
			fs.writeRaw( "d", bc.ptr(), bc.total()*esz );

			esz = wh.elemSize();
			fs << "]" << "sp_weight" << "[";
			fs.writeRaw( "d", wh.ptr(), wh.total()*esz );

			esz = bh.elemSize();
			fs << "]" << "sp_bias" << "[";
			fs.writeRaw( "d", bh.ptr(), bh.total()*esz );

			fs << "]";
		}

		
		void read( const FileNode& fn )
		{
			CURRENT_FUNCTION;
			
			read_params( fn);
			
			__initWeightParams(mParams, optimalTheta);
			optimalTheta.clear();
			//int i, l_count = layer_count();
			//read_params( ( *it2 ) );
			size_t esz = wc.elemSize();
			cout << esz << endl;
			FileNode w = fn["cov_weight"];
			DVec cov_weight, con_bias, sp_weight,sp_bias;
			utils.readVectorOrMat(fn["cov_weight"], cov_weight);
			utils.readVectorOrMat(fn["con_bias"], con_bias);
			utils.readVectorOrMat(fn["sp_weight"], sp_weight);
			utils.readVectorOrMat(fn["sp_bias"], sp_bias);
//			optimalTheta.insert(std::end(optimalTheta), std::begin(cov_weight), std::end(cov_weight));
//			optimalTheta.insert(std::end(optimalTheta), std::begin(con_bias), std::end(con_bias));
//			optimalTheta.insert(std::end(optimalTheta), std::begin(sp_weight), std::end(sp_weight));
//			optimalTheta.insert(std::end(optimalTheta), std::begin(sp_bias), std::end(sp_bias));
//

			optimalTheta.insert(optimalTheta.end(), cov_weight.begin(), cov_weight.end());
			optimalTheta.insert(optimalTheta.end(), con_bias.begin(), con_bias.end());
			optimalTheta.insert(optimalTheta.end(), sp_weight.begin(), sp_weight.end());
			optimalTheta.insert(optimalTheta.end(), sp_bias.begin(), sp_bias.end());

			vecToParamMat( optimalTheta, wc, bc, wh, bh );
 			trained = true;
		}

// 		void save( const String& filename ) const
// 		{
// 			FileStorage fs;
// 			fs.open( filename, FileStorage::WRITE );
// 			return  write( fs );
// 		}

		String getDefaultName()const { return "opencv_ml_dcnn"; }
		////////////////////////////Algorithm Definition////////////////////////////
		
		//CV_WRAP virtual cv::Mat getLayerSizes() const = 0;
		//CV_WRAP virtual void initWeightParams( int imageDim, int filterDim, int numFilters, int poolDim, int numClasses, DVec& grad ) = 0;
		CV_WRAP static Ptr<dcnnImpl> create();
		CV_WRAP static Ptr<dcnnImpl> load(const String& filepath);
		CV_WRAP bool gradientCheck( Ptr<ml::TrainData>tdata );
		CV_WRAP void setModelParameter( ModelParams model );
		CV_WRAP void setLearningParameter( LearnParams model );
		CV_WRAP void modelTest( Ptr<ml::TrainData>tdata );
		
		private:
		void setTrainMethod( int method, double param1, double param2 );
		void preprocessTrain( const Mat& inputs, const Mat& outputs, Mat& sample_weights, int flags );
		
		void __initWeightParams( ModelParams mp,/*int _imageDim, int _filterDim, int _filterNums, int _poolDim, int _numClasses,*/ DVec& _theta );
		int __trainlWithBackprop( Mat inputs, Mat outputs, const Mat& _sw, TermCriteria termCrit );
		double dcnnFireUp( DVec theta, const Mat inputs, const Mat labels, /*int numClasses, int filterDim, int filterNum, int poolDim,*/ DVec& _grad );
		void dcnnConvolution( const Mat inputs, const Mat _wc, const Mat _bc, /*MVec*/Tensor &features );
		void dcnnSubpooling( int nPoolDim, Tensor convolvedFeatures, Tensor& activationsPooled/*maybe useless, reserved row*/, Mat& sm );
		void dcnnSoftmax( const Mat _wh, const Mat _bh, Mat activationsPooled, const Mat inputs, Mat& entropy );
		double dcnnCost( const Mat _groundTruth, const Mat _wc, const Mat _wh, Mat _probs );
		void dcnnBackPropagation( const Mat _wc, const Mat _wh, const Mat inputs, const Tensor convolvedMaps, const Mat _groundTruth, const Mat _probs, const Mat _pooledBlobs, DVec& _grad );
		void dcnnComputeNumericalGradient( DVec theta, const Mat images, const Mat labels, int numClasses, int filterDim, int filterNum, int poolDim, /*DVec grad,*/ DVec& numGrad );
		void dcnnWeightErrorBackToConvolutionlayer( Mat _wc, const Mat deltaCon, const Mat images, Mat& gradWC, Mat& gradBC );
		void dcnnUpsampling( const Tensor convolvedMaps, const Tensor ds, Mat& _deltaCon );
		void dcnnPredict( const Mat inputs, const Mat responses, Mat& prediction );
		void dcnnBiasErrorBackToConvolutionlayer( const Mat src, MVec& vec, Mat& dst );
		void shuffleData( Mat& inputs, Mat& responses, Mat& _shuffle );
		void usingDummyData();
		
		void paramMatToVec( /*, const Mat param*/const Mat param1, const Mat param2, const Mat param3, const Mat param4, DVec& vec );
		void vecToParamMat( const DVec vec, Mat& tmp_wc, Mat& tmp_bc, Mat& tmp_wh, Mat& tmp_bh );
		
		
		void write_params( FileStorage& fs ) const;
		void read_params( const FileNode& fn );
		
		void mat2DtoTensor( const Mat src, int nOutDim, int nImageNum, int nFilterNum, Tensor& ds );
		Mat tensorTo2DMat( const Tensor src, const int* size );
		Mat kron( const Mat src1, const Mat src2 );
		void convertTensorToDvec(const Tensor in, DVec& out);
		int getConvolutionDimension() const;
		int getOutDimensionInPoolingLayer() const;
		void logsig( const Mat src, Mat& dst );
		void reshapeRawImage( const Mat src, const int nDimRow, const int nDimCol, Mat& dst );
		
		bool trained;
		
		vector<int> layer_sizes;
		// neuron weights
		MVec  weights;

		ModelParams mParams;

		LearnParams lParams;

		// cnn parameters
		Mat wc, wh;
		Mat bc, bh;

		// corresponding parameter to cnn parameters in vector 
		DVec theta;
		DVec optimalTheta;

		// Dummy Data
		Mat groundTruth, probs, pooledBlobs;
		MVec _dummyInputs;


		Utilities utils;
	};



	/*
		struct SigmoidFunctor
		{

			SigmoidFunctor( ) {}

			template<typename TFloat>
			inline TFloat operator()( TFloat x )
			{
				return ( TFloat ) 1 / ( ( TFloat ) 1 + exp( -x ) );
			}


			inline Mat operatorM(  Mat src, Mat dst )
			{
				Mat ret = Mat::zeros( src.rows, src.cols, src.type );
				for (size_t c = 0; c < src.cols; c++)
				{
					for (size_t r = 0; r < src.rows; r++)
					{
						ret.at<double>( c, r ) = ( double ) 1 / ( ( double ) 1 + exp( -src.at<double>( c, r ) ) );
					}
				}
				return ret;
			}
		};*/

	
}
