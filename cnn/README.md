# deepconvolutionnetwork

<h1>training</h1> 
This model has only convolution layer + mean sub-pooling layer and a dense fully connected output layer.
For penalty function we used L2-norm and for minimizing the mini-batch SGD+momentum has been applied.


<h2>write CMakeLists.txt</h2>

cmake_minimum_required(VERSION 2.8)
project( cnn )
find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( cnn main.cpp Utilities.cpp dcnn.cpp )
target_link_libraries( cnn ${OpenCV_LIBS} )

<h2>Compiling</h2>
# generating make file
: cmake .

# compiling
: make

# run module
: ./cnn


<h1>test</h1>

<h2>confusion matrix</h2>
The testing result looks as follows:

++++++++++++++>Predicting<++++++++++++++
[Correctly Classified Instances:	9369	0.9369]
[Incorrectly Classified Instances:	631	0.0631]
[Total Number of Instances:	10000]

====Detailed Accuracy By Class====
	Class		TP Rate		FP Rate		Precision		Recall		F-Measure		ROC Area		
		0 		0.984 		0.006 		0.951 		0.984 		0.967
		1 		0.978 		0.003 		0.973 		0.978 		0.975
		2 		0.920 		0.008 		0.932 		0.920 		0.926
		3 		0.929 		0.008 		0.928 		0.929 		0.928
		4 		0.944 		0.009 		0.917 		0.944 		0.930
		5 		0.904 		0.005 		0.945 		0.904 		0.924
		6 		0.962 		0.006 		0.942 		0.962 		0.952
		7 		0.922 		0.007 		0.937 		0.922 		0.929
		8 		0.919 		0.007 		0.935 		0.919 		0.927
		9 		0.902 		0.010 		0.906 		0.902 		0.904

====Confusion Matrix====
		0		1		2		3		4		5		6		7		8		9	<------Labeled as
| 0 |	964		0		1		1		0		3		6		2		2		1		980
| 1 |	0		1110		4		2		0		0		4		1		14		0		1135
| 2 |	7		4		949		14		9		2		10		17		14		6		1032
| 3 |	5		1		18		938		0		18		2		12		8		8		1010
| 4 |	1		2		4		1		927		0		12		4		3		28		982
| 5 |	10		2		2		26		6		806		15		7		11		7		892
| 6 |	9		3		3		2		9		7		922		1		2		0		958
| 7 |	0		8		28		7		7		1		0		948		2		27		1028
| 8 |	8		5		4		10		10		9		8		8		895		17		974
| 9 |	10		6		5		10		43		7		0		12		6		910		1009
Prediction accuracy: 93.69% in 10000s Samples
