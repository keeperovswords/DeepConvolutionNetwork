# deepconvolutionnetwork

<h1>training</h1> 
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

