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

<h2>Predicting</h2>
The testing result looks as follows:


<table>
  <tbody>
    <tr>
	<th align="center"><th align="center">Total Samples </th></th>
	<th align="center">Correctly Classified</th>
	<th align="center">Incorrectly Classified</th>
    </tr>
    <tr>
      <td>10000</td>
      <td align="center">0.984</td>
      <td align="center">9369</td>
      <td align="center">631</td>
    </tr>
  </tbody>
</table>
Prediction accuracy: 93.69% in 10000s Samples
###DETAILS###
<table>
  <tbody>
    <tr>
      <th>Class</th>
      <th align="center">TP Rate</th>
      <th align="right">FP Rate</th>
      <th align="right">Precision</th>
      <th align="right">Recall</th>
      <th align="right">F-Measure</th>
    </tr>
    <tr>
      <td>0</td>
      <td align="center">0.984</td>
      <td align="center">0.006</td>
      <td align="center">0.951</td>
      <td align="center">0.984</td>
      <td align="center">0.967</td>
    </tr>
    <tr>
      <td>1</td>
      <td align="center">0.978</td>
      <td align="center">0.003</td>
      <td align="center">0.973</td>
      <td align="center">0.978</td>
      <td align="center">0.975</td>
    </tr>
    <tr>
      <td>2</td>
      <td align="center">0.920</td>
      <td align="center">0.008</td>
      <td align="center">0.932</td>
      <td align="center">0.920</td>
      <td align="center">0.926</td>
    </tr>
    <tr>
      <td>3</td>
      <td align="center">0.929</td>
      <td align="center">0.008</td>
      <td align="center">0.928</td>
      <td align="center">0.929</td>
      <td align="center">0.928</td>
    </tr>
    <tr>
      <td>4</td>
      <td align="center">0.944</td>
      <td align="center">0.009</td>
      <td align="center">0.917</td>
      <td align="center">0.944</td>
      <td align="center">0.930</td>
    </tr>
    <tr>
      <td>5</td>
      <td align="center">0.904</td>
      <td align="center">0.005</td>
      <td align="center">0.945</td>
      <td align="center">0.904</td>
      <td align="center">0.924</td>
    </tr>
    <tr>
      <td>6</td>
      <td align="center">0.962</td>
      <td align="center">0.006</td>
      <td align="center">0.942</td>
      <td align="center">0.962</td>
      <td align="center">0.952</td>
    </tr>
    <tr>
      <td>7</td>
      <td align="center">0.922</td>
      <td align="center">0.007</td>
      <td align="center">0.937</td>
      <td align="center">0.922</td>
      <td align="center">0.929</td>
    </tr>
    <tr>
      <td>8</td>
      <td align="center">0.919</td>
      <td align="center">0.007</td>
      <td align="center">0.935</td>
      <td align="center">0.919</td>
      <td align="center">0.927</td>
    </tr>
    <tr>
      <td>9</td>
      <td align="center">0.902 </td>
      <td align="center">0.010</td>
      <td align="center">0.906</td>
      <td align="center">0.902</td>
      <td align="center">0.904</td>
    </tr>
  </tbody>
</table>

<h2>confusion matrix</h2>
<table>
  <tbody>
    <tr>
      <th></th>
      <th>0</th>
      <th align="center">1</th>
      <th align="right">2</th>
      <th align="right">3</th>
      <th align="right">4</th>
      <th align="right">5</th>
      <th align="right">6</th>
      <th align="right">7</th>
      <th align="right">8</th>
      <th align="right">9</th>
      <th align="right">labled as</th>
    </tr>
    <tr>
      <td>0</td>
      <td align="center">964</td>
      <td align="center">0</td>
      <td align="center">1</td>
      <td align="center">1</td>
      <td align="center">0</td>
      <td align="center">3</td>
      <td align="center">6</td>
      <td align="center">2</td>
      <td align="center">2</td>
      <td align="center">1</td>
      <td align="center">980</td>
    </tr>
    <tr>
      <td>1</td>
      <td align="center">0</td>
      <td align="center">1110</td>
      <td align="center">4</td>
      <td align="center">2</td>
      <td align="center">0</td>
      <td align="center">0</td>
      <td align="center">4</td>
      <td align="center">1</td>
      <td align="center">14</td>
      <td align="center">1</td>
      <td align="center">1135</td>
    </tr>
    <tr>
      <td>2</td>
      <td align="center">7</td>
      <td align="center">4</td>
      <td align="center">949</td>
      <td align="center">14</td>
      <td align="center">9</td>
      <td align="center">2</td>
      <td align="center">10</td>
      <td align="center">17</td>
      <td align="center">14</td>
      <td align="center">6</td>
      <td align="center">1032</td>
    </tr>
    <tr>
      <td>3</td>
      <td align="center">5</td>
      <td align="center">1</td>
      <td align="center">18</td>
      <td align="center">938</td>
      <td align="center">0</td>
      <td align="center">18</td>
      <td align="center">2</td>
      <td align="center">12</td>
      <td align="center">8</td>
      <td align="center">8</td>
      <td align="center">1010</td>
    </tr>
    <tr>
      <td>4</td>
      <td align="center">1</td>
      <td align="center">2</td>
      <td align="center">4</td>
      <td align="center">1</td>
      <td align="center">927</td>
      <td align="center">0</td>
      <td align="center">12</td>
      <td align="center">4</td>
      <td align="center">3</td>
      <td align="center">28</td>
      <td align="center">982</td>
    </tr>
    <tr>
      <td>5</td>
      <td align="center">10</td>
      <td align="center">2</td>
      <td align="center">2</td>
      <td align="center">26</td>
      <td align="center">6</td>
      <td align="center">806</td>
      <td align="center">15</td>
      <td align="center">7</td>
      <td align="center">11</td>
      <td align="center">7</td>
      <td align="center">982</td>
    </tr>
    <tr>
      <td>6</td>
      <td align="center">9</td>
      <td align="center">3</td>
      <td align="center">3</td>
      <td align="center">2</td>
      <td align="center">9</td>
      <td align="center">7</td>
      <td align="center">922</td>
      <td align="center">1</td>
      <td align="center">2</td>
      <td align="center">0</td>
      <td align="center">958</td>
    </tr>
    <tr>
      <td>7</td>
      <td align="center">0</td>
      <td align="center">8</td>
      <td align="center">28</td>
      <td align="center">7</td>
      <td align="center">7</td>
      <td align="center">1</td>
      <td align="center">0</td>
      <td align="center">948</td>
      <td align="center">2</td>
      <td align="center">27</td>
      <td align="center">1028</td>
    </tr>
    <tr>
      <td>8</td>
      <td align="center">8</td>
      <td align="center">5</td>
      <td align="center">4</td>
      <td align="center">10</td>
      <td align="center">10</td>
      <td align="center">9</td>
      <td align="center">8</td>
      <td align="center">8</td>
      <td align="center">895</td>
      <td align="center">17</td>
      <td align="center">974</td>
    </tr>
    <tr>
      <td>9</td>
      <td align="center">10</td>
      <td align="center">6</td>
      <td align="center">5</td>
      <td align="center">10</td>
      <td align="center">43</td>
      <td align="center">7</td>
      <td align="center">0</td>
      <td align="center">12</td>
      <td align="center">6</td>
      <td align="center">910</td>
      <td align="center">1009</td>
    </tr>
  </tbody>
</table>









Prediction accuracy: 93.69% in 10000s Samples
