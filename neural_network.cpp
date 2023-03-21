#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;

const int N_INPUT = 28 * 28;  // 28x28 input image size
const int N_OUTPUT = 10;      // 10 classes (digits 0-9)
const int N_HIDDEN = 100;      // number of hidden units

// sigmoid activation function
double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}

// forward propagation
void test_neural_network(double input_image[N_INPUT], double input_weights[N_INPUT][N_HIDDEN],
             double output_weights[N_HIDDEN][N_OUTPUT], double output[N_OUTPUT]) {
  double hidden[N_HIDDEN];
  // dot product between input and hidden
  for (int i = 0; i < N_HIDDEN; i++) {
    for (int j = 0; j < N_INPUT; j++) {
      hidden[i] += input_weights[j][i] * input_image[j];
    }
    hidden[i] = sigmoid(hidden[i]);
  }
  // dot product between hiddent and output
  for (int i = 0; i < N_OUTPUT; i++) {
    for (int j = 0; j < N_HIDDEN; j++) {
      output[i] += output_weights[j][i] * hidden[j];
    }
    output[i] = sigmoid(output[i]);
  }

  for (int i=0 ;i<N_OUTPUT;i++) {
    cout<<output[i]<<" ";
  }
}

int main() {
  double input_image[N_INPUT];
  double input_weights[N_INPUT][N_HIDDEN];
  double output_weights[N_HIDDEN][N_OUTPUT];
  double output[N_OUTPUT];


  // initialize input_image, taking random pixels, will be replaced by actual image from arduino

  for (int i=0;i<N_INPUT;i++) {
    input_image[i] = rand() %256  ;
  }

  // initialize input_weights and output_weights with uniformly random values
  for (int i = 0; i < N_INPUT; i++) {
    for (int j = 0; j < N_HIDDEN; j++) {
      input_weights[i][j] = (double)rand() / RAND_MAX * 2 - 1;
    }
  }
  for (int i = 0; i < N_HIDDEN; i++) {
    for (int j = 0; j < N_OUTPUT; j++) {
      output_weights[i][j] = (double)rand() / RAND_MAX * 2 - 1;
    }
  }

  test_neural_network(input_image, input_weights, output_weights, output);

  // use output for prediction, evaluation, etc.

  return 0;
}
