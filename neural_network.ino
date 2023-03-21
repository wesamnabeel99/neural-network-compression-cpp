#include <math.h>
#include <stdlib.h>


const int IMAGE_SIZE = 28; // rows = 10 , cols = 10
const int N_INPUT = IMAGE_SIZE * IMAGE_SIZE;  // square image size
const int N_OUTPUT = 10;      // 10 classes (digits 0-9)
const int N_HIDDEN = 15;      // number of hidden units
const int N_KERNAL = 3; // kernal size


// sigmoid activation function
double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}

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
}

void convulve(double image[IMAGE_SIZE][IMAGE_SIZE], double kernal[N_KERNAL][N_KERNAL]) {
  int output[26][26];

  // convlution
  for (int i = 0; i < 26; i++) {
    for (int j = 0; j < 26; j++) {
      for (int m = 0; m < 3; m++) {
        for (int n = 0; n < 3; n++) {
          output[i][j] += image[i + m][j + n] + kernal[m][n];
        }
      }
    }
  }
}


void setup() {
  Serial.begin(9600);
}

void loop() {

  // init variables
  double input_image[IMAGE_SIZE][IMAGE_SIZE];
  double input_weights[N_INPUT][N_HIDDEN];
  double output_weights[N_HIDDEN][N_OUTPUT];
  double output[N_OUTPUT];
  double kernal[N_KERNAL][N_KERNAL];


  //init random image for test
  for (int i = 0; i < IMAGE_SIZE; i++) {
    for (int j = 0; j < IMAGE_SIZE; j++) {
      input_image[i][j] = rand() % 256  ;
    }
  }

  // initialize input_weights, output_weights & kernal weights with uniformly random values (should be replaced with trained weights)
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

  for (int i = 0; i < N_KERNAL; i++) {
    for (int j = 0; j < N_KERNAL; j++) {
      kernal[i][j] = (double)rand() / RAND_MAX * 2 - 1;
    }
  }

  convulve(input_image, kernal);

  // flatten 2d image to 1d image

  double flatten_image[N_INPUT];

  for (int i = 0; i < IMAGE_SIZE; i++) {
    for (int j = 0; j < IMAGE_SIZE; j++) {
      flatten_image[i + j] = input_image[i][j];
    }
  }

  // forward probagation, get the output
  test_neural_network(flatten_image, input_weights, output_weights, output);

}
