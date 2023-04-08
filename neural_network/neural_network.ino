#include <math.h>
#include <stdlib.h>


const int IMAGE_SIZE = 28; 
const int N_INPUT = IMAGE_SIZE * IMAGE_SIZE;  // square image size
const int N_OUTPUT = 10;      // 10 classes (digits 0-9)
const int KERNEL_SIZE = 3;
const int N_KERNAL = 3; 
const int STRIDE = 2;
const int CONVOLVED_IMAGE_SIZE = IMAGE_SIZE-KERNEL_SIZE+1;
const int POOL_SIZE = CONVOLVED_IMAGE_SIZE/STRIDE;

int input_image[N_INPUT];
float kernel[KERNEL_SIZE][KERNEL_SIZE];
int input_image_square[IMAGE_SIZE][IMAGE_SIZE];
float pool_output[POOL_SIZE][POOL_SIZE];
float convolved_image[CONVOLVED_IMAGE_SIZE][CONVOLVED_IMAGE_SIZE];


float output_weights[N_INPUT][N_OUTPUT];
float output[N_OUTPUT];

  
// sigmoid activation function
double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}


void setup() {
  Serial.begin(9600);
  
 // Serial.println("setup done  !!");

  
}

void loop() {
  
  for (int i=0;i<IMAGE_SIZE;i++) {
    for (int j=0;j<IMAGE_SIZE;j++)
      input_image_square[i][j] = input_image[IMAGE_SIZE*i + j];
  }

  // print the squared image
  for (int i =0;i<IMAGE_SIZE;i++) {
    for (int j=0;j<IMAGE_SIZE;j++) {
      Serial.print(input_image_square[i][j]);
      Serial.print(" ");
      }
      Serial.println("");
    }

    
  // convolve the input image
    Serial.println("convolution!!!");
  for (int i=0;i<CONVOLVED_IMAGE_SIZE;i++) {
    for (int j=0;j<CONVOLVED_IMAGE_SIZE;j++) {
      float sum = 0;
      for (int k=0;k<KERNEL_SIZE;k++) {
        for (int l=0;l<KERNEL_SIZE;l++) {
          sum += input_image_square[i+k][j+l] * kernel[k][l];
        }
      }
      convolved_image[i][j] = sum;
      }
  }
  Serial.println("max pooling!");

  // implement max pooling
    for (int i = 0; i < CONVOLVED_IMAGE_SIZE; i += STRIDE) {
        for (int j = 0; j < CONVOLVED_IMAGE_SIZE; j += STRIDE) {
            int max_val = convolved_image[i][j];
            for (int m = i; m < i + STRIDE; m++) {
                for (int n = j; n < j + STRIDE; n++) {
                    if (convolved_image[m][n] > max_val) {
                        max_val = convolved_image[m][n];
                    }
                }
            }
            pool_output[i/STRIDE][j/STRIDE] = max_val;
        }
    }

    
  // Reshape the 2D array to a 1D array for input to the neural network
  float flattened_input[POOL_SIZE* POOL_SIZE];
  int k = 0;
  for (int i = 0; i < POOL_SIZE; i++) {
    for (int j = 0; j < POOL_SIZE; j++) {
      flattened_input[k] = pool_output[i][j];
      k++;
    }
  }

  // Feed the flattened input to the fully connected neural network
  for (int i = 0; i < N_OUTPUT; i++) {
    float sum = 0;
    for (int j = 0; j < N_INPUT; j++) {
      sum += flattened_input[j] * output_weights[j][i];
    }
    output[i] = sigmoid(sum);
  }

  float maximumOutput = output[0];
  int winningClass = 0;
  
  for (int i = 1; i< N_OUTPUT;i++) {
    if (output[i] > maximumOutput) {
        maximumOutput= output[i];
        winningClass = i;
      }
  }
      Serial.println("output!!!");
      Serial.println(winningClass);
}
