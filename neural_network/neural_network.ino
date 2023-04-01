#include <math.h>
#include <stdlib.h>


const int IMAGE_SIZE = 28; 
const int N_INPUT = IMAGE_SIZE * IMAGE_SIZE;  // square image size
const int N_OUTPUT = 10;      // 10 classes (digits 0-9)
const int KERNEL_SIZE = 3;
const int N_KERNAL = 3; // kernal size


// sigmoid activation function
double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}


void setup() {
  Serial.begin(9600);
}

void loop() {

  // init variables
  float input_image[IMAGE_SIZE][IMAGE_SIZE];
  
  float kernel[KERNEL_SIZE][KERNEL_SIZE];
  int convolved_image_size = N_INPUT-KERNEL_SIZE+1;

  float convolved_image[convolved_image_size][convolved_image_size];
  
  for (int i=0;i<convolved_image_size;i++) {
    for (int j=0;j<convolved_image_size;j++) {
      float sum = 0;
      for (int k=0;k<KERNEL_SIZE;k++) {
        for (int l=0;l<KERNEL_SIZE;l++) {
          sum += input_image[i+k][j+l] * kernel[k][l];
        }
      }
      convolved_image[i][j] = sum;
      }
  }
  
  float output_weights[N_INPUT][N_OUTPUT];
  float output[N_OUTPUT];


}
