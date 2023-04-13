#include <math.h>
#include <stdlib.h>


const uint8_t IMAGE_SIZE = 28; 
const uint8_t N_OUTPUT = 10;      // 10 classes (digits 0-9)
const uint8_t KERNEL_SIZE = 3;
const uint8_t STRIDE = 2;
const uint8_t CONVOLVED_IMAGE_SIZE = IMAGE_SIZE-KERNEL_SIZE+1;
const uint8_t POOL_SIZE = (CONVOLVED_IMAGE_SIZE-1)/STRIDE + 1; 
const uint8_t N_INPUT = POOL_SIZE * POOL_SIZE;  // input neurouns
const float SCALE = 100000.0;

byte input_image_square[IMAGE_SIZE][IMAGE_SIZE] = 
{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 221, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 229, 219, 104, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 235, 140, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 118, 227, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 236, 133, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 243, 93, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 243, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 189, 236, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 1, 208, 169, 0, 0, 0, 0, 0, 0, 64, 151, 151, 135, 74, 1, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 26, 254, 89, 0, 0, 0, 0, 6, 142, 254, 224, 211, 181, 241, 70, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 26, 254, 68, 0, 0, 0, 2, 161, 254, 104, 7, 0, 0, 80, 223, 15, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 57, 254, 15, 0, 0, 0, 150, 231, 68, 1, 0, 0, 0, 9, 231, 26, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 79, 254, 15, 0, 0, 24, 228, 66, 0, 0, 0, 0, 0, 0, 196, 87, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 73, 254, 43, 0, 0, 116, 251, 7, 0, 0, 0, 0, 0, 0, 196, 100, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 13, 230, 147, 0, 0, 60, 255, 70, 0, 0, 0, 0, 0, 4, 209, 84, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 203, 232, 4, 0, 42, 253, 74, 0, 0, 0, 0, 0, 114, 233, 17, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 87, 252, 147, 0, 0, 154, 229, 132, 123, 123, 63, 93, 248, 65, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 169, 249, 137, 23, 8, 80, 100, 101, 107, 145, 192, 51, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 139, 251, 224, 144, 115, 115, 195, 254, 187, 48, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 141, 203, 244, 180, 129, 67, 2, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};



float convolved_image[CONVOLVED_IMAGE_SIZE][CONVOLVED_IMAGE_SIZE];
float output[N_OUTPUT] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};


// trained output_weights
int output_weights[POOL_SIZE*POOL_SIZE][N_OUTPUT] = {{}};
float sigmoid(float x) {
  return 1.0 / (1.0 + exp(-x));
}


void setup() {
  Serial.begin(9600);
  
}

void loop() {


  // convolve the input image
  float kernel[KERNEL_SIZE][KERNEL_SIZE]= {{0.84763249,0.24134909,0.47557599},{0.37828915,0.76494428,0.94806886}, {0.90188304,0.49482442,0.57135601}}; //random kernel
  float kernel_sum = 0.0;


  Serial.println("convolution!!!");
  
              int compressed = 0;
      int nonCompressed = 0;
      
  for (int i=0;i<CONVOLVED_IMAGE_SIZE;i++) {
    
    //Serial.println("hello there, I'm convolving and there's no problem");

    for (int j=0;j<CONVOLVED_IMAGE_SIZE;j++) {
      float sum = 0;

      for (int k=0;k<KERNEL_SIZE;k++) {
        for (int l=0;l<KERNEL_SIZE;l++) {
          if (kernel[k][l]>=0.4) {
                      sum += input_image_square[i+k][j+l]/255.0 * kernel[k][l];
                      nonCompressed++;
            } else {
              compressed++;
              }
        }
      }
      convolved_image[i][j] = sum;
      }

  }
      //Serial.print("compressed:");
      //Serial.print(compressed);
      //Serial.print("      - non compressed:");
      //Serial.print(nonCompressed);
      //Serial.println("");
      //delay(6000);
      
    for (int i=0;i<CONVOLVED_IMAGE_SIZE;i++) {
    
        
      for (int j=0;j<CONVOLVED_IMAGE_SIZE;j++) {
      // Serial.print(convolved_image[i][j]);
        //Serial.print("  ");
      }
    //Serial.println();
    }
  Serial.println(F("max pooling!"));


  // implement max pooling
float pool_output[POOL_SIZE][POOL_SIZE];

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
      sum += flattened_input[j] * output_weights[j][i]/SCALE;
    }
    output[i] = sigmoid(sum);
  }
  


  //Serial.print("output neurons:");
  for (int i=0;i<N_OUTPUT;i++) {
    Serial.print(output[i]);
    Serial.print(" ");
  }
  
  Serial.print("\nand we have a winner!! :");
  float maximumOutput = output[0];
  int winningClass = 0;
  
  for (int i = 1; i< N_OUTPUT;i++) {
    if (output[i] > maximumOutput) {
        maximumOutput= output[i];
        winningClass = i;
      }
  }
     Serial.println(winningClass);
 
        delay(2000);
}
