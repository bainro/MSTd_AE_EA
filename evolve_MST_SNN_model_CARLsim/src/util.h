#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

float** loadData(string file, int numRow, int numCol);

int randGenerator (int i);

void shuffleTrials(int numTrials, int numTrain, int *trainTrials);

float calcRMS(float** x, float** y, int numRow, int numCol);
  
float calcCorr(float** X, float** Y, int numRow, int numCol);
