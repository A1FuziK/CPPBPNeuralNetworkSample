#include "BPNeuralNetwork.h"

int main(int argc, char* argv[])
{
    double inputData[][4] = {
        {0,0,0,0},
        {0,0,0,1},
        {0,0,1,0},
        {0,0,1,1},
        {0,1,0,0},
        {0,1,0,1},
        {0,1,1,0},
        {0,1,1,1},
        {1,0,0,0},
        {1,0,0,1},
        {1,0,1,0},
        {1,0,1,1},
        {1,1,0,0},
        {1,1,0,1},
        {1,1,1,0},
        {1,1,1,1}
        };

    double outData[][16] = {
        {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1}
        };


	int layers = 4, layerSpec[4] = {4,64,64,16};

	double learnRate = 0.1, momentum = 0.5, errorThreshHold = 0.00001, dTemp = 0.0;

	BPNeuralNetwork* bp = new BPNeuralNetwork(layers, layerSpec, learnRate, momentum);


	std::cout << "Let's train the network..." << "\n";

	long maxIterations = 2000000;
	long iteration;
	bool bTrained = false;

	long trainCycles = 0;
	double comulatedDistance = 0.0;

	for (iteration = 0; iteration < maxIterations && bTrained == false; iteration++)
	{
		bTrained = true;
		for (int iBuffLearn = 0; iBuffLearn < 16; iBuffLearn++)
		{
			for (int iInnerLeanCycle = 0; iInnerLeanCycle < 10; iInnerLeanCycle++)
			{
				bp->backPropagate((double*)&inputData[iBuffLearn], (double*)&outData[iBuffLearn]);
				dTemp = bp->meanSquareError((double*)&outData[iBuffLearn]);

				comulatedDistance += dTemp;
				trainCycles++;
				if (dTemp <= (comulatedDistance / trainCycles) || dTemp < errorThreshHold)
					break;
			}

			if (dTemp < errorThreshHold)
			{
				
			}
			else
			{
				bTrained = false;
			}
		}

		if (iteration % (maxIterations / 100) == 0)
		{
			std::cout << "Still training... last meanSquareError: " << dTemp
				<< " average meanSquareError: " << (comulatedDistance / trainCycles) << "\n";
		}

	}

	std::cout << trainCycles << " train cycles completed... in " << iteration << " main iterations... "
		<< " average meanSquareError: " << (comulatedDistance / trainCycles)
		<< " last meanSquareError: " << dTemp << "\n";

	//Save weights to file...
	//bp->saveNet("minta.net");

	std::string stringWeights = bp->getNetWeights();
	//std::cout << "Network weights:" << "\n" << stringWeights << "\n";


	delete bp;
	bp = nullptr;

	std::cout << "Test the trained - and reinitialized network..." << "\n";
	
	bp = new BPNeuralNetwork(layers, layerSpec, learnRate, momentum);
	bp->setNetWeights(stringWeights);

	//Load weights from file...
	//bp->loadNet("minta.net");


	for (int iBuffLearn = 0; iBuffLearn < 16; iBuffLearn++)
	{
		bp->feedForward((double*) & inputData[iBuffLearn]);

		std::cout << iBuffLearn << "\t";

		for (int iOut = 0; iOut < 16; iOut++)
		{
			double dOut = bp->outValue(iOut);

			if (dOut >= 1.1)
			{
				std::cout << "H ";
			}
			else if (dOut >= 0.8)
			{
				std::cout << "1 ";
			}
			else if (dOut <= 0.2)
			{
				std::cout << "0 ";
			}
			else if (dOut <= 0)
			{
				std::cout << "L ";
			}
			else
			{
				std::cout << "M ";
			}
		}
		std::cout << "\n";
	}


	delete bp;
	bp = nullptr;
    
    return 0;
}
