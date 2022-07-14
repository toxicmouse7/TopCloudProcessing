#include <iostream>

#include "CloudProcessor.hpp"

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        std::cout << "Provide .ply, radius and output image path" << std::endl;
        exit(-1);
    }

    vtkObject::GlobalWarningDisplayOff();

    CloudProcessor processor(argv[1]);

    processor.passThrough(0, 2.5, "z");
    processor.voxelGrid(0.002, 0.002, 0.002);
    processor.radiusOutlierRemoval(0.025, 10);

    processor.reset(processor.extractMaxCluster());
    processor.moveToBase();

    processor.exportRGBExperimental(argv[3]);

    //processor.exportRGBImage(argv[3], {1920, 1080}, std::stoi(argv[2]));

    return 0;
}
