#include <iostream>

#include "CloudProcessor.hpp"

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Provide .ply" << std::endl;
        exit(-1);
    }

    vtkObject::GlobalWarningDisplayOff();

    CloudProcessor processor(argv[1]);

    processor.passThrough(0, 2.5, "z");
    processor.voxelGrid(0.01, 0.01, 0.01);
    processor.radiusOutlierRemoval(0.025, 10);

    processor.reset(processor.extractMaxCluster());
    processor.moveToBase();

    processor.exportRGBImage("1.bmp", {1920, 1080}, 4);

    return 0;
}
