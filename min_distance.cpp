#include <iostream>
#include <string>
#include <vector>
#include <cmath>

struct coord
{
    int x, y;
};

int main()
{
    // Input matrix
    int DataMat[5][5] = { {0, 0, 0, 0, 1},
                          {0, 1, 0, 0, 0},
                          {0, 1, 1, 0, 0},
                          {0, 0, 0, 0, 0},
                          {0, 0, 0, 0, 0}
                        };
            
    std::vector<coord> TargetVec;
                        
    // First loop, find all targets
    for(int i=0; i<5; i++)
    {
        for(int j=0; j<5; j++)
        {
            if(DataMat[i][j])
            {
                coord pt;
                pt.x = j;
                pt.y = i;
                
                TargetVec.push_back(pt);
            }
        }
    }
    
    int NumOfTarget = TargetVec.size();
    
    std::cout << "Found "  << NumOfTarget << " target points!" << std::endl;
    
    // Second loop, calculate distance for all points
    int OutputMat[5][5];
    for(int i=0; i<5; i++)
    {
        for(int j=0; j<5; j++)
        {
            // Use a very large number for initial distance
            int MinDist = 10000000;
            
            // Calculate distance between this point to all targets
            for(int k=0; k<NumOfTarget; k++)
            {
                int dist = std::abs(j-TargetVec[k].x) + std::abs(i-TargetVec[k].y);
                MinDist = std::min(dist, MinDist);
            }
            
            OutputMat[i][j] = MinDist;
        }
    }
    
    // Print results
    for(int i=0; i<5; i++)
    {
        for(int j=0; j<5; j++)
        {
            std::cout << OutputMat[i][j] << " ";
        }
        std::cout << std::endl;
    }
    
    return 0;
 
}
