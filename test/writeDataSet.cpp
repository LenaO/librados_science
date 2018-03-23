#include <string>
#include <iostream>
#include <rados_data.h>
#include <vector>
#include<algorithm>
int main() {

    JRados::JRadosDataSet DataSet("lena", "lena_testing");

    size_t shape[2] = {32L,32L};
    uint8_t data[32*32];
    DataSet.writeLayer(2, shape, data, JRados::UINT8 );
    uint8_t boxData[12*13];
    size_t start[2] = {0,3};
    size_t end[2] ={0+12,3+13};

    for(int i =0; i<12; i++) 

        for(int j =0; j<13; j++) {
            boxData[i*13+j] =1;
        }
 

    DataSet.writeBox( start,end, boxData);

}

