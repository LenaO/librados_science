#include <string>
#include <iostream>
#include <rados_data.h>
#include <vector>
#include<algorithm>
int main() {

    JRados::JRadosDataSet DataSet("lena", "lena_testing");

    size_t shape[2] = {32L,32L};
    uint8_t data[32*32];
     for(int i =0; i<32; i++)

        for(int j =0; j<32; j++) {
            data[i*32+j] =3;
        }
 

    DataSet.writeLayer(2, shape, data, JRados::UINT8 );
    uint8_t boxData[4*13];
    size_t start[2] = {3,10};
    size_t end[2] ={3+4,10+13};

    for(int i =0; i<13; i++) 

        for(int j =0; j<4; j++) {
            boxData[i*4+j] =4;
        }
 

    DataSet.writeBox( start,end, boxData);

}

