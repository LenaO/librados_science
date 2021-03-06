#include <string>
#include <iostream>
#include <rados_data.h>
#include <vector>
#include<algorithm>
int main() {

    JRados::JRadosDataSet DataSet("lena", "lena_testing");
    size_t shape[2] = {32L,32L};
    uint8_t data[32*32];
    size_t tmp = 32*32;
    DataSet.readLayer(data, tmp);
    for(int i =0; i<32; i++) {

        for(int j =0; j<32; j++) {
            std::cout<<(int)data[i*32+j]<<" ,";
        }
        std::cout<<std::endl;
    }



    uint8_t boxData[12*13];
    size_t start[2] = {0,3};
    size_t end[2] ={0+12,3+13};
    DataSet.readBox( start,end, boxData);
    for(int i =0; i<13; i++) {

        for(int j =0; j<12; j++) {
            std::cout<<(int)boxData[i*12+j]<<" ,";
        }
        std::cout<<std::endl;
    }
}

