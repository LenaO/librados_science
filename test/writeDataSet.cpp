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



}

