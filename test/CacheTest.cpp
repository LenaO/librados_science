#include <rados_cache.h>

int main() {

    JRados::JRadosCache DataCache("image_cache");
    size_t shape[2] = {32L,32L};
    uint8_t data[32*32];
    for (int i =0; i<32; i++){
        for(int j=0; j<32; j++){
             data[i*32+j]= i+32*j;

        }

    }
    DataCache.writeData("54", data,2, shape);
    std::cout<<"Does 9 exist in cache? " <<  DataCache.exist("9")<<std::endl;
    std::cout<<"Does 32 exist in cache? " <<  DataCache.exist("32")<<std::endl;
    uint8_t *recv_data = NULL;
    size_t ndims;
    size_t* new_shape;
    DataCache.readData("32", recv_data, ndims, new_shape);
   std::cout<<recv_data<<"new_shape "<<new_shape[0]<<std::endl;
    for (int i =0; i<new_shape[0]; i++){
        for(int j=0; j<new_shape[1]; j++){
             std::cout<<" "<<(int)recv_data[i*new_shape[0]+j];

        }
        std::cout<<std::endl;
    }
    float* recv3;
    size_t* float_shape;

    DataCache.readData("test2", recv3, ndims, float_shape);

    for (int i =0; i<float_shape[0]; i++){
        std::cout<<recv3[i]<<";";
    }
        std::cout<<std::endl;
}

