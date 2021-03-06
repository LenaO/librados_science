#include <rados_cache.h>

int main() {

    JRados::RadosCacheDict DataCache("image_cache");
    size_t shape[2] = {32L,32L};
    uint8_t data[32*32];
    for (int i =0; i<32; i++){
        for(int j=0; j<32; j++){
             data[i*32+j]= i+32*j;

        }

    }
    DataCache.writeData("54", data,2, shape);
    std::cout<<"Does 9 exist in cache? " <<  DataCache.existData("9")<<std::endl;
    std::cout<<"Does 32 exist in cache? " <<  DataCache.existData("32")<<std::endl;
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

    DataCache["32"].writeData("sda", data, 2, shape);
    DataCache["32"].readData("sda",  recv_data, ndims, new_shape);

    for (int i =0; i<new_shape[0]; i++){
        for(int j=0; j<new_shape[1]; j++){
             std::cout<<" "<<(int)recv_data[i*new_shape[0]+j];

        }
        std::cout<<std::endl;
    }


}

