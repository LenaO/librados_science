#include "rados_data.h"

using namespace JRados;
rados_t JRadosObject::cluster;
int JRadosObject::_is_init = 0;
int JRadosObject::_is_open = false;
 rados_ioctx_t JRadosObject::_io_ctx;


int JRadosObject::open(){
    int err;


    err = rados_ioctx_create(cluster, _pool_name.c_str(), &_io_ctx);
    if (err < 0) {
        err = rados_pool_create(cluster, _pool_name.c_str());
        if (err < 0) {
            std::cerr<<"Could not not create new rados pool "<< _pool_name <<" "<< strerror(-err);
            exit(-1);
        }

        err = rados_ioctx_create(cluster, _pool_name.c_str(), &_io_ctx);

    }
    _is_open = 1;

    return err;

}


