#include <iostream>
#include <string>
#include <rados/librados.hpp>
#include <vector>
#include <type_traits>
#include <map>
#include <algorithm>
#include <sstream>
#define CEPH_CONFIG "/gpfs/homeb/pcp0/pcp0063/.ceph/conf"

#define CEPH_MAX_WRITE  (1<<26)




namespace JRados {
class JRadosObject {
    public:
        explicit JRadosObject(std::string name): _name(name), _pool_name("") 
    {
        if(_is_init == 0)
            connect_cluster();
        ++_is_init;
    }


        explicit JRadosObject(std::string name, std::string pool_name):_name(name),  _pool_name(pool_name)
    {
        if(_is_init == 0)
            connect_cluster();
        ++_is_init;

    }

        bool object_exists(std::string object_name ){
            rados_read_op_t read_op =  rados_create_read_op();
            rados_read_op_assert_exists(read_op);
            int ret =rados_read_op_operate(read_op, _io_ctx, object_name.c_str(), NULL);
            rados_release_read_op(read_op); 
            if(ret!=0) {
                return false;
            }
            return true;

        }

        ~JRadosObject(){
            if((--_is_init)  == 0){
                rados_shutdown(cluster);
                std::cout<<"Shut down Cluster "<<std::endl;
            }
        }

        const std::string &get_name() const{
            return _name;
        }
        bool is_open() const {
            return _is_open;
        }


        void close() {
            if(_is_open)
                rados_ioctx_destroy(_io_ctx);

            _is_open = false;
        }
        int deleteObject(){
            if(_pool_name.empty()){
                std::cerr<<"Cannot delete Img with no name"<<std::endl;
                return -1;
            }
            return 0;
        }


        int open();
        const std::string get_name(){
            return _name;
        }
        int writeAttr(std::string const attr,  std::string& data)
        {
            int ret = 0;

            ret = rados_setxattr(_io_ctx, _name.c_str(), attr.c_str(), data.c_str(), data.length());
            if (ret < 0) {
                std::cerr << "Couldn't set attr! error " << ret << std::endl;
                return EXIT_FAILURE;
            }
            return ret;
        }

        template< typename T>
            int writeAttr( std::string const attr,  const T* data)
            {
                int ret;
                ret = rados_setxattr(_io_ctx, _name.c_str(), attr.c_str(),reinterpret_cast<const char*>(data) ,sizeof(T));
                if (ret < 0) {
                    std::cerr << "Couldn't set attr! error " << ret << std::endl;
                    return EXIT_FAILURE;
                }

                return ret;
            }

        template< typename T>
            int writeAttr( std::string const attr,  const T* data, int n)
            {
                int ret;
                ret = rados_setxattr(_io_ctx, _name.c_str(), attr.c_str(),reinterpret_cast<const char*>(data) , n*sizeof(T));
                if (ret < 0) {
                    std::cerr << "Couldn't set attr! error " << ret << std::endl;
                    return EXIT_FAILURE;
                }

                return ret;
            }



        int getAttr(std::string const attr, std::string &result) const {
            int ret;
            char data[4096];
            memset(data, 0, 4096);

            ret = rados_getxattr(_io_ctx, _name.c_str(), attr.c_str(), data, 4096);
            if (ret < 0) {
                return -1;
            }

            result = std::string(data,ret);

            return ret;
        }

        int getAttr(std::string const attr, char* &result) const {
            int ret;
            char data[4096];
            ret = rados_getxattr(_io_ctx, _name.c_str(), attr.c_str(), data, 4096);
            if (ret < 0) {
                /*If it does not exist, it is not a bad fail*/

                return -1;
            }
            result = new char[ret];
            memcpy(result, data, ret);
            return ret;
        }
        int readData(const char *data, const size_t bytes) {

            size_t bytes_to_read = std::min(size_t{CEPH_MAX_WRITE}, bytes);
            size_t bytes_remaining =  bytes;
            int err = 0;
            char* curr_ptr = const_cast<char*>(data);
            size_t offset = 0;

            std::vector<rados_completion_t> completions(bytes/bytes_to_read + ( bytes%bytes_to_read!=0 ? 1:0));
            std::vector<rados_completion_t>::iterator it = completions.begin();

            while(bytes_remaining>0){
                rados_aio_create_completion(NULL, NULL, NULL, &(*it));
                err += rados_aio_read(_io_ctx, _name.c_str(), *it, curr_ptr, bytes_to_read,0);
                curr_ptr += bytes_to_read;
                offset+=bytes_to_read;
                bytes_remaining-=bytes_to_read;
                bytes_to_read = std::min(size_t{CEPH_MAX_WRITE}, bytes_remaining);
                it++;
            }
            for(auto comp : completions) {
                rados_aio_wait_for_complete(comp);
                if(rados_aio_get_return_value(comp)<0){
                    std::cerr<<"Error in read completion  for layer read"<<std::endl;

                    throw std::runtime_error(strerror(rados_aio_get_return_value(comp) ));
                }
                else err+=rados_aio_get_return_value(comp);
                rados_aio_release(comp);
            }

            return err;

        }

        int writeData(const char* data, const size_t bytes) {

            size_t bytes_to_write = std::min(size_t{CEPH_MAX_WRITE}, bytes);
            size_t bytes_remaining =  bytes;
            int err = 0;
            char* curr_ptr = const_cast<char*>(data);
            size_t offset = 0;
            size_t counter = 0;

            std::vector<rados_completion_t> completions(bytes/bytes_to_write + ( bytes%bytes_to_write!=0 ? 1:0));
            if(!_is_open)
                return -1;
            std::vector<rados_completion_t>::iterator it = completions.begin();

            while(bytes_remaining>0){

                rados_aio_create_completion(NULL, NULL, NULL, &(*it));
                bytes_to_write = std::min(size_t{CEPH_MAX_WRITE}, bytes_remaining);
                rados_aio_write(_io_ctx, _name.c_str(), *it, curr_ptr, bytes_to_write, counter);
                bytes_remaining-=bytes_to_write;
                curr_ptr += bytes_to_write;
                offset += bytes_to_write;
                ++counter;

               it++;
            }
            rados_aio_flush(_io_ctx);
            counter = 0;
            for(auto comp : completions) {

                rados_aio_wait_for_complete(comp);
                if(rados_aio_get_return_value(comp)<0){
                    std::cerr<<"Error in write completion  for layer write"<<std::endl;

                    throw std::runtime_error(strerror(-err));
                }
                rados_aio_release(comp);
            }
            err = rados_notify2(_io_ctx, _name.c_str(), NULL, 0, 100, NULL,NULL);
            return err;

        }

    protected:

        static rados_ioctx_t _io_ctx;

    private:
        static rados_t cluster;
        bool _changed;

        static int _is_init;
        static int _is_open;
        std::string _pool_name;
        std::string _name;
        /*        JcephObject level0;
                  int _build_hierachie();
                  std::map<std::string, attr_shape> shapes;
                  */
        int connect_cluster(){
            int ret;
            char conf_file[] = CEPH_CONFIG;

            ret = rados_create2(&(cluster), "ceph", "client.lena",0);
            if (ret < 0) {
                std::cerr << "Couldn't initialize the cluster handle! error " << ret << std::endl;
                return EXIT_FAILURE;
            }
            if(const char* env_p = std::getenv("CEPH_CONFIG_FILE")){
                strcpy(conf_file, env_p);
            }
            ret = rados_conf_read_file(cluster, conf_file);
            if (ret < 0) {
                rados_shutdown(cluster);
                std::cerr << "Couldn't read the Ceph configuration file! error " << ret << std::endl;
                return EXIT_FAILURE;
            }

            ret = rados_connect(cluster);
            if (ret < 0) {
                rados_shutdown(cluster);

                std::cerr<<"Cannot connect to cluster, maybe you config is wrong\n"<<ret<<std::endl;;
                return ret;
            }


            open();
            return ret;

        }
};

enum jceph_type: uint8_t {
    CHAR = 0,
    SHORT = 1,
    UINT8 = 2,
    INT8 =3,
    UINT16 =4,
    INT16 =5,
    UINT32 = 6,
    INT32 = 7,
    UINT64 = 8,
    INT64 = 9,
    FLOAT = 10,
    DOUBLE =11,
    STRING = 12
};

static const std::array<size_t, 13> jtype_to_size = {1,1,1,1,2,2,4,4,8,8,4,8,1};


struct DataSetShape{
    size_t ndims;
    jceph_type type;
};

class JRadosDataSet: public JRadosObject{
    public:
        JRadosDataSet(std::string name): JRadosObject(name), _shape(nullptr), _dims(nullptr){}
        JRadosDataSet(std::string name, std::string pool_name): JRadosObject(name,pool_name),_shape(nullptr), _dims(nullptr) { }

        JRadosDataSet(std::string name, std::string pool_name, size_t ndims, size_t *dims, jceph_type type): JRadosObject(name,pool_name),_shape(nullptr) {
            _shape = new DataSetShape{ndims, type};
            _dims = new size_t[ndims];
            memcpy(_dims, dims, sizeof(size_t) *ndims);
            writeAttr("shape" ,  _shape);
            writeAttr("dims" , dims, ndims);
        }


        int setShape(size_t ndims, size_t *dims, jceph_type type){
            if(_shape!=nullptr)
                delete _shape;
             if(dims != nullptr)
               delete dims;
            _shape = new DataSetShape{ndims, type};
            _dims = new size_t[ndims];
            memcpy(_dims, dims, sizeof(size_t) *ndims);
            writeAttr("shape" ,  _shape);
            writeAttr("dims" , dims, ndims);
            return 0;

        }
        template< typename T >
            typename std::enable_if<std::is_arithmetic<T>::value, T>::type
            writeLayer(const  size_t ndims, size_t const *dims, const T *data, jceph_type type );

        template< typename T >
            typename std::enable_if<std::is_arithmetic<T>::value, T>::type
            writeLayer(const size_t ndims, const size_t *dims, std::vector<T> const &data, jceph_type type);
        template< typename T >
            typename std::enable_if<std::is_arithmetic<T>::value, T>::type
            readLayer(T *data, size_t &bytes_read);

        template <typename T>
            typename std::enable_if<std::is_arithmetic<T>::value, T>::type
            readBox(size_t start[2], size_t end[2],  T const *data );
        template <typename T>
            typename std::enable_if<std::is_arithmetic<T>::value, T>::type
            writeBox(size_t start_p[2], size_t end_p[2],  T const *data );

        uint8_t getType() {
            if(_shape==nullptr|| _dims==nullptr) {
                int err = _get_shape();
                if(err<0)
                    return err;

            }
            return _shape->type;
        }

        int getShape(DataSetShape* &Shape){
            if(_shape==nullptr|| _dims==nullptr) {
                int err = _get_shape();
                if(err<0)
                    return err;
               }
                Shape = _shape;
                return 0;

            }

        size_t getLayerPoints(){
            if(_shape==nullptr|| _dims==nullptr) {
               int err=_get_shape();
             if(err<0)
                    return err;

            }
            size_t bytes = 1;
            for(size_t i = 0; i<_shape->ndims; ++i) {
                bytes*= _dims[i];
            }
            return bytes;
        }

        size_t getLayerSize(){
            size_t points = getLayerPoints();
            if(points<0)
                return points;
            return points*jtype_to_size[_shape->type];
        }

        size_t  getNdims(){
            if(_shape==nullptr|| _dims==nullptr){
                int err =  _get_shape();
                if(err<0)
                    return err;
            }
            return _shape->ndims;

        }
        int getDims(size_t *dims){
            if(_shape==nullptr|| _dims==nullptr){
               int err =  _get_shape();

                if(err<0)
                    return err;
            }
            memcpy(dims, _dims, _shape->ndims*sizeof(size_t));
            return 0;
        }

    std::vector<std::string>& getAttrs(){
            if(_attr_keys.size() ==0) 
             _getAttrList();

        return _attr_keys;
    }
    private:
        int _get_shape(){
            rados_read_op_t read_op = rados_create_read_op();
            rados_read_op_assert_exists(read_op);
            int ret =rados_read_op_operate(read_op, _io_ctx,get_name().c_str(), NULL);
            if(ret!=0) {
                return -ENOENT;
            }
            ret=getAttr("shape", reinterpret_cast<char*&>(_shape));
            if(ret <0)
                return -1;
            ret = getAttr("dims",  reinterpret_cast<char*&>(_dims));
           return 0;

        }



        int _getAttrList(){
            rados_xattrs_iter_t iter;
            rados_getxattrs( _io_ctx, get_name().c_str(), &iter);
            int r;
            while(1){
                size_t len;
                const char *key, *val;
                r = rados_getxattrs_next(iter, &key, &val, &len);
                if (r) {
                    std::cerr<<"rados_getxattrs( "<<get_name()<<"): rados_getxattrs_next "
                        "returned error "<<r<<std::endl;
                    exit(1);
                }
                if (!key)
                    break;


                std::string key_string(key);
                if(key_string=="shape"){
                    _shape = (DataSetShape*)malloc(len);
                    memcpy(_shape, val, len);
                }
                else if(key_string == "dims") {
                    _dims = (size_t*)malloc(len);
                    memcpy(_dims, val, len);
                }
                else {
                    _attr_keys.push_back(key);


                }

            }
            return 0;
        }



        DataSetShape *_shape;
        std::vector<std::string> _attr_keys;
        size_t *_dims;


};


template< typename T >
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
JRadosDataSet::writeLayer(const size_t ndims, const size_t *dims, const T *data, jceph_type type) {

    size_t bytes = 1;

    if(_shape != NULL)
        delete _shape;
    _shape=new DataSetShape{ndims, type};


     if(_dims != NULL)
        delete _dims;

    _dims = new size_t[ndims];
    for(size_t i = 0; i<ndims; ++i) {
        _dims[i]=dims[i];
         bytes*= dims[i];
    }

    bytes*=sizeof(T);

    int err;
    size_t offset = 0;
    writeAttr("shape" ,  _shape);
    writeAttr("dims" , dims, ndims);

    writeData( reinterpret_cast<const char*>(data), bytes);
    return 0;

}
template< typename T >
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
JRadosDataSet::writeLayer(const size_t ndims, const  size_t *dims, std::vector<T> const &data, jceph_type type) {

    T const * data_ptr = data.data();

    return writeLayer(ndims, dims , data_ptr, type);

}

template< typename T >
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
JRadosDataSet::readLayer( T *data, size_t &bytes_read) {

    const char *pointer = (reinterpret_cast<const char*>(data));
    int type = getType();
    if(type<0)
        return type;
    size_t elemsize =  jtype_to_size[type];
    size_t layerPoints =  getLayerPoints();

    assert(sizeof(T) == elemsize);

    const size_t bytes = std::min( layerPoints*sizeof(T), bytes_read);

    int ret = readData(pointer, bytes);

    bytes_read = bytes;
    return 0;

}

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
JRadosDataSet::readBox(size_t start_p[2], size_t end_p[2],  T const *data ){

    size_t offset;
    const size_t start_x = start_p[0];
    const size_t start_y = start_p[1];
    const size_t w = end_p[0] - start_p[0];
    const size_t h = end_p[1] - start_p[1];
    char* curr_ptr =  const_cast<char*>(reinterpret_cast<const char*>(data));
    //rados_completion_t comp;
    if(_shape==nullptr|| _dims==nullptr) {
       int  err= _get_shape();
       if(err<0)
            return err;
    }
    assert(start_x+w <= _dims[0]);
    assert(start_y+h <= _dims[1]);
    assert(sizeof(T) == jtype_to_size[_shape->type]);
 //   std::vector<int> prval(w);
   // std::vector<size_t> bytes_read(w);
    std::vector<int> prval(h);
    std::vector<size_t> bytes_read(h);

    //const size_t bytes_to_read = sizeof(T) * (h);
    const size_t bytes_to_read = sizeof(T) * (w);
    int r = w/512 + (w%512!=0 ? 1:0);
    std::vector<rados_completion_t> completions(r);
    std::vector<rados_completion_t>::iterator it = completions.begin();
    size_t w_left = h;
    //size_t w_left = w;
    size_t start = start_y;
    //size_t start = start_x;
    size_t i =0;


    for (int k=0; k<r; ++it, ++k) {
        rados_aio_create_completion(NULL, NULL, NULL, &(*it));
        rados_read_op_t op = rados_create_read_op();
        for(size_t y=start; y<start+std::min(w_left,size_t(512)); y++,i++) {

            offset = (start_x+ _dims[0] * y)*sizeof(T);
            //offset = (start_y+ _dims[1] * y)*sizeof(T);
            rados_read_op_read(op, offset, bytes_to_read, curr_ptr, &bytes_read[i], &prval[i]);
            curr_ptr += bytes_to_read;
        }
        start+=512;
        w_left-=512;
        rados_aio_read_op_operate(op,_io_ctx, *it,  get_name().c_str(), 0);

        rados_release_read_op(op);
    }


    //    rados_aio_flush(_io_ctx);
    for(auto comp : completions) {
        rados_aio_wait_for_complete(comp);
        rados_aio_release(comp);
    }/*

        if(rados_aio_get_return_value(comp)<0){
        std::cerr<<"Error in write completion  for layer write"<<std::endl;
        throw std::runtime_error(strerror(-err));
        }

        std::cout<<"JHE2"<<std::endl;

        }*/
//  }
return 0;


}


template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
JRadosDataSet::writeBox(size_t start_p[2], size_t end_p[2],  T const *data ){

    size_t offset;
    const size_t start_x = start_p[0];
    const size_t start_y = start_p[1];
    const size_t w = end_p[0] - start_p[0];
    const size_t h = end_p[1] - start_p[1];
    char* curr_ptr =  const_cast<char*>(reinterpret_cast<const char*>(data));
    //rados_completion_t comp;
    if(_shape==nullptr|| _dims==nullptr){
        int err = _get_shape();
        if(err<0)
            return err;

    }
    assert(start_x+w <= _dims[0]);
    assert(start_y+h <= _dims[1]);
    assert(sizeof(T) == jtype_to_size[_shape->type]);


    //const size_t bytes_to_write = sizeof(T) * (h);
    const size_t bytes_to_write = sizeof(T) * (w);
    int r = w/512 + (w%512!=0 ? 1:0);
    std::vector<rados_completion_t> completions(r);
    std::vector<rados_completion_t>::iterator it = completions.begin();
    size_t w_left = h;
    //size_t w_left = w;
    size_t start = start_y;
    //size_t start = start_x;
    size_t i =0;


    for (int k=0; k<r; ++it, ++k) {
        rados_aio_create_completion(NULL, NULL, NULL, &(*it));
        rados_write_op_t op = rados_create_write_op();
        for(size_t y=start; y<start+std::min(w_left,size_t(512)); y++,i++) {

            offset = (start_x+ _dims[0] * y)*sizeof(T);
            //offset = (start_y+ _dims[1] * y)*sizeof(T);
            rados_write_op_write(op, curr_ptr , bytes_to_write,offset);
            curr_ptr += bytes_to_write;
        }
        start+=512;
        w_left-=512;
        rados_aio_write_op_operate(op,_io_ctx, *it,  get_name().c_str(), NULL, 0);
        rados_release_write_op(op);
    }
    //    rados_aio_flush(_io_ctx);
    for(auto comp : completions) {
        rados_aio_wait_for_complete(comp);

        if(rados_aio_get_return_value(comp)<0){
            std::cerr<<"Error in write completion  for layer write"<<std::endl;
            throw std::runtime_error(strerror(-1));
        }
        rados_aio_release(comp);

    }
//  }
return 0;


}



}
