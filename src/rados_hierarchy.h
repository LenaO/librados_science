
#include "rados_data.h"

#include <conduit/conduit.hpp>



namespace JRados {

    static uint8_t
conduit_dtype_to_jrados_dtype(const conduit::DataType &dt)
{
    uint8_t res = 0;

    conduit::index_t dt_id = dt.id();
    switch(dt_id)
    {
        case conduit::DataType::INT8_ID:  res = INT8;  break;
        case conduit::DataType::INT16_ID: res = INT16 ; break;
        case conduit::DataType::INT32_ID: res = INT32; break;
        case conduit::DataType::INT64_ID: res = INT64; break;

        case conduit::DataType::UINT8_ID:  res = UINT8;  break;
        case conduit::DataType::UINT16_ID: res = UINT16; break;
        case conduit::DataType::UINT32_ID: res = UINT32; break;
        case conduit::DataType::UINT64_ID: res = UINT64; break;

        case conduit::DataType::FLOAT32_ID: res = FLOAT; break;
        case conduit::DataType::FLOAT64_ID: res = DOUBLE; break;

        case conduit::DataType::CHAR8_STR_ID: res =STRING; break;

    }
    return res;
}


//-----------------------------------------------------------------------------
static 
    conduit::DataType
jrados_dtype_to_conduit_dtype(uint8_t dt, size_t num_elems)
{
    conduit::DataType res =conduit::DataType::empty();
    switch (dt)
    {
        case INT8: res = conduit::DataType::int8(num_elems); break;
        case INT16: res = conduit::DataType::int16(num_elems); break;
        case INT32: res = conduit::DataType::int32(num_elems); break;
        case INT64: res = conduit::DataType::int64(num_elems); break;
        case UINT8: res = conduit::DataType::uint8(num_elems); break;
        case UINT16: res = conduit::DataType::uint16(num_elems); break;
        case UINT32: res = conduit::DataType::uint32(num_elems); break;
        case UINT64: res = conduit::DataType::uint64(num_elems); break;

        case FLOAT: res = conduit::DataType::float32(num_elems); break;
        case DOUBLE: res = conduit::DataType::float64(num_elems); break;
        case STRING: res  = conduit::DataType::char8_str(num_elems); break;



    }
    return res;
}

class JRadosCSchema: public JRadosObject{
    public:
        JRadosCSchema(std::string name): JRadosObject(name), _exist(-1){}
        JRadosCSchema(std::string name, std::string pool_name): JRadosObject(name,pool_name), _exist(-1){
        }
        int writeSchema(const std::string& json_string) {
            int ret;
            const std::string name = get_name();
            size_t size = json_string.length();

            rados_write_op_t write_op =  rados_create_write_op();
            rados_write_op_write_full(write_op, json_string.c_str(), size);
            rados_write_op_setxattr(write_op,  "type", "Schema", 6);
            rados_write_op_setxattr(write_op,"Len",  reinterpret_cast<const char*>(&size), sizeof(size_t));
           ret = rados_write_op_operate(write_op, _io_ctx, name.c_str(), NULL, 0);

            if(ret<0) {
                std::cerr<<"Error writing Schema "<<name<<" to ceph cluster "<<std::endl;
            }
            rados_release_write_op(write_op);
            _exist =1;

            return ret;
        }
    int writeSchema(conduit::Schema& node) {
         std::string json_string = node.to_json(true,2);
         return writeSchema(json_string);
    }

        std::string getSchema() {
            int ret;
            const std::string name = get_name();
            char *data;
            size_t string_size;

            rados_read_op_t read_op =  rados_create_read_op();
            rados_read_op_assert_exists(read_op);
            rados_read_op_cmpxattr(read_op,"type", LIBRADOS_CMPXATTR_OP_EQ , "Schema", 6);


            ret = rados_read_op_operate(read_op, _io_ctx, name.c_str(), 0);
            if(ret < 0) {
                std::cerr<<"Error: Schema "<<name<<" does not exist or another object with this name exist "<<std::endl;
                return "";
            }

            rados_release_read_op(read_op);
            _exist = 1;

            rados_getxattr(_io_ctx,name.c_str(), "Len", reinterpret_cast<char*>(&string_size), sizeof(size_t));
            data = new char[string_size];
            memset (data, 0, string_size);
            ret =  rados_read(_io_ctx, name.c_str(), data, string_size,0);

            if(ret<0) {
                std::cerr<<"Error reading Schema "<<name<<" to ceph cluster "<<std::endl;
                return "";
            }

            std::string json_string(data, string_size);

            delete data;
            return json_string;
        }

        bool exist() {
            if(_exist == -1) {
                int ret;
                const std::string name = get_name();
                rados_read_op_t read_op =  rados_create_read_op();
                rados_read_op_assert_exists(read_op);
                rados_read_op_cmpxattr(read_op,"type", LIBRADOS_CMPXATTR_OP_EQ , "Schema", 6);
                ret =  rados_read_op_operate(read_op, _io_ctx, name.c_str(), 0);
                if(ret < 0) {
                    _exist = 0;
                }
                else _exist = 1;

                rados_release_read_op(read_op);
            }

            return _exist == 1? true:false;
    }


    private:
        int _exist;
};

enum mode{
    OPEN = 0,
    CREATE =1,
    OPEN_OR_CREATE =2,
};



class JRadosHObject {

    public:
        JRadosHObject(std::string name, std::string pool, int mode);


        template< typename T>
            typename std::enable_if<std::is_arithmetic<T>::value, T>::type
            writeDataSet(size_t ndims, size_t const *dims, T *data, jceph_type type);

        template< typename T>
            typename std::enable_if<std::is_arithmetic<T>::value, T>::type
            readDataSet(T *data, size_t& max_points);

        template< typename T>
            typename std::enable_if<std::is_arithmetic<T>::value, T>::type
            readDataBox(T *data, size_t start[2], size_t end[2]);

         template< typename T>
            typename std::enable_if<std::is_arithmetic<T>::value, T>::type
            writeDataBox(T *data, size_t start[2], size_t end[2]);



        int createDataSet( size_t ndims, size_t *dims, jceph_type type){

        if(_node->dtype().is_empty()){
            std::cerr<<"Can not add dataset for "<<_name<<" node is not a leaf  or Datser already exist\n"<<std::endl;
            return -1;
         }

         if(_data_set!= nullptr){
                std::cerr<<"Dataset "<<_name <<"exist"<<std::endl;
                return -1;
         }
             std::string object_name =_root->_name+'/'+_node->path();

             _data_set=new JRadosDataSet(object_name, _pool_name, ndims, dims, type);

            return  0;
        }
      int createDataSet( std::string name, size_t ndims, size_t *dims, jceph_type type){

           JRadosHObject child = (*this)[name];
           return child.createDataSet(ndims, dims, type);


        }

        size_t getDataSetBytes();
        size_t getDataSetPoints();
        int addAttr (std::string name, std::vector<std::string>& data );

        template< typename T>
            typename std::enable_if<std::is_arithmetic<T>::value, T>::type
            addAttr(std::string name, size_t ndims, size_t const *dims, T *data, jceph_type type );

        template< typename T>
            typename std::enable_if<std::is_arithmetic<T>::value, T>::type
            getAttr(std::string name, T *data);
        std::vector<std::string> getAttrs();



        void print() const {
            _node->print();
        }
         bool is_changed() const{
            return _changed;
        }
        void set_changed(){
            _changed = true;
        }
        void set_saved(){
            _changed = false;

        }

/*Operators*/
        JRadosHObject& operator[](const std::string child) {

            if(_is_leaf) {
                std::cerr<<"Node is leaf, cannnot add new group"<<std::endl;
                return *this;

            }

            std::map<std::string,JRadosHObject*>::iterator it;
            it = this->children.find(child);
            if(it != children.end())
                return *it->second;
            JRadosHObject* new_obj = new JRadosHObject(child, _pool_name);
            children[child] = new_obj;
            new_obj->_node = &((*_node)[child]);
            new_obj->_root = _root;
            _changed-true;
            _root->_changed=true;
            return *new_obj;
        }

        int save() ;

        JRadosHObject operator[] (const std::string child) const {
            return (*this)[child];
        }


    private:

        JRadosHObject(std::string name, std::string pool): 
            _name(name), _pool_name(pool), _schema(name, pool),_data_set(nullptr), _is_root(false), _is_leaf(false){}
        JRadosCSchema _schema;
        conduit::Schema *_node;
        std::string _name;
        std::string _pool_name;
        bool _is_leaf;
        JRadosDataSet* _data_set;
        std::map<std::string, JRadosHObject*> children;
        std::string _root_name;
        JRadosHObject  *_root;
        int handleNullDataSet();
        bool _is_root;
        bool _changed;
        int createDataSet();


};

template< typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
JRadosHObject::writeDataSet(size_t ndims, size_t const *dims, T *data, jceph_type type ) {

    if(_node->number_of_children() != 0){
        std::cerr<<"Can not write DataSet, node is not a leaf \n"<<std::endl;
        return -1;
    }
    std::string object_name =_root->_name+'/'+_node->path();
    size_t bytes = 1;

    for(size_t i = 0; i<ndims; ++i) {
        bytes*= dims[i];
    }


    _node->set(jrados_dtype_to_conduit_dtype(type, bytes));
    if(_data_set==nullptr) {
        _data_set=new JRadosDataSet(object_name, _pool_name);
    }
    else{
        if(object_name != _data_set->get_name()){
            std::cerr<<"something went wrong "<<std::endl;
            return -1;
        }
    }

    _data_set->writeLayer(ndims, dims, data, type);
    _changed=false; /*(already on disk)*/
    _root->_changed=true;

    _is_leaf = true;
}


inline int JRadosHObject::handleNullDataSet(){

    std::string object_name =_root->_name+'/'+_node->path();
    _data_set=new JRadosDataSet(object_name, _pool_name);
    return 0;

}

template< typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
JRadosHObject::readDataSet(T *data, size_t &max_points) {

    if(_node->dtype().is_object()){
        std::cerr<<"Can not Read "<<_name<<" node is not a leaf \n"<<std::endl;
        return -1;
    }


    if(_data_set==nullptr) {
       handleNullDataSet(); 

    }
    size_t read_bytes = max_points*sizeof(T);
    _data_set->getAttrs();
    _data_set->readLayer(data, read_bytes);
    max_points = read_bytes/sizeof(T);
    _changed=false; /*(already on disk)*/
    _root->_changed=true;

    _is_leaf = true;
}

template< typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
JRadosHObject::readDataBox(T *data, size_t start[2], size_t end[2]) {

    if(_node->dtype().is_object()){
        std::cerr<<"Can not Read "<<_name<<" node is not a leaf \n"<<std::endl;
        return -1;
    }
    /*We are a leaf, for the right naming, we have to find the root - node */

    std::string object_name =_root->_name+'/'+_node->path();


    if(_data_set==nullptr) {

        handleNullDataSet(); 
    }
    _data_set->readBox(start, end, data);
}

template< typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
JRadosHObject::writeDataBox(T *data, size_t start[2], size_t end[2]) {

    if(_node->dtype().is_object()){
        std::cerr<<"Can not Write "<<_name<<" node is not a leaf \n"<<std::endl;
        return -1;
    }

    /*We are a leaf, for the right naming, we have to find the root - node */

    std::string object_name =_root->_name+'/'+_node->path();
    // std::cout<<"read the data_set "<<object_name<<std::endl;

    if(_data_set==nullptr) {

        handleNullDataSet(); 
    }

    _data_set->writeBox(start, end, data);
}






}
