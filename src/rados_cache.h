#ifndef _rados_cache_h_
#define _rados_cache_h_

#include <iostream>
#include <string>
#include <rados/librados.hpp>
#include <vector>
#include <type_traits>
#include <map>
#include <algorithm>
#include <sstream>
#include "rados_data.h"

namespace JRados {

    class JRadosCache: public JRadosObject{
        public:
            JRadosCache(std::string name):_name(name), JRadosObject(name,name)  {


            } 
          /*  ~JRadosCache(){
            for (auto it:


            } 

*/
            bool exist(std::string key){

                std::map<std::string,JRadosDataSet*>::iterator it;
                   it  =  _cache_data.find(key);
             if(it != _cache_data.end()){
                   return true;
                }
                else{
                    if(object_exists(key)){
                        _cache_data[key]= new JRadosDataSet(key, _name);
                        return true; 
                    }
                    else
                        return false;
                }
            }
            template< typename T >
                typename std::enable_if<std::is_arithmetic<T>::value, T>::type
                 addAttr(const std::string obj_name, const std::string attr_name, T* value){

                    return  _set_for_read(obj_name)->writeAttr(attr_name, value);

                }

            template< typename T >
                typename std::enable_if<std::is_arithmetic<T>::value, T>::type
                 getAttr(const std::string obj_name, const std::string attr_name, T* value){
                    if(!exist(obj_name))
                        return -1;
                    char *result;
                      _cache_data[name]->getAttr(attr_name, result);
                    return (T*) result;

                }

            template< typename T >
            typename std::enable_if<std::is_arithmetic<T>::value, T>::type
            readData(const std::string name,  T* &data,  size_t &Ndims, size_t &dims){
                return readData(name, data, Ndims, dims);
            }
            size_t getShape(std::string name, int &Ndims, size_t* &dims, int& type) {
                DataSetShape shape;
                size_t points =  _getFieldData(name, shape, dims);
                if(points<= 0)
                    return points;
                type = shape.type;
                Ndims = (int) shape.ndims;
                return  points;
            }

            template< typename T >
                typename std::enable_if<std::is_arithmetic<T>::value, T>::type
                readDataRaw(const std::string name,  T* data,  size_t points){
                    if(!exist(name))
                        return -1;
                    JRadosDataSet *Set = _cache_data[name];
                    size_t bytes = points*sizeof(T);
                    int err = Set->readLayer(data, bytes);
                    if(err<0)
                        printf("Error in reading char field\n");

                    return bytes/sizeof(T);
                }
            int  readData(const std::string name, char * &data, size_t &Ndims, size_t* &dims ){
                DataSetShape shape;
                size_t points =  _getFieldData(name, shape, dims);

                size_t bytes = points*sizeof(char);
                if(points<= 0)
                    return points;
                data = new char[points];
                if(shape.type !=CHAR){
                    printf("Type missmacht, expected char, got ");
                    return -1;

                }
                Ndims=shape.ndims;
                JRadosDataSet *Set = _cache_data[name];
                int err = Set->readLayer(data, bytes);
                if(err<0)
                    printf("Error in reading char field\n");
                return bytes/sizeof(char);
            }

            int  readData(const std::string name, int8_t *&data, size_t& Ndims, size_t* &dims ){
                DataSetShape shape;
                size_t points =  _getFieldData(name, shape, dims);
                if(points<= 0)
                    return points;

                size_t bytes = points*sizeof(int8_t);
                data = new int8_t[points];
                if(shape.type !=INT8){
                    printf("Type missmacht, expected INT8, got ");
                    return -1;

                }
                JRadosDataSet *Set = _cache_data[name];
                Ndims=shape.ndims;
                int err = Set->readLayer(data, bytes);
                if(err<0)
                    printf("Error in reading char field\n");
                return bytes/sizeof(int8_t);
            }

            int  readData(const std::string name, uint8_t *&data, size_t& Ndims, size_t* &dims ){
                DataSetShape shape;
                size_t points =  _getFieldData(name, shape, dims);
                if(points<= 0)
                    return points;
                data = new uint8_t[points];

                size_t bytes = points*sizeof(uint8_t);
                if(shape.type !=UINT8){
                    printf("Type missmacht, expected UIN8, got ");
                    return -1;

                }
                JRadosDataSet *Set = _cache_data[name];
                Ndims=shape.ndims;
                int err = Set->readLayer(data, bytes);
                if(err<0)
                    printf("Error in reading char field\n");
                return bytes/sizeof(uint8_t);
            }

            int  readData(const std::string name, int16_t *&data, size_t &Ndims, size_t* &dims ){
                DataSetShape shape;
                size_t points =  _getFieldData(name, shape, dims);
                if(points<= 0)
                    return points;

                size_t bytes = points*sizeof(int16_t);
                data = new int16_t[points];
                if(shape.type !=INT16){
                    printf("Type missmacht, expected INT16, got ");
                    return -1;

                }
                JRadosDataSet *Set = _cache_data[name];
                Ndims=shape.ndims;
                int err = Set->readLayer(data, bytes);
                if(err<0)
                    printf("Error in reading char field\n");
                return bytes/sizeof(int16_t);
            }

            int  readData(const std::string name, uint16_t *&data, size_t &Ndims, size_t* &dims ){
                DataSetShape shape;
                size_t points =  _getFieldData(name, shape, dims);
                if(points<= 0)
                    return points;

                size_t bytes = points*sizeof(uint8_t);
                data = new uint16_t[points];
                if(shape.type !=UINT16){
                    printf("Type missmacht, expected UINT16, got ");
                    return -1;

                }
                JRadosDataSet *Set = _cache_data[name];
                Ndims=shape.ndims;
                int err = Set->readLayer(data, bytes);
                if(err<0)
                    printf("Error in reading char field\n");
                return bytes/sizeof(uint16_t);
            }

            int  readData(const std::string name, int32_t *&data,  size_t& Ndims, size_t* &dims ){
                DataSetShape shape;
                size_t points =  _getFieldData(name, shape, dims);
                if(points<= 0)
                    return points;
                data = new int32_t[points];

                size_t bytes = points*sizeof(int32_t);
                if(shape.type !=INT32){
                    printf("Type missmacht, expected INT32, got ");
                    return -1;

                }
                JRadosDataSet *Set = _cache_data[name];
                Ndims=shape.ndims;
                int err = Set->readLayer(data, bytes);
                if(err<0)
                    printf("Error in reading char field\n");
                return bytes/sizeof(int32_t);
            }

            int  readData(const std::string name, uint32_t *&data, size_t& Ndims, size_t* &dims ){
                DataSetShape shape;
                size_t points =  _getFieldData(name, shape, dims);
                if(points<= 0)
                    return points;
                data = new uint32_t[points];
                size_t bytes = points*sizeof(uint32_t);
                if(shape.type !=UINT32){
                    printf("Type missmacht, expected UINT32, got ");
                    return -1;

                }
                JRadosDataSet *Set = _cache_data[name];
                Ndims=shape.ndims;
                int err = Set->readLayer(data, bytes);
                if(err<0)
                    printf("Error in reading char field\n");
                return bytes/sizeof(uint32_t);
            }

            int  readData(const std::string name, int64_t *&data, size_t &Ndims, size_t* &dims ){
                DataSetShape shape;
                size_t points =  _getFieldData(name, shape, dims);
                if(points<= 0)
                    return points;
                size_t bytes = points*sizeof(int64_t);
                data = new int64_t[points];
                if(shape.type !=INT64){
                    printf("Type missmacht, expected INT64, got ");
                    return -1;

                }
                JRadosDataSet *Set = _cache_data[name];
                Ndims=shape.ndims;
                int err = Set->readLayer(data, bytes);
                if(err<0)
                    printf("Error in reading char field\n");
                return bytes/sizeof(int64_t);
            }

            int  readData(const std::string name, uint64_t *&data, size_t& Ndims, size_t* &dims ){
                DataSetShape shape;
                size_t points =  _getFieldData(name, shape, dims);
                if(points<= 0)
                    return points;

                size_t bytes = points*sizeof(uint64_t);
                data = new uint64_t[points];
                if(shape.type !=UINT64){
                    printf("Type missmacht, expected UINT32, got ");
                    return -1;

                }
                JRadosDataSet *Set = _cache_data[name];
                Ndims=shape.ndims;
                int err = Set->readLayer(data, bytes);
                if(err<0)
                    printf("Error in reading char field\n");
                return bytes/sizeof(uint64_t);
            }
            int  readData(const std::string name, float *&data, size_t& Ndims, size_t* &dims ){
                DataSetShape shape;
                size_t points =  _getFieldData(name, shape, dims);
                if(points<= 0)
                    return points;
                data = new float[points];

                size_t bytes = points*sizeof(float);
                if(shape.type !=FLOAT){
                    printf("Type missmacht, expected FLOAT, got %d\n", shape.type);
                    return -1;

                }
                JRadosDataSet *Set = _cache_data[name];
                Ndims=shape.ndims;
                int err = Set->readLayer(data, bytes);
                if(err<0)
                    printf("Error in reading char field\n");
                return bytes/sizeof(float);
            }

            int  readData(const std::string name, double *&data, size_t& Ndims, size_t* &dims ){
                DataSetShape shape;
                size_t points =  _getFieldData(name, shape, dims);
                if(points<= 0)
                    return points;
                size_t bytes = points*sizeof(double);
                data = new double[points];
                if(shape.type !=DOUBLE){
                    printf("Type missmacht, expected char, got ");
                    return -1;

                }
                JRadosDataSet *Set = _cache_data[name];
                Ndims=shape.ndims;
                int err = Set->readLayer(data, bytes);
                if(err<0)
                    printf("Error in reading char field\n");
                return bytes/sizeof(double);
            }
            template< typename T >
            typename std::enable_if<std::is_arithmetic<T>::value, T>::type
            writeData(const std::string name, const T* data, const size_t Ndims, size_t const* dims){
                   return writeData(name,data,Ndims, dims);

            }


            int writeData(const std::string name, const char* data, const size_t Ndims, size_t const* dims) {
                return  _set_for_read(name)->writeLayer(Ndims, dims, data, CHAR );
            }

            int writeData(std::string name, const int8_t* data, const size_t Ndims, size_t const* dims) {
                return _set_for_read(name)->writeLayer(Ndims, dims, data, INT8 );
            }
            int writeData(std::string name, const uint8_t* data, const size_t Ndims, size_t const* dims) {
                return _set_for_read(name)->writeLayer(Ndims, dims, data, UINT8 );
            }

            int writeData(std::string name, const int16_t* data,const size_t Ndims, size_t const* dims) {
                return _set_for_read(name)->writeLayer(Ndims, dims, data, INT16 );
            }

            int writeData(std::string name, const uint16_t* data, const size_t Ndims, size_t const* dims) {
                return _set_for_read(name)->writeLayer(Ndims, dims, data, UINT16 );
            }

            int writeData(std::string name, const int32_t* data, const size_t Ndims, size_t const* dims) {
                return _set_for_read(name)->writeLayer(Ndims, dims, data, INT32 );
            }

            int writeData(std::string name, const uint32_t* data, const size_t Ndims, size_t const* dims) {
                return _set_for_read(name)->writeLayer(Ndims, dims, data, UINT32 );
            }
            int writeData(std::string name, const int64_t* data, const size_t Ndims, size_t const* dims) {
                return _set_for_read(name)->writeLayer(Ndims, dims, data, INT64 );
            }

            int writeData(std::string name,const  uint64_t* data, const size_t Ndims, size_t const* dims) {
                return _set_for_read(name)->writeLayer(Ndims, dims, data, UINT64 );
            }

            int writeData(std::string name, const float* data, const size_t Ndims, size_t const* dims) {

                return _set_for_read(name)->writeLayer(Ndims, dims, data, FLOAT );
            }

            int writeData(std::string name, const double* data, const size_t Ndims, size_t const* dims) {
                return _set_for_read(name)->writeLayer(Ndims, dims, data, DOUBLE );
            }






        private:
            std::string _name;
            std::map<std::string, JRadosDataSet*> _cache_data;

            size_t _getFieldData(std::string name, DataSetShape& shape, size_t* &dims ) {
                if(!exist(name))
                    return -1;
                JRadosDataSet *Set = _cache_data[name];
                DataSetShape* tmpShape = NULL;
                Set->getShape(tmpShape);

                memcpy(&shape, tmpShape, sizeof(DataSetShape));
                dims= new size_t[shape.ndims];
                Set->getDims(dims);
                return Set->getLayerPoints();
            }

            inline    JRadosDataSet *_set_for_read(std::string name) {

                if(!exist(name)){
                    JRadosDataSet*  set =  new JRadosDataSet(name, _name);
                    _cache_data[name]= set;
                    return set;
                }
                else
                    return  _cache_data[name];


            }




    };


/* I am not implemnting this as template, because I want some securiy in the Types */


class RadosCacheDict 
{

public:
    RadosCacheDict(std::string root):_path(""), _parent(NULL), _is_root(true) {
        Cache = new JRadosCache(root);

    }


    bool has_key(std::string key,  RadosCacheDict* &child){
        auto it =  _children.find(key);
        if(it != _children.end()){
            child = it->second;
            return true;
        }
        child=NULL;
        return  false;

    }

    bool has_key(std::string key){
        auto it =  _children.find(key);
        if(it != _children.end()){
            return true;
        }
        return  false;
    }

    RadosCacheDict* getChild(std::string name) {
        RadosCacheDict *child;
        if(has_key(name, child))
            return child;
        child = new RadosCacheDict(name, this);
        _children[name]=child;
        return child;
    }

    template< typename T >
        typename std::enable_if<std::is_arithmetic<T>::value, T>::type
        writeData(const std::string key, T *data,  size_t points) {
            std::string data_key= _path+"_"+key;
            return Cache->writeData(data_key, data, 1, &points);

        }
    template< typename T >
        typename std::enable_if<std::is_arithmetic<T>::value, T>::type
        writeData(const std::string key, const T *data, const  size_t Ndims, size_t const *dims) {
        std::string data_key;
            if(_is_root)
                data_key=key;
            else
                 data_key= _path+"_"+key;
            return Cache->writeData(data_key, data, Ndims, dims);

        }

    template< typename T >
        typename std::enable_if<std::is_arithmetic<T>::value, T>::type
        readData(const std::string key, T * &data, size_t &Ndims, size_t* &dims) {
        std::string data_key;
            if(_is_root)
                data_key=key;
            else
                data_key= _path+"_"+key;
            return Cache->readData(data_key, data, Ndims, dims);

        }
    template< typename T >
        typename std::enable_if<std::is_arithmetic<T>::value, T>::type
        readData(const std::string key, T *data) {
            std::string data_key= _path+"_"+key;
            size_t Ndims;
            size_t * dims;
            return Cache->readData(data_key, data, Ndims, dims);

        }

    bool existData(const std::string key){
        std::string data_key;
            if(_is_root)
                data_key=key;
            else
                 data_key= _path+"_"+key;
        return Cache->exist(data_key);
    }
    RadosCacheDict& operator[](const std::string child){ 
        return *(getChild(child));

    }


private:

    RadosCacheDict(std::string name,  RadosCacheDict *parent): _parent(parent), _is_root(false) {
        Cache  = parent->Cache;
        if(parent->_is_root)
            _path = name;
        else
            _path=parent->_path+"_"+name;

    }


 JRadosCache *Cache;
 std::map<std::string, RadosCacheDict*> _children;
 RadosCacheDict *_parent;
 std::string _path;
 bool _is_root;
};



}
#endif
