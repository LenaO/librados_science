#ifdef HAVE_NUMPY
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/numpyconfig.h>
#include <numpy/arrayobject.h>
#endif
 
#include "rados_cache.h"


#ifdef HAVE_NUMPY
#include "rados_hierarchy.h"

typedef struct {
    PyObject_HEAD
 PyObject *name;
 JRados::JRadosCache *_cache;    /* Type-specific fields go here. */
} RadosCacheObject;


static void
RadosCache_dealloc(RadosCacheObject* self)
{

    Py_XDECREF(self->name);

    delete(self->_cache);
    Py_TYPE(self)->tp_free((PyObject*)self);

}

static PyObject *
RadosCache_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    RadosCacheObject *self;

    self = (RadosCacheObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->name = PyString_FromString("");
        if (self->name == NULL) {
            Py_DECREF(self);
            return NULL;
        }

        self->_cache = NULL;
    }

    return (PyObject *)self;
}

static int
RadosCache_init(RadosCacheObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *tmp;
    char*  Name;
    static char *kwlist[] = {"name",NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist,
                                      &Name)
                                      )
        return -1;

    if (Name) {
        tmp = self->name;
        self->name = PyString_FromString(Name);
        Py_XDECREF(tmp);
     if(!self->_cache){
        self->_cache= new JRados::JRadosCache( Name);}
    }

    return 0;
}


static PyObject*  RadosCache_writeData(RadosCacheObject * self, PyObject* args)

{

    PyArrayObject *in_array;
    char  *obj_name;
    if (!PyArg_ParseTuple(args, "sO!", &obj_name, &PyArray_Type, &in_array))
        return NULL;

    size_t Ndims = PyArray_NDIM(in_array);
    npy_intp* int_dims = PyArray_DIMS(in_array);
    size_t *dims = new size_t[Ndims];
    for(int i= 0; i<Ndims; i++) {
        dims[i] = int_dims[i];
    }
    int ret = 0;
    PyArray_Descr *dtype =  PyArray_DESCR(in_array);

    switch (dtype->type_num){
     case NPY_INT8:{
            int8_t* data = (int8_t*)PyArray_DATA(in_array);
            ret = self->_cache->writeData(obj_name, data, Ndims,  dims);}
            break;
     case NPY_UINT8:{
            uint8_t* data = (uint8_t*)PyArray_DATA(in_array);
            ret = self->_cache->writeData(obj_name, data, Ndims,  dims);}
            break;
     case NPY_INT16:{
            int16_t* data = (int16_t*)PyArray_DATA(in_array);
            ret = self->_cache->writeData(obj_name, data, Ndims,  dims);}
            break;
     case NPY_UINT16:{
            uint16_t* data = (uint16_t*)PyArray_DATA(in_array);
            ret = self->_cache->writeData(obj_name, data, Ndims,  dims);}
            break;
     case NPY_INT32:{
            int32_t* data = (int32_t*)PyArray_DATA(in_array);
            ret = self->_cache->writeData(obj_name, data, Ndims,  dims);}
            break;
     case NPY_UINT32:{
            uint32_t* data = (uint32_t*)PyArray_DATA(in_array);
            ret = self->_cache->writeData(obj_name, data, Ndims,  dims);}
            break;
     case NPY_INT64:{
            int64_t* data = (int64_t*)PyArray_DATA(in_array);
            ret = self->_cache->writeData(obj_name, data, Ndims,  dims);}
            break;
     case NPY_UINT64:{
            uint64_t* data = (uint64_t*)PyArray_DATA(in_array);
            ret = self->_cache->writeData(obj_name, data, Ndims,  dims);}
            break;
     case NPY_FLOAT32:{
            float* data = (float*)PyArray_DATA(in_array);
            ret = self->_cache->writeData(obj_name, data, Ndims,  dims);}
            break;
     case NPY_FLOAT64:{
            double* data = (double*)PyArray_DATA(in_array);
            ret = self->_cache->writeData(obj_name, data, Ndims,  dims);}
            break;
    }
    delete dims;
    return PyLong_FromLong(ret);
}

static PyObject*  RadosCache_ObjectExists(RadosCacheObject * self, PyObject* args)
{
    PyObject *out;
    char  *obj_name;
    bool exists;
    if (!PyArg_ParseTuple(args, "s", &obj_name))
        return NULL;

    exists= self->_cache->exist(obj_name);

    return PyBool_FromLong((long) exists);

}

static PyObject*  RadosCache_readData(RadosCacheObject * self, PyObject* args)
{
    PyObject *out_array;
    char  *obj_name;
    size_t* dims;
    int Ndims;
    int jtype;
    char *data;
    int read;
    if (!PyArg_ParseTuple(args, "s", &obj_name))
        return NULL;

    size_t ret = self->_cache->getShape(obj_name, Ndims, dims, jtype);
    if(ret<0)
        return NULL;
    const std::string name(obj_name);
    npy_intp* int_dims = new npy_intp[Ndims];

    for(int i= 0; i<Ndims; i++) {
        int_dims[i] = dims[i];
    }



    switch(jtype)  {

        case JRados::UINT8 :{
                                data = (char*) malloc(ret * sizeof(uint8_t));
                                read =self->_cache->readDataRaw(name, (uint8_t*)data, ret);
                                out_array= PyArray_SimpleNewFromData(Ndims, int_dims, NPY_UINT8, (void*)data);
                            }
                            break;

        case JRados::INT8 :{
                               data = (char*) malloc(ret * sizeof(int8_t));
                               read =self->_cache->readDataRaw(name, (int8_t*)data, ret);
                               out_array= PyArray_SimpleNewFromData(Ndims, int_dims, NPY_INT8, (void*)data);
                           }
                           break;
        case JRados::UINT16 :{
                                data = (char*) malloc(ret * sizeof(uint16_t));
                                read =self->_cache->readDataRaw(name, (uint16_t*)data, ret);
                                out_array= PyArray_SimpleNewFromData(Ndims, int_dims, NPY_UINT16, (void*)data);
                            }
                            break;

        case JRados::INT16 :{
                               data = (char*) malloc(ret * sizeof(int16_t));
                               read =self->_cache->readDataRaw(name, (int16_t*)data, ret);
                               out_array= PyArray_SimpleNewFromData(Ndims, int_dims, NPY_INT16, (void*)data);
                           }
                           break;
        case JRados::UINT32 :{
                                data = (char*) malloc(ret * sizeof(uint32_t));
                                read =self->_cache->readDataRaw(name, (uint32_t*)data, ret);
                                out_array= PyArray_SimpleNewFromData(Ndims, int_dims, NPY_UINT32, (void*)data);
                            }
                            break;

        case JRados::INT32 :{
                               data = (char*) malloc(ret * sizeof(int32_t));
                               read =self->_cache->readDataRaw(name, (int32_t*)data, ret);
                               out_array= PyArray_SimpleNewFromData(Ndims, int_dims, NPY_INT32, (void*)data);
                           }
                           break;
        case JRados::UINT64 :{
                                data = (char*) malloc(ret * sizeof(uint64_t));
                                read =self->_cache->readDataRaw(name, (uint64_t*)data, ret);
                                out_array= PyArray_SimpleNewFromData(Ndims, int_dims, NPY_UINT64, (void*)data);
                            }
                            break;

        case JRados::INT64 :{
                               data = (char*) malloc(ret * sizeof(int64_t));
                               read =self->_cache->readDataRaw(name, (int64_t*)data, ret);
                               out_array= PyArray_SimpleNewFromData(Ndims, int_dims, NPY_INT64, (void*)data);
                           }
                           break;

        case JRados::FLOAT :{
                                data = (char*) malloc(ret * sizeof(float));
                                read =self->_cache->readDataRaw(name, (float*)data, ret);
                                out_array= PyArray_SimpleNewFromData(Ndims, int_dims, NPY_FLOAT32, (void*)data);
                            }
                            break;

        case JRados::DOUBLE :{
                               data = (char*) malloc(ret * sizeof(double));
                               read =self->_cache->readDataRaw(name, (double*)data, ret);
                               out_array= PyArray_SimpleNewFromData(Ndims, int_dims, NPY_FLOAT64, (void*)data);
                           }
                           break;






    }
    delete dims;
    delete int_dims;
    return (PyObject*) out_array;
}




static PyMethodDef RadosCache_methods[] = {

    {"ObjectExists", (PyCFunction)RadosCache_ObjectExists, METH_VARARGS,
     "Tests, if object exists in cache"
    },
    {"writeData", (PyCFunction)RadosCache_writeData, METH_VARARGS,
     "write  data from numpy-array to cache"
    },
    {"readData", (PyCFunction)RadosCache_readData, METH_VARARGS,
     "read data from cache to numpy-array"
    },
    {NULL}  /* Sentinel */
};

static PyMemberDef RadosCache_members[] = {
    {"Name", T_OBJECT_EX, offsetof(RadosCacheObject, name), 0,
     "Cache Name"},
    {NULL}  /* Sentinel */
};

static PyTypeObject RadosCacheType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "radoscache.RadosCache",             /* tp_name */
    sizeof(RadosCacheObject), /* tp_basicsize */
    0,                         /* tp_itemsize */
    0,                         /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "Rados-Cache objects",           /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    RadosCache_methods,             /* tp_methods */
    RadosCache_members,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)RadosCache_init,      /* tp_init */
    0,                         /* tp_alloc */
    RadosCache_new,                 /* tp_new */

};

static PyMethodDef RadosCacheMethods[] = {
 {NULL}
};


typedef struct {
    PyObject_HEAD
 PyObject *_name;
PyObject *_poolname;
 JRados::JRadosDataSet *_set;
    /* Type-specific fields go here. */

} RadosDataSetObject;


static void
RadosDataSet_dealloc(RadosDataSetObject* self)
{

    Py_XDECREF(self->_name);

    delete(self->_set);
    Py_TYPE(self)->tp_free((PyObject*)self);

}

static PyObject *
RadosDataSet_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    RadosDataSetObject *self;

    self = (RadosDataSetObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->_name = PyString_FromString("");
        if (self->_name == NULL) {
            Py_DECREF(self);
            return NULL;
        }
        self->_poolname = PyString_FromString("");
        if (self->_poolname == NULL) {
            Py_DECREF(self);
            return NULL;
        }
        self->_set = NULL;
    }

    return (PyObject *)self;
}

static int
RadosDataSet_init(RadosDataSetObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *tmp;
    char*  Name;
    char*  PoolName;
    static char *kwlist[] = {"name","pool_name",NULL};
    if (! PyArg_ParseTupleAndKeywords(args, kwds, "ss", kwlist,
                                      &Name, &PoolName)
                                      )
        return -1;

    if (Name && PoolName) {
        tmp = self->_name;
        self->_name = PyString_FromString(Name);
        Py_XDECREF(tmp);

        tmp = self->_poolname;

        self->_poolname = PyString_FromString(PoolName);
        Py_XDECREF(tmp);


     if(!self->_set){
        self->_set= new JRados::JRadosDataSet( Name, PoolName);}
    }

    return 0;
}

static PyObject*  
RadosDataSet_remove(RadosDataSetObject *self)
{
    int ret = self->_set->remove();

    return PyLong_FromLong(ret);

}



static PyObject*  RadosDataSet_getDims(RadosDataSetObject * self){

    int Ndims;
    int jtype;
    size_t* dims;
    size_t ret = self->_set->getShape(Ndims, dims, jtype);

    PyObject*    py_dims =  PyTuple_New(Ndims);
    for(int i =0; i< Ndims; i++)
        PyTuple_SetItem(py_dims, i,  PyLong_FromLong(dims[i]));

    return py_dims;
}


static PyObject*  RadosDataSet_writeData(RadosDataSetObject * self, PyObject* args)

{

    PyArrayObject *tmp_array, *in_array;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &tmp_array))
        return NULL;


    in_array = PyArray_GETCONTIGUOUS(tmp_array);
    size_t Ndims = PyArray_NDIM(in_array);
    npy_intp* int_dims = PyArray_DIMS(in_array);
    size_t *dims = new size_t[Ndims];
    for(int i= 0; i<Ndims; i++) {
        dims[i] = int_dims[Ndims-i-1];
    }
    int ret = 0;
    PyArray_Descr *dtype =  PyArray_DESCR(in_array);

    switch (dtype->type_num){
     case NPY_INT8:{
            int8_t* data = (int8_t*)PyArray_DATA(in_array);
            ret = self->_set->writeLayer(Ndims, dims, data, JRados::INT8 );}
            break;
     case NPY_UINT8:{
            uint8_t* data = (uint8_t*)PyArray_DATA(in_array);
            ret = self->_set->writeLayer(Ndims, dims, data, JRados::UINT8 );}
            break;
     case NPY_INT16:{
            int16_t* data = (int16_t*)PyArray_DATA(in_array);
            ret = self->_set->writeLayer(Ndims, dims, data, JRados::INT16 );}
            break;
     case NPY_UINT16:{
            uint16_t* data = (uint16_t*)PyArray_DATA(in_array);
            ret = self->_set->writeLayer(Ndims, dims, data, JRados::UINT16 );}
            break;
     case NPY_INT32:{
            int32_t* data = (int32_t*)PyArray_DATA(in_array);
            ret = self->_set->writeLayer(Ndims, dims, data, JRados::INT32 );}
            break;
     case NPY_UINT32:{
            uint32_t* data = (uint32_t*)PyArray_DATA(in_array);
            ret = self->_set->writeLayer(Ndims, dims, data, JRados::UINT32 );}
            break;
     case NPY_INT64:{
            int64_t* data = (int64_t*)PyArray_DATA(in_array);
            ret = self->_set->writeLayer(Ndims, dims, data, JRados::INT64 );}
            break;
     case NPY_UINT64:{
            uint64_t* data = (uint64_t*)PyArray_DATA(in_array);
            ret = self->_set->writeLayer(Ndims, dims, data, JRados::UINT64 );}
            break;
     case NPY_FLOAT32:{
            float* data = (float*)PyArray_DATA(in_array);
            ret = self->_set->writeLayer(Ndims, dims, data, JRados::FLOAT );}
            break;
     case NPY_FLOAT64:{
            double* data = (double*)PyArray_DATA(in_array);
            ret = self->_set->writeLayer(Ndims, dims, data, JRados::DOUBLE );}
            break;
   }
    delete dims;

    Py_XDECREF(in_array);
    return PyLong_FromLong(ret);
}

static PyObject*  RadosDataSet_readData(RadosDataSetObject * self)
{
    PyObject *out_array;
    size_t* dims;
    int Ndims;
    int jtype;
    char *data;
    int read;
    size_t ret = self->_set->getShape(Ndims, dims, jtype);
    if(ret<0)
        return NULL;

    npy_intp* int_dims = new npy_intp[Ndims];

    for(int i= 0; i<Ndims; i++) {
        int_dims[i] = dims[Ndims-i-1];
    }



    switch(jtype)  {

        case JRados::UINT8 :{
                                printf("malloc %ld points\n",ret);
                                data = (char*) malloc(ret * sizeof(uint8_t));
                                read =self->_set->readLayer((uint8_t*)data, ret);
                                out_array= PyArray_SimpleNewFromData(Ndims, int_dims, NPY_UINT8, (void*)data);
                            }
                            break;

        case JRados::INT8 :{
                               data = (char*) malloc(ret * sizeof(int8_t));
                               read =self->_set->readLayer((int8_t*)data, ret);
                               out_array= PyArray_SimpleNewFromData(Ndims, int_dims, NPY_INT8, (void*)data);
                           }
                           break;
        case JRados::UINT16 :{
                                data = (char*) malloc(ret * sizeof(uint16_t));
                                read =self->_set->readLayer((uint16_t*)data, ret);
                                out_array= PyArray_SimpleNewFromData(Ndims, int_dims, NPY_UINT16, (void*)data);
                            }
                            break;

        case JRados::INT16 :{
                               data = (char*) malloc(ret * sizeof(int16_t));
                               read =self->_set->readLayer((int16_t*)data, ret);
                               out_array= PyArray_SimpleNewFromData(Ndims, int_dims, NPY_INT16, (void*)data);
                           }
                           break;
        case JRados::UINT32 :{
                                data = (char*) malloc(ret * sizeof(uint32_t));
                                read =self->_set->readLayer( (uint32_t*)data, ret);
                                out_array= PyArray_SimpleNewFromData(Ndims, int_dims, NPY_UINT32, (void*)data);
                            }
                            break;

        case JRados::INT32 :{
                               data = (char*) malloc(ret * sizeof(int32_t));
                               read =self->_set->readLayer((int32_t*)data, ret);
                               out_array= PyArray_SimpleNewFromData(Ndims, int_dims, NPY_INT32, (void*)data);
                           }
                           break;
        case JRados::UINT64 :{
                                data = (char*) malloc(ret * sizeof(uint64_t));
                                read =self->_set->readLayer((uint64_t*)data, ret);
                                out_array= PyArray_SimpleNewFromData(Ndims, int_dims, NPY_UINT64, (void*)data);
                            }
                            break;

        case JRados::INT64 :{
                               data = (char*) malloc(ret * sizeof(int64_t));
                               read =self->_set->readLayer((int64_t*)data, ret);
                               out_array= PyArray_SimpleNewFromData(Ndims, int_dims, NPY_INT64, (void*)data);
                           }
                           break;

        case JRados::FLOAT :{
                                data = (char*) malloc(ret * sizeof(float));
                                read =self->_set->readLayer((float*)data, ret);
                                out_array= PyArray_SimpleNewFromData(Ndims, int_dims, NPY_FLOAT32, (void*)data);
                            }
                            break;

        case JRados::DOUBLE :{
                               data = (char*) malloc(ret * sizeof(double));
                               read =self->_set->readLayer((double*)data, ret);
                               out_array= PyArray_SimpleNewFromData(Ndims, int_dims, NPY_FLOAT64, (void*)data);
                           }
                           break;


    }
    delete dims;
    delete int_dims;
    return (PyObject*) out_array;
}

static inline std::string  jtype_to_string(int jtype){

    switch(jtype)  {

        case JRados::UINT8:     return  "uint8";
        case JRados::INT8:      return "int8";
        case JRados::UINT16 :   return "uint16";
        case JRados::INT16 :  return "int16";
        case JRados::UINT32:     return  "uint32";
        case JRados::INT32:      return "int32";
        case JRados::UINT64 :   return "uint64";
        case JRados::INT64 :  return "int64";
        case JRados::FLOAT : return "float";
        case JRados::DOUBLE : return "double";
        default: return "unknown";

}
}

static size_t writeLine1D( RadosDataSetObject * self, PyArrayObject *in_array, size_t start, size_t end, int jtype ){


    PyArray_Descr *dtype =  PyArray_DESCR(in_array);
    size_t ret;
    switch (dtype->type_num){
        case NPY_INT8:{
                          if(jtype != JRados::INT8){
                              std::cerr<<"Datatype missmatch, Array has "<<jtype_to_string(jtype)<<"but array is int8"<<std::endl;
                              return -1;
                          } 
                          int8_t* data = (int8_t*)PyArray_DATA(in_array);
                          ret = self->_set->writeLine1D(start,end, data);}
                      break;
        case NPY_UINT8:{
                           if(jtype != JRados::UINT8){
                               std::cerr<<"Datatype missmatch, Array has "<<jtype_to_string(jtype)<<"but array is uint8"<<std::endl;
                               return -1;
                           } 
                           uint8_t* data = (uint8_t*)PyArray_DATA(in_array);
                           ret = self->_set->writeLine1D(start,end, data);}
                       break;
       case NPY_INT16:{
                           if(jtype != JRados::INT16){
                               std::cerr<<"Datatype missmatch, Array has "<<jtype_to_string(jtype)<<"but array is int8"<<std::endl;
                               return -1;
                           }
                           int16_t* data = (int16_t*)PyArray_DATA(in_array);
                           ret = self->_set->writeLine1D(start,end,data);}
                      break;
       case NPY_UINT16:{
                           if(jtype != JRados::UINT16){
                               std::cerr<<"Datatype missmatch, Array has "<<jtype_to_string(jtype)<<"but array is uint16"<<std::endl;
                               return -1;
                           }

                           uint16_t* data = (uint16_t*)PyArray_DATA(in_array);
                           ret = self->_set->writeLine1D(start, end, data);}
                       break;
       case NPY_INT32:{
                          if(jtype != JRados::INT32){
                              std::cerr<<"Datatype missmatch, Array has "<<jtype_to_string(jtype)<<"but array is int32"<<std::endl;
                              return -1;
                          }

                          int32_t* data = (int32_t*)PyArray_DATA(in_array);
                          ret = self->_set->writeLine1D(start, end, data);}
                      break;
       case NPY_UINT32:{
                           if(jtype != JRados::UINT32){
                               std::cerr<<"Datatype missmatch, Array has "<<jtype_to_string(jtype)<<"but array is uint32"<<std::endl;
                               return -1;
                           }
                           uint32_t* data = (uint32_t*)PyArray_DATA(in_array);
                           ret = self->_set->writeLine1D(start, end, data);}
                       break;
       case NPY_INT64:{
                          if(jtype != JRados::INT64){
                              std::cerr<<"Datatype missmatch, Array has "<<jtype_to_string(jtype)<<"but array is int64"<<std::endl;
                              return -1;
                          }
                          int64_t* data = (int64_t*)PyArray_DATA(in_array);
                          ret = self->_set->writeLine1D(start, end, data);}
                      break;
       case NPY_UINT64:{
                           if(jtype != JRados::INT32){
                               std::cerr<<"Datatype missmatch, Array has "<<jtype_to_string(jtype)<<"but array is uint64"<<std::endl;
                               return -1;
                           }

                           uint64_t* data = (uint64_t*)PyArray_DATA(in_array);
                           ret = self->_set->writeLine1D(start, end, data);}
                       break;
       case NPY_FLOAT32:{
                            if(jtype != JRados::FLOAT){
                                std::cerr<<"Datatype missmatch, Array has "<<jtype_to_string(jtype)<<"but array is float32"<<std::endl;
                                return -1;
                            }
                            float* data = (float*)PyArray_DATA(in_array);
                            ret = self->_set->writeLine1D(start, end, data);}
                        break;
       case NPY_FLOAT64:{
                            if(jtype != JRados::DOUBLE){
                                std::cerr<<"Datatype missmatch, Array has "<<jtype_to_string(jtype)<<"but array is float64"<<std::endl;
                                return -1;
                            }


                            double* data = (double*)PyArray_DATA(in_array);

                            ret = self->_set->writeLine1D(start, end, data);}
                        break;
    }

    return ret;
}



static size_t writeBox2D( RadosDataSetObject * self, PyArrayObject *in_array, size_t start[2], size_t end[2], int jtype ){


    PyArray_Descr *dtype =  PyArray_DESCR(in_array);
    size_t ret;
    switch (dtype->type_num){
        case NPY_INT8:{
                          if(jtype != JRados::INT8){
                              std::cerr<<"Datatype missmatch, Array has "<<jtype_to_string(jtype)<<"but array is int8"<<std::endl;
                              return -1;
                          } 
                          int8_t* data = (int8_t*)PyArray_DATA(in_array);
                          ret = self->_set->writeBox(start,end, data);}
                      break;
        case NPY_UINT8:{
                           if(jtype != JRados::UINT8){
                               std::cerr<<"Datatype missmatch, Array has "<<jtype_to_string(jtype)<<"but array is uint8"<<std::endl;
                               return -1;
                           } 
                           uint8_t* data = (uint8_t*)PyArray_DATA(in_array);
                           ret = self->_set->writeBox(start,end, data);}
                       break;
       case NPY_INT16:{
                           if(jtype != JRados::INT16){
                               std::cerr<<"Datatype missmatch, Array has "<<jtype_to_string(jtype)<<"but array is int8"<<std::endl;
                               return -1;
                           }
                           int16_t* data = (int16_t*)PyArray_DATA(in_array);
                           ret = self->_set->writeBox(start,end,data);}
                      break;
       case NPY_UINT16:{
                           if(jtype != JRados::UINT16){
                               std::cerr<<"Datatype missmatch, Array has "<<jtype_to_string(jtype)<<"but array is uint16"<<std::endl;
                               return -1;
                           }

                           uint16_t* data = (uint16_t*)PyArray_DATA(in_array);
                           ret = self->_set->writeBox(start, end, data);}
                       break;
       case NPY_INT32:{
                          if(jtype != JRados::INT32){
                              std::cerr<<"Datatype missmatch, Array has "<<jtype_to_string(jtype)<<"but array is int32"<<std::endl;
                              return -1;
                          }

                          int32_t* data = (int32_t*)PyArray_DATA(in_array);
                          ret = self->_set->writeBox(start, end, data);}
                      break;
       case NPY_UINT32:{
                           if(jtype != JRados::UINT32){
                               std::cerr<<"Datatype missmatch, Array has "<<jtype_to_string(jtype)<<"but array is uint32"<<std::endl;
                               return -1;
                           }
                           uint32_t* data = (uint32_t*)PyArray_DATA(in_array);
                           ret = self->_set->writeBox(start, end, data);}
                       break;
       case NPY_INT64:{
                          if(jtype != JRados::INT64){
                              std::cerr<<"Datatype missmatch, Array has "<<jtype_to_string(jtype)<<"but array is int64"<<std::endl;
                              return -1;
                          }
                          int64_t* data = (int64_t*)PyArray_DATA(in_array);
                          ret = self->_set->writeBox(start, end, data);}
                      break;
       case NPY_UINT64:{
                           if(jtype != JRados::INT32){
                               std::cerr<<"Datatype missmatch, Array has "<<jtype_to_string(jtype)<<"but array is uint64"<<std::endl;
                               return -1;
                           }

                           uint64_t* data = (uint64_t*)PyArray_DATA(in_array);
                           ret = self->_set->writeBox(start, end, data);}
                       break;
       case NPY_FLOAT32:{
                            if(jtype != JRados::FLOAT){
                                std::cerr<<"Datatype missmatch, Array has "<<jtype_to_string(jtype)<<"but array is float32"<<std::endl;
                                return -1;
                            }
                            float* data = (float*)PyArray_DATA(in_array);
                            ret = self->_set->writeBox(start, end, data);}
                        break;
       case NPY_FLOAT64:{
                            if(jtype != JRados::DOUBLE){
                                std::cerr<<"Datatype missmatch, Array has "<<jtype_to_string(jtype)<<"but array is float64"<<std::endl;
                                return -1;
                            }

                            double* data = (double*)PyArray_DATA(in_array);
                            ret = self->_set->writeBox(start, end, data);}
                        break;
    }

    return ret;
}

static PyObject*  RadosDataSet_writeBox(RadosDataSetObject * self, PyObject* args, PyObject *kwds)

{

    PyArrayObject *tmp_array,*in_array;
    PySliceObject *slicex =NULL,  *slicey=NULL;
    Py_ssize_t size, start[2],stop[2],step[2], nsize;
    size_t* dims;
    int Framedims;
    int jtype;
    size_t ret =0;
    uint64_t xstart=-1, ystart=-1;
    static char *kwlist[] = {"data","xslice", "yslice", "xstart", "ystart",NULL};


    if (! PyArg_ParseTupleAndKeywords(args, kwds,"O!|O!O!KK", kwlist, &PyArray_Type, &tmp_array, &PySlice_Type, &slicex ,&PySlice_Type, &slicey,&xstart,&ystart))
        return NULL;

    in_array = PyArray_GETCONTIGUOUS(tmp_array);

    size_t Ndims = PyArray_NDIM(in_array);

    npy_intp* int_dims = PyArray_DIMS(in_array);

    if(xstart!=-1){
        start[0]=xstart;
        stop[0]=xstart+ (Ndims> 1? int_dims[1]:int_dims[0]);
    }

    else if(slicex!=NULL)
        PySlice_GetIndices(slicex,  Ndims> 1? int_dims[1]:int_dims[0], start, stop, step);
    else{
        start[0]=0;
        stop[0]= (Ndims> 1? int_dims[1]:int_dims[0]);

    }


    if(ystart!=-1){
        start[1]=ystart;
        stop[1]=ystart+ (Ndims>1 ? int_dims[0]:1);
    }
    else if(slicey!=NULL && Ndims >1 )
        PySlice_GetIndices(slicey, int_dims[0], start+1, stop+1, step+1);

    else {
        start[1]=0;
        stop[1]= (Ndims > 1? int_dims[0]:1);
    }

    ret = self->_set->getShape(Framedims, dims, jtype);


    if(Framedims == 1){
        ret =  writeLine1D(self, in_array, start[0], stop[0], jtype );
        if(ret<0)
            return NULL;

    }
    else if(Framedims == 2){
        size_t sstart[2] ={start[0],start[1]};
        size_t  sstop[2]= {stop[0], stop[1]};
        ret =  writeBox2D(self, in_array, sstart, sstop, jtype );
        if(ret<0)
            return NULL;
    }
    else{
        printf("Currently, only 1 and 2 D Boxes are supported");
        return NULL;
    }

    Py_DECREF(in_array);
    return PyLong_FromLong(ret);
}

static PyObject*  readLine1D(RadosDataSetObject * self, size_t start, size_t end, size_t points, int jtype)
{
    PyObject *out_array;

    npy_intp int_dims = end-start;

    char* data;
    size_t read;

    switch(jtype)  {

        case JRados::UINT8 :{
                                data = (char*) malloc(points * sizeof(uint8_t));
                                read =self->_set->readLine1D(start, end, (uint8_t*)data);
                                out_array= PyArray_SimpleNewFromData(1, &int_dims, NPY_UINT8, (void*)data);
                            }
                            break;

        case JRados::INT8 :{
                               data = (char*) malloc(points * sizeof(int8_t));
                               read =self->_set->readLine1D(start, end, (int8_t*)data);
                               out_array= PyArray_SimpleNewFromData(1, &int_dims, NPY_INT8, (void*)data);
                           }
                           break;
        case JRados::UINT16 :{
                                data = (char*) malloc(points * sizeof(uint16_t));
                                read =self->_set->readLine1D(start, end , (uint16_t*)data);
                                out_array= PyArray_SimpleNewFromData(1, &int_dims, NPY_UINT16, (void*)data);
                            }
                            break;

        case JRados::INT16 :{
                               data = (char*) malloc(points * sizeof(int16_t));
                               read =self->_set->readLine1D(start, end, (int16_t*)data);
                               out_array= PyArray_SimpleNewFromData(1, &int_dims, NPY_INT16, (void*)data);
                           }
                           break;
        case JRados::UINT32 :{
                                data = (char*) malloc(points * sizeof(uint32_t));
                                read =self->_set->readLine1D(start, end,  (uint32_t*)data);
                                out_array= PyArray_SimpleNewFromData(1, &int_dims, NPY_UINT32, (void*)data);
                            }
                            break;

        case JRados::INT32 :{
                               data = (char*) malloc(points * sizeof(int32_t));
                               read =self->_set->readLine1D(start, end, (int32_t*)data);
                               out_array= PyArray_SimpleNewFromData(1, &int_dims, NPY_INT32, (void*)data);
                           }
                           break;
        case JRados::UINT64 :{
                                data = (char*) malloc(points * sizeof(uint64_t));
                                read =self->_set->readLine1D(start, end, (uint64_t*)data);
                                out_array= PyArray_SimpleNewFromData(1, &int_dims, NPY_UINT64, (void*)data);
                            }
                            break;

        case JRados::INT64 :{
                               data = (char*) malloc(points * sizeof(int64_t));
                               read =self->_set->readLine1D(start, end, (int64_t*)data);
                               out_array= PyArray_SimpleNewFromData(1, &int_dims, NPY_INT64, (void*)data);
                           }
                           break;

        case JRados::FLOAT :{
                                data = (char*) malloc(points * sizeof(float));
                                read =self->_set->readLine1D(start, end, (float*)data);
                                out_array= PyArray_SimpleNewFromData(1, &int_dims, NPY_FLOAT32, (void*)data);
                            }
                            break;

        case JRados::DOUBLE :{
                               data = (char*) malloc(points * sizeof(double));
                               read =self->_set->readLine1D(start, end, (double*)data);
                               out_array= PyArray_SimpleNewFromData(1, &int_dims, NPY_FLOAT64, (void*)data);
                           }
                           break;


    }

    return (PyObject*) out_array;
}
static PyObject*  readBox2D(RadosDataSetObject * self, size_t start[2], size_t end[2], size_t points, int jtype)
{
    PyObject *out_array;

    npy_intp int_dims[2] ={end[0]-start[0], end[1]-start[1]};

    char* data;
    size_t read;

    switch(jtype)  {

        case JRados::UINT8 :{
                                data = (char*) malloc(points * sizeof(uint8_t));
                                read =self->_set->readBox(start, end, (uint8_t*)data);
                                out_array= PyArray_SimpleNewFromData(2, int_dims, NPY_UINT8, (void*)data);
                            }
                            break;

        case JRados::INT8 :{
                               data = (char*) malloc(points * sizeof(int8_t));
                               read =self->_set->readBox(start, end, (int8_t*)data);
                               out_array= PyArray_SimpleNewFromData(2, int_dims, NPY_INT8, (void*)data);
                           }
                           break;
        case JRados::UINT16 :{
                                data = (char*) malloc(points * sizeof(uint16_t));
                                read =self->_set->readBox(start, end , (uint16_t*)data);
                                out_array= PyArray_SimpleNewFromData(2, int_dims, NPY_UINT16, (void*)data);
                            }
                            break;

        case JRados::INT16 :{
                               data = (char*) malloc(points * sizeof(int16_t));
                               read =self->_set->readBox(start, end, (int16_t*)data);
                               out_array= PyArray_SimpleNewFromData(2, int_dims, NPY_INT16, (void*)data);
                           }
                           break;
        case JRados::UINT32 :{
                                data = (char*) malloc(points * sizeof(uint32_t));
                                read =self->_set->readBox(start, end,  (uint32_t*)data);
                                out_array= PyArray_SimpleNewFromData(2, int_dims, NPY_UINT32, (void*)data);
                            }
                            break;

        case JRados::INT32 :{
                               data = (char*) malloc(points * sizeof(int32_t));
                               read =self->_set->readBox(start, end, (int32_t*)data);
                               out_array= PyArray_SimpleNewFromData(2, int_dims, NPY_INT32, (void*)data);
                           }
                           break;
        case JRados::UINT64 :{
                                data = (char*) malloc(points * sizeof(uint64_t));
                                read =self->_set->readBox(start, end, (uint64_t*)data);
                                out_array= PyArray_SimpleNewFromData(2, int_dims, NPY_UINT64, (void*)data);
                            }
                            break;

        case JRados::INT64 :{
                               data = (char*) malloc(points * sizeof(int64_t));
                               read =self->_set->readBox(start, end, (int64_t*)data);
                               out_array= PyArray_SimpleNewFromData(2, int_dims, NPY_INT64, (void*)data);
                           }
                           break;

        case JRados::FLOAT :{
                                data = (char*) malloc(points * sizeof(float));
                                read =self->_set->readBox(start, end, (float*)data);
                                out_array= PyArray_SimpleNewFromData(2, int_dims, NPY_FLOAT32, (void*)data);
                            }
                            break;

        case JRados::DOUBLE :{
                               data = (char*) malloc(points * sizeof(double));
                               read =self->_set->readBox(start, end, (double*)data);
                               out_array= PyArray_SimpleNewFromData(2, int_dims, NPY_FLOAT64, (void*)data);
                           }
                           break;


    }

    return (PyObject*) out_array;
}

static PyObject*  RadosDataSet_readBox(RadosDataSetObject * self, PyObject* args, PyObject *kwds)

{

    PyObject *out_array;
    PySliceObject *slicex =NULL,  *slicey=NULL;
    Py_ssize_t size, start[2],stop[2],step[2], nsize;
    size_t* dims;
    int Framedims;
    int jtype;
    size_t ret =0;
    uint64_t xstart=-1, ystart=-1;

    uint64_t xstop=-1, ystop=-1;
    static char *kwlist[] = {"xslice", "yslice", "xstart", "ystart","xstop", "ystop",NULL};


    if (! PyArg_ParseTupleAndKeywords(args, kwds,"|O!O!KKKK", kwlist,&PySlice_Type, &slicex ,&PySlice_Type, &slicey,&xstart,&ystart, &xstop, &ystop))
        return NULL;


    size_t points  = self->_set->getShape(Framedims, dims, jtype);

    if(xstart!=-1){
        start[0]=xstart;
        stop[0]= (xstop!=-1? xstop: dims[0]);
    }
    else if(slicex!=NULL){
        PySlice_GetIndices(slicex, dims[0], start, stop, step);
    }
    else{
        start[0]=0;
        stop[0]= (xstop !=-1? xstop: dims[0]);
    }

    if(Framedims >1) {
        if(ystart!=-1){
            start[1]=ystart;
            stop[1]=(ystop!= -1? ystop: dims[1]);
        }
        else if(slicey!=NULL ){
            PySlice_GetIndices(slicey, dims[0], start+1, stop+1, step+1);
        }
        else {
            start[1]=0;
            stop[1]= ystop!= -1? ystop: dims[1];
        }
    }
    printf("%ld %ld %ld %ld\n", start[0], stop[0], start[1], stop[1]);


    if(Framedims == 1){
        out_array =  readLine1D(self, start[0], stop[0], points, jtype );
        if(out_array<0)
            return NULL;

    }
    else if(Framedims == 2){
        size_t sstart[2] ={start[0],start[1]};
        size_t  sstop[2]= {stop[0], stop[1]};
        out_array =  readBox2D(self, sstart, sstop, points, jtype );
        if(ret<0)
            return NULL;
    }
    else{
        printf("Currently, only 1 and 2 D Boxes are supported");
        return NULL;
    }

    return (PyObject*) out_array;
}




static PyMemberDef RadosDataSet_members[] = {
    {"Name", T_OBJECT_EX, offsetof(RadosDataSetObject, _name), 0,
     "DataSet Name"},
    {"PoolName", T_OBJECT_EX, offsetof(RadosDataSetObject, _poolname), 0,
     "DataSet Pool Name"},

    {NULL}  /* Sentinel */
};

static PyMethodDef RadosDataSet_methods[] = {
    {"writeData", (PyCFunction)RadosDataSet_writeData, METH_VARARGS,
     "write  data from numpy-array to a Data Set"
    },
    {"readData", (PyCFunction)RadosDataSet_readData, METH_NOARGS,
     "write  data from numpy-array to a Data Set"
    },
     {"writeBox", (PyCFunction)RadosDataSet_writeBox, METH_VARARGS|METH_KEYWORDS, 
       "write from/to box in array"
    },
    {"readBox", (PyCFunction)RadosDataSet_readBox, METH_VARARGS|METH_KEYWORDS, 
       "read from dataset to box in array"
    },
    { "getDims", (PyCFunction) RadosDataSet_getDims, METH_NOARGS,
        "ret dimensions of dataSet" 
    },
   {"remove", (PyCFunction)RadosDataSet_remove, METH_NOARGS,
     "remove the dataSet object"
    },


 {NULL}
};


static PyTypeObject RadosDataSetType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "radoscache.RadosDataSet",             /* tp_name */
    sizeof(RadosDataSetObject), /* tp_basicsize */
    0,                         /* tp_itemsize */
    0,                         /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "Rados-DataSet objects",           /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    RadosDataSet_methods,             /* tp_methods */
    RadosDataSet_members,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)RadosDataSet_init,      /* tp_init */
    0,                         /* tp_alloc */
    RadosDataSet_new,                 /* tp_new */

};

typedef struct {
    PyObject_HEAD
 PyObject *_name;
 PyObject *_poolname;
 JRados::JRadosCSchema *_set;
    /* Type-specific fields go here. */

} RadosSchemaObject;

static void
RadosSchema_dealloc(RadosSchemaObject* self)
{

    Py_XDECREF(self->_name);

    delete(self->_set);
    Py_TYPE(self)->tp_free((PyObject*)self);

}

static PyObject *
RadosSchema_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    RadosSchemaObject *self;

    self = (RadosSchemaObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->_name = PyString_FromString("");
        if (self->_name == NULL) {
            Py_DECREF(self);
            return NULL;
        }
        self->_poolname = PyString_FromString("");
        if (self->_poolname == NULL) {
            Py_DECREF(self);
            return NULL;
        }
        self->_set = NULL;
    }

    return (PyObject *)self;
}
static PyObject*
RadosSchema_remove(RadosSchemaObject *self)
{
    int ret = self->_set->remove();

    return PyLong_FromLong(ret);

}



static int 
RadosSchema_init(RadosSchemaObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *tmp;
    char*  Name;
    char*  PoolName;
    static char *kwlist[] = {"name","pool_name",NULL};
    if (! PyArg_ParseTupleAndKeywords(args, kwds, "ss", kwlist,
                                      &Name, &PoolName)
                                      )
        return -1;

    if (Name && PoolName) {
        tmp = self->_name;
        self->_name = PyString_FromString(Name);
        Py_XDECREF(tmp);

        tmp = self->_poolname;

        self->_poolname = PyString_FromString(PoolName);
        Py_XDECREF(tmp);


     if(!self->_set){
        self->_set= new JRados::JRadosCSchema( Name, PoolName);}
    }

    return 0;
}

static PyObject*  RadosSchema_getSchema(RadosSchemaObject * self)

{

    std::string schema = self->_set->getSchema();
    return (PyObject*) PyString_FromString(schema.c_str());

}

static PyObject*  RadosSchema_exists(RadosSchemaObject * self)

{

    return (PyObject*) PyBool_FromLong(self->_set->exist());

}

static PyObject*  RadosSchema_writeSchema(RadosSchemaObject * self, PyObject* args)
{

    char*  schema;
      if (! PyArg_ParseTuple(args, "s", &schema) )
        return NULL;

    int ret = self->_set->writeSchema(schema);

      return PyLong_FromLong(ret);
}


static PyMemberDef RadosSchema_members[] = {
    {"Name", T_OBJECT_EX, offsetof(RadosSchemaObject, _name), 0,
     "Schema Name"},
    {"PoolName", T_OBJECT_EX, offsetof(RadosSchemaObject, _poolname), 0,
     "Schema Pool Name"},

    {NULL}  /* Sentinel */
};

static PyMethodDef RadosSchema_methods[] = {

{"exist", (PyCFunction)RadosSchema_exists, METH_NOARGS,
     "Test, if Object exist"
    },

{"getSchema", (PyCFunction)RadosSchema_getSchema, METH_NOARGS,
     "get  schema"
    },
{"writeSchema", (PyCFunction)RadosSchema_writeSchema, METH_VARARGS,
     "get  schema"
    },
{"remove", (PyCFunction)RadosSchema_remove, METH_NOARGS,
     "Remove Schema-Object"
    },



 {NULL}
};


static PyTypeObject RadosSchemaType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "radoscache.RadosSchema",             /* tp_name */
    sizeof(RadosSchemaObject), /* tp_basicsize */
    0,                         /* tp_itemsize */
    0,                         /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "Rados-Schema objects",           /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    RadosSchema_methods,             /* tp_methods */
    RadosSchema_members,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)RadosSchema_init,      /* tp_init */
    0,                         /* tp_alloc */
    RadosSchema_new,                 /* tp_new */

};




static PyMethodDef RadosMethods[]
{
 {NULL}
};



static PyObject *RadosError;

PyMODINIT_FUNC
initscirados(void)
{
    PyObject *m;


    if (PyType_Ready(&RadosCacheType) < 0)
        return;
    if (PyType_Ready(&RadosDataSetType) < 0)
        return;
    if (PyType_Ready(&RadosSchemaType) < 0)
        return;


    m = Py_InitModule3("scirados", RadosMethods,  "Rados Cache for saving data");

    if (m == NULL)
        return;

    Py_INCREF(&RadosCacheType);
    Py_INCREF(&RadosDataSetType);
    Py_INCREF(&RadosSchemaType);
    PyModule_AddObject(m, "RadosCache", (PyObject *)&RadosCacheType);
    PyModule_AddObject(m, "RadosDataSet", (PyObject *)&RadosDataSetType);
    PyModule_AddObject(m, "RadosSchema", (PyObject *)&RadosSchemaType);

    RadosError = PyErr_NewException("sciRados.error", NULL, NULL);
    Py_INCREF(RadosError);
    PyModule_AddObject(m, "error", RadosError);

   import_array();
}


#endif
