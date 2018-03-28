#ifdef HAVE_NUMPY
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/numpyconfig.h>
#include <numpy/arrayobject.h>
#endif
 
#include "rados_cache.h"


#ifdef HAVE_NUMPY

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
    static char *kwlist[] = {"name"};

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
                               out_array= PyArray_SimpleNewFromData(Ndims, int_dims, NPY_FLOAT, (void*)data);
                           }
                           break;






    }
    delete dims;
    delete int_dims;
    return (PyObject*) out_array;
}

static PyMethodDef RadosCache_methods[] = {
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



static PyObject *RadosCacheError;

PyMODINIT_FUNC
initradoscache(void)
{
    PyObject *m;
    RadosCacheType.tp_new = PyType_GenericNew;

    if (PyType_Ready(&RadosCacheType) < 0)
        return;


    m = Py_InitModule3("radoscache", RadosCacheMethods,  "Rados Cache for saving data");

    if (m == NULL)
        return;

    Py_INCREF(&RadosCacheType);
    PyModule_AddObject(m, "RadosCache", (PyObject *)&RadosCacheType);
    RadosCacheError = PyErr_NewException("cephcache.error", NULL, NULL);
    Py_INCREF(RadosCacheError);
    PyModule_AddObject(m, "error", RadosCacheError);

   import_array();
}


#endif
