
#ifdef HAVE_NUMPY
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/numpyconfig.h>
#include <numpy/arrayobject.h>
#endif

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

#if 0

static PyObject *RadosDataSetError;

PyMODINIT_FUNC
initradosdataset(void)
{
    PyObject *m;
    RadosDataSetType.tp_new = PyType_GenericNew;

    if (PyType_Ready(&RadosDataSetType) < 0)
        return;


    m = Py_InitModule3("RadosDataSet", RadosDataSetMethods,  "Rados DataSet for controlling data");

    if (m == NULL)
        return;

    Py_INCREF(&RadosDataSetType);
    PyModule_AddObject(m, "RadosDataSet", (PyObject *)&RadosDataSetType);
    RadosDataSetError = PyErr_NewException("radosDataSet.error", NULL, NULL);
    Py_INCREF(RadosDataSetError);
    PyModule_AddObject(m, "error", RadosDataSetError);

   import_array();
}


#endif
