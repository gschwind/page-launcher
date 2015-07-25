
#include <Python.h>
#include <pygobject.h>

#include <stdio.h>

//#include <numpy/arrayobject.h>
//#include <numpy/npy_common.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct module_state {
	PyObject * Error;
};

static PyObject * py_print(PyObject * self, PyObject * args) {
	char * str;
	if (!PyArg_ParseTuple(args, "s", &str))
		return NULL;
	printf("%s\n", str);

	Py_RETURN_NONE;
}

#define TPL_FUNCTION(name) {#name, py_##name, METH_VARARGS, "Not documented"}
#define TPL_FUNCTION_DOC(name, doc) {#name, py_##name, METH_VARARGS, doc}

static PyMethodDef methods[] =
{
	TPL_FUNCTION_DOC(print, "simple print for testing"),
	{NULL, NULL, 0, NULL} // sentinel
};

static int PageLauncherHook_traverse(PyObject *m, visitproc visit, void *arg) {
	Py_VISIT(((struct module_state*)PyModule_GetState(m))->Error);
	return 0;
}

static int PageLauncherHook_clear(PyObject *m) {
	Py_CLEAR(((struct module_state*)PyModule_GetState(m))->Error);
	return 0;
}

static struct PyModuleDef moduledef = {
	PyModuleDef_HEAD_INIT,
	"PageLauncherHook",
	NULL,
	sizeof(struct module_state),
	methods,
	NULL,
	PageLauncherHook_traverse,
	PageLauncherHook_clear,
	NULL
};

PyMODINIT_FUNC
PyInit_PageLauncherHook(void)
{

	PyObject * m;
	m = PyModule_Create(&moduledef);
	if (m == NULL)
		return NULL;

	/* create the PageLauncherHook exception */
	((struct module_state*)PyModule_GetState(m))->Error = PyErr_NewException("PageLauncherHook.error", NULL, NULL);
	Py_INCREF(((struct module_state*)PyModule_GetState(m))->Error);
	PyModule_AddObject(m, "error", ((struct module_state*)PyModule_GetState(m))->Error);

        
	/** init numpy **/
	//import_array();

	return m;

}

