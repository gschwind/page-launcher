#include <Python.h>
#include <pygobject.h>
#include <clutter/clutter.h>
#include <gdk/gdk.h>

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

static PyObject * py_print_type(PyObject * self, PyObject * args) {
	PyObject * pyobj;
	if (!PyArg_ParseTuple(args, "O", &pyobj))
		return NULL;

	if(pygobject_check(pyobj, pygobject_lookup_class(CLUTTER_TYPE_STAGE))) {
		GObject * obj = pygobject_get(pyobj);
		GType t = G_OBJECT_TYPE(obj);
		printf("type = %s\n", g_type_name(t));
	} else {
		printf("unknown object\n");
	}

	Py_RETURN_NONE;
}

static PyObject * py_set_strut(PyObject * self, PyObject * args) {
	PyObject * py_gdk_window;
	PyObject * py_list;
	if (!PyArg_ParseTuple(args, "OO", &py_gdk_window, &py_list))
		return NULL;

	if(!pygobject_check(py_gdk_window, pygobject_lookup_class(GDK_TYPE_WINDOW))) {
		return NULL;
	}

	if(!PyList_Check(py_list)) {
		return NULL;
	}

	if (PyList_Size(py_list) == 4) {
		/* if _net_wm_strut */
		long strut[4];
		for (int i = 0; i < 4; ++i) {
			PyObject * v = PyList_GetItem(py_list, i);
			if (PyLong_Check(v)) {
				strut[i] = PyLong_AsLong(v);
			} else {
				return NULL;
			}
		}

		GdkAtom cardinal = gdk_atom_intern("CARDINAL", FALSE);
		GdkAtom _net_wm_strut = gdk_atom_intern("_NET_WM_STRUT", FALSE);

		gdk_property_change(GDK_WINDOW(pygobject_get(py_gdk_window)),
				_net_wm_strut, cardinal, 32, GDK_PROP_MODE_REPLACE,
				(guchar*) strut, 4);
	} else if (PyList_Size(py_list) == 12) {
		/* if _net_wm_strut_partial */

		long strut_partial[12];
		for (int i = 0; i < 12; ++i) {
			PyObject * v = PyList_GetItem(py_list, i);
			if (PyLong_Check(v)) {
				strut_partial[i] = PyLong_AsLong(v);
			} else {
				return NULL;
			}
		}

		GdkAtom cardinal = gdk_atom_intern("CARDINAL", FALSE);
		GdkAtom _net_wm_strut_partial = gdk_atom_intern("_NET_WM_STRUT_PARTIAL", FALSE);

		gdk_property_change(GDK_WINDOW(pygobject_get(py_gdk_window)),
				_net_wm_strut_partial, cardinal, 32, GDK_PROP_MODE_REPLACE,
				(guchar*) strut_partial, 12);
	} else {
		/* else error */
		return NULL;
	}

	Py_RETURN_NONE;
}

#define TPL_FUNCTION(name) {#name, py_##name, METH_VARARGS, "Not documented"}
#define TPL_FUNCTION_DOC(name, doc) {#name, py_##name, METH_VARARGS, doc}

static PyMethodDef methods[] =
{
	TPL_FUNCTION_DOC(print, "simple print for testing"),
	TPL_FUNCTION_DOC(print_type, "Print the type of a GObject"),
	TPL_FUNCTION_DOC(set_strut, "set _NET_WM_STRUT"),
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

	pygobject_init(-1, -1, -1);

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

