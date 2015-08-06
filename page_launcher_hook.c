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

static PyObject * g_func;

static GdkFilterReturn call_python_filter (GdkXEvent *xevent, GdkEvent *event, gpointer data) {

	printf("my filter ! %p\n", data);
	PyObject * func = (PyObject*)data;

	PyObject_Print(g_func, stdout, 0);
	printf("\n");

	//PyGObject * pyg_event = pyg_pointer_new(GDK_TYPE_EVENT, event);
	// Segfault PyGObject * pyg_event = pygobject_new(event);
	//PyGObject * pyg_event = pyg_boxed_new(GDK_TYPE_EVENT, event, FALSE, FALSE);
	PyObject* args = Py_BuildValue("(i,i)",20,30);
	PyObject* result = PyObject_CallObject(g_func, args);
	Py_DECREF(args);
	if (result != NULL) {
		PyObject_Print(result, stdout, 0);
		Py_DECREF(result);
	} else {
		printf("error\n");
		PyErr_Print();
		PyErr_Clear();
	}


	//Py_DecRef(args);
	printf("end\n");
	/* TODO */
	return GDK_FILTER_CONTINUE;

}

static PyObject * py_gdk_add_filter(PyObject * self, PyObject * args) {
	PyObject * pyobj; // the GdkWindow
	PyObject * func; // the function to call

	if (!PyArg_ParseTuple(args, "OO", &pyobj, &func))
		return NULL;

    if (!PyCallable_Check(func)) {
        PyErr_SetString(PyExc_TypeError, "parameter must be callable");
        return NULL;
    }

	if(pygobject_check(pyobj, pygobject_lookup_class(GDK_TYPE_WINDOW)) && PyCallable_Check(func)) {
		GdkWindow * window = GDK_WINDOW(pygobject_get(pyobj));
		Py_INCREF(func);
		Py_INCREF(func);
		Py_INCREF(func);
		Py_INCREF(func);
		Py_INCREF(func);
		Py_INCREF(func);
		Py_INCREF(func);

		printf("YYYY %d\n", ((PyObject*)(func))->ob_refcnt);


//		PyObject_Print(func, stdout, 0);
//		printf("\nXXXX %p\n", func);
//		PyObject* result = PyObject_CallFunction(func, "i", 10);
//		printf("return = %p\n", result);
//		if (result != NULL)
//			PyObject_Print(result, stdout, 0);
//		else
//			return NULL;
//
//		Py_DECREF(result);

		g_func = func;
		gdk_window_add_filter(window, &call_python_filter, (gpointer)func);
	} else {
		printf("unknown object\n");
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
	TPL_FUNCTION_DOC(gdk_add_filter, "implement gdk_add_filter"),
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

