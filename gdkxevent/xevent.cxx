/*
 * Copyright (2017) Benoit Gschwind
 *
 * xevent.cxx is part of page-compositor.
 *
 * page-compositor is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * page-compositor is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with page-compositor.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "xevent.hxx"

#include "xanyevent.hxx"
#include "xclientmessageevent.hxx"

#include "attr_handler.hxx"

static void page_XEvent_dealloc(PyObject * self) {
        Py_TYPE(self)->tp_free(self);
}

static PyObject * page_XEvent_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
        return type->tp_alloc(type, 0);
}

static int page_XEvent_init(PyObject * _self, PyObject *args, PyObject *kwds) {
	auto self = reinterpret_cast<page_XEvent*>(_self);

	return 0;
}

static PyObject * xevent_get_attr(PyObject* o, PyObject * _attr) {
	std::string attr = PyUnicode_AsUTF8(_attr);

	if (attr == "XAnyEvent") {
		return PyObject_CallFunctionObjArgs(reinterpret_cast<PyObject*>(&page_XAnyEventType), o, NULL);
	} else if (attr == "XClientMessageEvent") {
		return PyObject_CallFunctionObjArgs(reinterpret_cast<PyObject*>(&page_XClientMessageEventType), o, NULL);
	}

	return NULL;
}

static int xevent_set_attr(PyObject* o, PyObject * _attr, PyObject * value) {
	std::string attr = PyUnicode_AsUTF8(_attr);

	if(attr == "XAnyEvent") {
		// TODO
	} else if (attr == "XClientMessageEvent") {

	}

	return 0;
}

PyTypeObject page_XEventType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "PageLauncherHook.XEvent",             /* tp_name */
    sizeof(page_XEvent), /* tp_basicsize */
    0,                         /* tp_itemsize */
	page_XEvent_dealloc,                         /* tp_dealloc */
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
	xevent_get_attr,           /* tp_getattro */
	xevent_set_attr,           /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "PageLauncherHook.XEvent objects",           /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    0,   /* tp_methods */
    0,   /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    page_XEvent_init, /* tp_init */
    0,                         /* tp_alloc */
	page_XEvent_new,              /* tp_new */

};

