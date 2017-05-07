/*
 * Copyright (2017) Benoit Gschwind
 *
 * gdkxevent.cxx is part of page-compositor.
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

#include "xdestroywindowevent.hxx"

#include "attr_handler.hxx"

template<>
_attr_handler<page_XDestroyWindowEvent>::_map_type _attr_handler<page_XDestroyWindowEvent>::map = {
		DEF_MAP_FUNCTION(XDestroyWindowEvent, xdestroywindow, type),
		DEF_MAP_FUNCTION(XDestroyWindowEvent, xdestroywindow, serial),
		DEF_MAP_FUNCTION(XDestroyWindowEvent, xdestroywindow, send_event),
		DEF_MAP_FUNCTION(XDestroyWindowEvent, xdestroywindow, event),
		DEF_MAP_FUNCTION(XDestroyWindowEvent, xdestroywindow, window)
};

static void page_XDestroyWindowEvent_dealloc(PyObject * self) {
        Py_TYPE(self)->tp_free(self);
}

static PyObject * page_XDestroyWindowEvent_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
        return type->tp_alloc(type, 0);
}

static int page_XDestroyWindowEvent_init(PyObject * _self, PyObject *args, PyObject *kwds)
{
	auto self = reinterpret_cast<page_XDestroyWindowEvent*>(_self);
	PyObject * py_xevent;

	if (!PyArg_ParseTuple(args, "O", &py_xevent))
		return -1;

	if(Py_TYPE(py_xevent) != &page_XEventType)
		return -1;

	Py_INCREF(py_xevent);
	self->ref_event = reinterpret_cast<page_XEvent*>(py_xevent);

	return 0;
}

PyTypeObject page_XDestroyWindowEventType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "PageLauncherHook.XDestroyWindowEvent",             /* tp_name */
    sizeof(page_XDestroyWindowEvent), /* tp_basicsize */
    0,                         /* tp_itemsize */
	page_XDestroyWindowEvent_dealloc,                         /* tp_dealloc */
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
	_attr_handler<page_XDestroyWindowEvent>::get_attr,           /* tp_getattro */
	_attr_handler<page_XDestroyWindowEvent>::set_attr,           /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "PageLauncherHook.XDestroyWindowEvent objects",           /* tp_doc */
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
    page_XDestroyWindowEvent_init, /* tp_init */
    0,                         /* tp_alloc */
	page_XDestroyWindowEvent_new,              /* tp_new */
};

