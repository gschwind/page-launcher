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

#include "xclientmessageevent.hxx"

#include "attr_handler.hxx"

static PyObject * _get_data_long(page_XClientMessageEvent * o) {
	PyObject * list = PyList_New(0);

	for(int i = 0; i < 5; ++i) {
		// TODO: error checking
		PyList_Append(list, PyLong_FromLong(o->ref_event->event.xclient.data.l[i]));
	}

	return list;
}

static PyObject * _get_data_short(page_XClientMessageEvent * o) {
	PyObject * list = PyList_New(0);

	for(int i = 0; i < 10; ++i) {
		// TODO: error checking
		PyList_Append(list, PyLong_FromLong(o->ref_event->event.xclient.data.s[i]));
	}

	return list;
}

static PyObject * _get_data_char(page_XClientMessageEvent * o) {
	PyObject * list = PyList_New(0);

	for(int i = 0; i < 20; ++i) {
		// TODO: error checking
		PyList_Append(list, PyLong_FromLong(o->ref_event->event.xclient.data.b[i]));
	}

	return list;
}

static void _set_data_long(page_XClientMessageEvent * o, PyObject * value) {
	// TODO
}

static void _set_data_short(page_XClientMessageEvent * o, PyObject * value) {
	// TODO
}

static void _set_data_char(page_XClientMessageEvent * o, PyObject * value) {
	// TODO
}

template<>
_attr_handler<page_XClientMessageEvent>::_map_type _attr_handler<page_XClientMessageEvent>::map = {
		DEF_MAP_FUNCTION(XClientMessageEvent, xclient, type),
		DEF_MAP_FUNCTION(XClientMessageEvent, xclient, serial),
		DEF_MAP_FUNCTION(XClientMessageEvent, xclient, send_event),
		DEF_MAP_FUNCTION(XClientMessageEvent, xclient, window),
		DEF_MAP_FUNCTION(XClientMessageEvent, xclient, message_type),
		DEF_MAP_FUNCTION(XClientMessageEvent, xclient, format),
		{"data_as_long", {&_get_data_long, &_set_data_long}},
		{"data_as_short", {&_get_data_short, &_set_data_short}},
		{"data_as_char", {&_get_data_char, &_set_data_char}}
};

static void page_XClientMessageEvent_dealloc(PyObject * self) {
        Py_TYPE(self)->tp_free(self);
}

static PyObject * page_XClientMessageEvent_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
        return type->tp_alloc(type, 0);
}

static int page_XClientMessageEvent_init(PyObject * _self, PyObject *args, PyObject *kwds)
{
	auto self = reinterpret_cast<page_XClientMessageEvent*>(_self);
	PyObject * py_xevent;

	if (!PyArg_ParseTuple(args, "O", &py_xevent))
		return -1;

	if(Py_TYPE(py_xevent) != &page_XEventType)
		return -1;

	Py_INCREF(py_xevent);
	self->ref_event = reinterpret_cast<page_XEvent*>(py_xevent);

	return 0;
}

PyTypeObject page_XClientMessageEventType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "PageLauncherHook.XClientMessageEvent",             /* tp_name */
    sizeof(page_XClientMessageEvent), /* tp_basicsize */
    0,                         /* tp_itemsize */
	page_XClientMessageEvent_dealloc,                         /* tp_dealloc */
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
	_attr_handler<page_XClientMessageEvent>::get_attr,           /* tp_getattro */
	_attr_handler<page_XClientMessageEvent>::set_attr,           /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "PageLauncherHook.XClientMessageEvent objects",           /* tp_doc */
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
    page_XClientMessageEvent_init, /* tp_init */
    0,                         /* tp_alloc */
	page_XClientMessageEvent_new,              /* tp_new */
};

