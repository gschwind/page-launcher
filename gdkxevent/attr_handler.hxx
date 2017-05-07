/*
 * Copyright (2017) Benoit Gschwind
 *
 * attr_handler.hxx is part of page-compositor.
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

#ifndef ATTR_HANDLER_HXX_
#define ATTR_HANDLER_HXX_

#include <Python.h>

#include <map>
#include <functional>

template<typename T>
struct identity { using type = T; };

template<typename T0>
PyObject * _get_attr(T0 * o, char identity<T0>::type::*member) {
	return PyLong_FromLong(o->*member);
}

template<typename T0>
PyObject * _get_attr(T0 * o, short identity<T0>::type::*member) {
	return PyLong_FromLong(o->*member);
}

template<typename T0>
PyObject * _get_attr(T0 * o, int identity<T0>::type::*member) {
	return PyLong_FromLong(o->*member);
}

template<typename T0>
PyObject * _get_attr(T0 * o, long identity<T0>::type::*member) {
	return PyLong_FromLong(o->*member);
}

template<typename T0>
PyObject * _get_attr(T0 * o, long long identity<T0>::type::*member) {
	return PyLong_FromLong(o->*member);
}

template<typename T0>
PyObject * _get_attr(T0 * o, unsigned char identity<T0>::type::*member) {
	return PyLong_FromUnsignedLong(o->*member);
}

template<typename T0>
PyObject * _get_attr(T0 * o, unsigned short identity<T0>::type::*member) {
	return PyLong_FromUnsignedLong(o->*member);
}

template<typename T0>
PyObject * _get_attr(T0 * o, unsigned int identity<T0>::type::*member) {
	return PyLong_FromUnsignedLong(o->*member);
}

template<typename T0>
PyObject * _get_attr(T0 * o, unsigned long identity<T0>::type::*member) {
	return PyLong_FromUnsignedLong(o->*member);
}

template<typename T0>
PyObject * _get_attr(T0 * o, unsigned long long identity<T0>::type::*member) {
	return PyLong_FromUnsignedLong(o->*member);
}

template<typename T0>
PyObject * _get_attr(T0 * o, float identity<T0>::type::*member) {
	return PyFloat_FromDouble(o->*member);
}

template<typename T0>
PyObject * _get_attr(T0 * o, double identity<T0>::type::*member) {
	return PyFloat_FromDouble(o->*member);
}

template<typename T0>
PyObject * _get_attr(T0 * o, long double identity<T0>::type::*member) {
	return PyFloat_FromDouble(o->*member);
}

template<typename T0>
void _set_attr(T0 * o, char identity<T0>::type::*member, PyObject * value) {
	if(PyLong_Check(value)) {
		o->*member = PyLong_AsLong(value);
	}
}

template<typename T0>
void _set_attr(T0 * o, short identity<T0>::type::*member, PyObject * value) {
	if(PyLong_Check(value)) {
		o->*member = PyLong_AsLong(value);
	}
}

template<typename T0>
void _set_attr(T0 * o, int identity<T0>::type::*member, PyObject * value) {
	if(PyLong_Check(value)) {
		o->*member = PyLong_AsLong(value);
	}
}

template<typename T0>
void _set_attr(T0 * o, long identity<T0>::type::*member, PyObject * value) {
	if(PyLong_Check(value)) {
		o->*member = PyLong_AsLong(value);
	}
}

template<typename T0>
void _set_attr(T0 * o, long long identity<T0>::type::*member, PyObject * value) {
	if(PyLong_Check(value)) {
		o->*member = PyLong_AsLong(value);
	}
}

template<typename T0>
void _set_attr(T0 * o, unsigned char identity<T0>::type::*member, PyObject * value) {
	if(PyLong_Check(value)) {
		o->*member = PyLong_AsUnsignedLong(value);
	}
}

template<typename T0>
void _set_attr(T0 * o, unsigned short identity<T0>::type::*member, PyObject * value) {
	if(PyLong_Check(value)) {
		o->*member = PyLong_AsUnsignedLong(value);
	}
}

template<typename T0>
void _set_attr(T0 * o, unsigned int identity<T0>::type::*member, PyObject * value) {
	if(PyLong_Check(value)) {
		o->*member = PyLong_AsUnsignedLong(value);
	}
}

template<typename T0>
void _set_attr(T0 * o, unsigned long identity<T0>::type::*member, PyObject * value) {
	if(PyLong_Check(value)) {
		o->*member = PyLong_AsUnsignedLong(value);
	}
}

template<typename T0>
void _set_attr(T0 * o, unsigned long long identity<T0>::type::*member, PyObject * value) {
	if(PyLong_Check(value)) {
		o->*member = PyLong_AsUnsignedLong(value);
	}
}

template<typename T>
struct _attr_handler {

	using _get_attr_type = PyObject * (*)(T *);
	using _set_attr_type = void (*)(T *, PyObject *);

	struct _func_handler {
		_get_attr_type get;
		_set_attr_type set;
		_func_handler(_get_attr_type get, _set_attr_type set) : get{get}, set{set} { }
	};

	using _map_type = std::map<std::string, _func_handler>;

	static _map_type map;

	static PyObject * get_attr(PyObject * o, PyObject *attr) {
		char * s = PyUnicode_AsUTF8(attr);
		auto x = _attr_handler::map.find(s);
		if(x != _attr_handler::map.end()) {
			return (x->second.get)(reinterpret_cast<T*>(o));
		}
		return NULL;
	}

	static int set_attr(PyObject * o, PyObject *attr, PyObject * value) {
		if(value == NULL)
			return 1;

		char * s = PyUnicode_AsUTF8(attr);
		auto x = _attr_handler::map.find(s);
		if(x != _attr_handler::map.end()) {
			(x->second.set)(reinterpret_cast<T*>(o), value);
			return 1;
		}
		return 0;
	}

};

//#define DEF_MAP_FUNCTION(type, member, name) \
//	std::pair<std::string, _attr_handler<page_##type>::_func_handler>(#name, \
//			_attr_handler<page_##type>::_func_handler([] (page_##type * o) -> PyObject * { return _get_attr(&(o->ref_event->event.member), &type::name); }, \
//		  [] (page_##type * o, PyObject * v) -> void { _set_attr(&(o->ref_event->event.member), &type::name, v); } ))

#define DEF_MAP_FUNCTION(type, member, name) \
	{#name, \
	{[] (page_##type * o) -> PyObject * { return _get_attr(&(o->ref_event->event.member), &type::name); }, \
	 [] (page_##type * o, PyObject * v) -> void { _set_attr(&(o->ref_event->event.member), &type::name, v); } } }


#endif /* ATTR_HANDLER_HXX_ */
