/*
 * Copyright (2017) Benoit Gschwind
 *
 * gdkxevent.hxx is part of page-compositor.
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

#ifndef GDKXEVENT_XANYEVENT_HXX_
#define GDKXEVENT_XANYEVENT_HXX_

#include <Python.h>
#include <X11/Xlib.h>

#include "xevent.hxx"

typedef struct {
    PyObject_HEAD
	page_XEvent * ref_event;
} page_XAnyEvent;

extern PyTypeObject page_XAnyEventType;

#endif /* GDKXEVENT_XANYEVENT_HXX_ */
