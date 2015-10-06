#include <Python.h>
#include <pygobject.h>
#include <clutter/clutter.h>
#include <gdk/gdk.h>
#include <X11/Xlib.h>
#include <gdk/gdkx.h>
#include <stdio.h>

//#include <numpy/arrayobject.h>
//#include <numpy/npy_common.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>


PyObject * panel=NULL; // the function to call

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



#define SYSTEM_TRAY_REQUEST_DOCK    0
#define SYSTEM_TRAY_BEGIN_MESSAGE   1
#define SYSTEM_TRAY_CANCEL_MESSAGE  2

#define SYSTEM_TRAY_ORIENTATION_HORZ 0
#define SYSTEM_TRAY_ORIENTATION_VERT 1

static PyObject * py_set_system_tray_orientation(PyObject * self, PyObject * args) {
	PyObject * py_gdk_window;
	PyObject * py_vert;
	if (!PyArg_ParseTuple(args, "OO", &py_gdk_window, &py_vert))
		return NULL;

	if(!pygobject_check(py_gdk_window, pygobject_lookup_class(GDK_TYPE_WINDOW))) {
		return NULL;
	}

	if(!PyBool_Check(py_vert)) {
		return NULL;
	}
	long data[1];
	data[0] = PyObject_IsTrue(py_vert) ? SYSTEM_TRAY_ORIENTATION_VERT : SYSTEM_TRAY_ORIENTATION_HORZ;

	GdkAtom cardinal = gdk_atom_intern("CARDINAL", FALSE);
	GdkAtom _net_system_tray_orientation = gdk_atom_intern("_NET_SYSTEM_TRAY_ORIENTATION", FALSE);

	gdk_property_change(GDK_WINDOW(pygobject_get(py_gdk_window)),
				_net_system_tray_orientation, cardinal, 32, GDK_PROP_MODE_REPLACE,
				(guchar*) data, 1);
	Py_RETURN_NONE;
}

static PyObject * py_set_system_tray_visual(PyObject * self, PyObject * args) {
	PyObject * pyobj_win; // the GdkWindow
	PyObject * pyobj_display; // the GdkWindow

	if (!PyArg_ParseTuple(args, "OO", &pyobj_win, &pyobj_display)) {
		return NULL;
	}

	if(!pygobject_check(pyobj_win, pygobject_lookup_class(GDK_TYPE_WINDOW))) {
		PyErr_SetString(PyExc_TypeError, "parameter must be A GdkWindow");
	}
	if(!pygobject_check(pyobj_display, pygobject_lookup_class(GDK_TYPE_DISPLAY))) {
		PyErr_SetString(PyExc_TypeError, "parameter must be A GdkDisplay");
	}

	GdkWindow * window = GDK_WINDOW(pygobject_get(pyobj_win));
	GdkDisplay * display = GDK_DISPLAY(pygobject_get(pyobj_display));
	

	long data[1];
	{
		XWindowAttributes window_attributes;
		XGetWindowAttributes(GDK_DISPLAY_XDISPLAY(display), gdk_x11_window_get_xid(window), &window_attributes);
		data[0] = XVisualIDFromVisual(window_attributes.visual);
	}

	GdkAtom visualid = gdk_atom_intern("VISUALID", FALSE);
	GdkAtom _net_system_tray_visual = gdk_atom_intern("_NET_SYSTEM_TRAY_VISUAL", FALSE);

	gdk_property_change(GDK_WINDOW(pygobject_get(pyobj_win)),
				_net_system_tray_visual, visualid, 32, GDK_PROP_MODE_REPLACE,
				(guchar*) data, 1);
	Py_RETURN_NONE;
}

static void
tray_manager_handle_dock_request (PyObject       *manager,
				      XClientMessageEvent  *xevent)
{
//printf("%s\n",__FUNCTION__);
PyObject * func = PyObject_GetAttrString(manager, "dock_request");
	if(!func) {
		PyErr_SetString(PyExc_TypeError, "dock_request callback does not exist");
		return;
	}

	if (!PyCallable_Check(func)) {
			PyErr_SetString(PyExc_TypeError, "dock_request must be callable");
		return;
		}




	PyObject* args = Py_BuildValue("(i,i)",xevent->data.l[2], GINT_TO_POINTER (xevent->window));	
 	PyObject_CallObject(func, args);
	Py_DECREF(args);
}

static void
tray_manager_handle_message_data (PyObject       *manager,
				       XClientMessageEvent  *xevent)
{
printf("FIXME %s\n",__FUNCTION__);
#if 0
  GList *p;
  int len;
  
  /* Try to see if we can find the
   * pending message in the list
   */
  for (p = manager->messages; p; p = p->next)
    {
      PendingMessage *msg = p->data;

      if (xevent->window == msg->window)
	{
	  /* Append the message */
	  len = MIN (msg->remaining_len, 20);

	  memcpy ((msg->str + msg->len - msg->remaining_len),
		  &xevent->data, len);
	  msg->remaining_len -= len;

	  if (msg->remaining_len == 0)
	    {
	      GtkSocket *socket;

	      socket = g_hash_table_lookup (manager->socket_table, GINT_TO_POINTER (msg->window));

	      if (socket)
		{
		  g_signal_emit (manager, manager_signals[MESSAGE_SENT], 0,
				 socket, msg->str, msg->id, msg->timeout);
		}
	      manager->messages = g_list_remove_link (manager->messages,
						      p);
	      
	      pending_message_free (msg);
	    }

	  return;
	}
    }
#endif
}

static void
tray_manager_handle_begin_message (PyObject       *manager,
				       XClientMessageEvent  *xevent)
{
printf("FIXME %s\n",__FUNCTION__);
#if 0
  GList *p;
  PendingMessage *msg;

  /* Check if the same message is
   * already in the queue and remove it if so
   */
  for (p = manager->messages; p; p = p->next)
    {
      PendingMessage *msg = p->data;

      if (xevent->window == msg->window &&
	  xevent->data.l[4] == msg->id)
	{
	  /* Hmm, we found it, now remove it */
	  pending_message_free (msg);
	  manager->messages = g_list_remove_link (manager->messages, p);
	  break;
	}
    }

  /* Now add the new message to the queue */
  msg = g_new0 (PendingMessage, 1);
  msg->window = xevent->window;
  msg->timeout = xevent->data.l[2];
  msg->len = xevent->data.l[3];
  msg->id = xevent->data.l[4];
  msg->remaining_len = msg->len;
  msg->str = g_malloc (msg->len + 1);
  msg->str[msg->len] = '\0';
  manager->messages = g_list_prepend (manager->messages, msg);
#endif
}

static void
tray_manager_handle_cancel_message (PyObject       *manager,
					XClientMessageEvent  *xevent)
{
	printf("%s\n",__FUNCTION__);
	PyObject * func = PyObject_GetAttrString(manager, "message_cancel");
	if(!func) {
		PyErr_SetString(PyExc_TypeError, "message_cancel callback does not exist");
return;
	}

	if (!PyCallable_Check(func)) {
			PyErr_SetString(PyExc_TypeError, "message_cancel must be callable");
			return;
		}

	

	PyObject* args = Py_BuildValue("(i,i)",xevent->data.l[2], GINT_TO_POINTER (xevent->window));	
 	PyObject_CallObject(func, args);
	Py_DECREF(args);
#if 0
  GtkSocket *socket;
  
  socket = g_hash_table_lookup (manager->socket_table, GINT_TO_POINTER (xevent->window));
  
  if (socket)
    {
      g_signal_emit (manager, manager_signals[MESSAGE_CANCELLED], 0,
		     socket, xevent->data.l[2]);
    }
#endif
}


static GdkFilterReturn
tray_manager_handle_event (PyObject       *manager,
			       XClientMessageEvent  *xevent)
{
  switch (xevent->data.l[1])
    {
    case SYSTEM_TRAY_REQUEST_DOCK:
      tray_manager_handle_dock_request (manager, xevent);
      return GDK_FILTER_REMOVE;

    case SYSTEM_TRAY_BEGIN_MESSAGE:
      tray_manager_handle_begin_message (manager, xevent);
      return GDK_FILTER_REMOVE;

    case SYSTEM_TRAY_CANCEL_MESSAGE:
      tray_manager_handle_cancel_message (manager, xevent);
      return GDK_FILTER_REMOVE;
    default:
      break;
    }

  return GDK_FILTER_CONTINUE;
}

static void
tray_manager_unmanage (PyObject *manager) {
printf("FIXME %s\n",__FUNCTION__);
}
static void
tray_undock (PyObject *manager, XDestroyWindowEvent * xevent) {
printf("%s\n",__FUNCTION__);

PyObject * func = PyObject_GetAttrString(manager, "undock_request");
	if(!func) {
		PyErr_SetString(PyExc_TypeError, "undock_request callback does not exist");
		return;
	}

	if (!PyCallable_Check(func)) {
			PyErr_SetString(PyExc_TypeError, "undock_request must be callable");
		return;
		}


	PyObject* args = Py_BuildValue("(i,i)",xevent->event, GINT_TO_POINTER (xevent->window));	
 	PyObject_CallObject(func, args);
	Py_DECREF(args);
}


static GdkFilterReturn call_python_filter_inter (GdkXEvent *gdkxevent, GdkEvent *event, gpointer data) {
	XEvent *xevent = (GdkXEvent *)gdkxevent;
	PyObject *manager = data;
	GdkFilterReturn retval = GDK_FILTER_CONTINUE;

//printf("%s\n",__FUNCTION__);

	if (xevent->type == Expose) {
		XExposeEvent * xexposeevent = (XExposeEvent *) xevent; 
		printf("Client Expose %d\n", xexposeevent->window);

{
//GdkWindow * wininter = gdk_x11_window_foreign_new_for_display(display, win_inter);
	//cairo_t * cr = gdk_cairo_create (wininter);
	Window win_inter = xexposeevent->window;
	Display * xdisplay = xexposeevent->display;
	XWindowAttributes window_inter_attributes;
	XGetWindowAttributes(xdisplay, win_inter, &window_inter_attributes);


GdkWindow * wininter = gdk_x11_window_foreign_new_for_display(gdk_x11_lookup_xdisplay (xdisplay), win_inter);
	gdk_window_resize(wininter,64,64);
	cairo_t * cr = gdk_cairo_create (wininter);
//	cairo_t * cr = cairo_xlib_surface_create(xdisplay, win_inter, window_inter_attributes.visual ,32,32); 
	if(cr) {
//	cairo_xlib_surface_set_size(cr, 32, 32);
 	cairo_set_operator(cr, CAIRO_OPERATOR_SOURCE);
 	cairo_set_source_rgba(cr, 1, 0, 0, 1);
 	cairo_paint(cr);
	//cairo_rectangle(cr, 0, 0, 32, 32);
    //cairo_fill(cr);
	cairo_destroy(cr);
 	} else {
		printf("Unable to create cairo surface",cr);
	}    
}

	}

 return retval;
}

static GdkFilterReturn call_python_filter (GdkXEvent *gdkxevent, GdkEvent *event, gpointer data) {
  XEvent *xevent = (GdkXEvent *)gdkxevent;
  PyObject *manager = data;
  GdkFilterReturn retval = GDK_FILTER_CONTINUE;

  	PyGILState_STATE gstate;
	gstate = PyGILState_Ensure();

//printf("%s\n",__FUNCTION__);

  if (xevent->type == ClientMessage)
    {
	printf("Client Message\n");

	PyObject * py_opcode = PyObject_GetAttrString(manager, "atom_opcode");
	PyObject * py_msg_data = PyObject_GetAttrString(manager, "atom_message_data");
	
	printf("Client Message %p %p\n", py_opcode, py_msg_data);

	long opcode_atom = PyLong_AsLong(py_opcode);
	long message_data_atom = PyLong_AsLong(py_msg_data);	
      if (xevent->xclient.message_type == opcode_atom)
	{
	  retval =  tray_manager_handle_event (manager, (XClientMessageEvent *)xevent);
	}
      else if (xevent->xclient.message_type == message_data_atom)
	{
	  tray_manager_handle_message_data (manager, (XClientMessageEvent *)xevent);
	  retval = GDK_FILTER_REMOVE;
	}
    }
  else if (xevent->type == SelectionClear)
    {
      tray_manager_unmanage (manager);
    }
  else if (xevent->type == DestroyNotify)
    {
      tray_undock (manager, (XDestroyWindowEvent *)xevent);
    }
	else {
	//printf("Not handled %d\n",xevent->type);
}
 

   	 PyGILState_Release(gstate);
  return retval;
}

static
int
error_handler(Display *display, XErrorEvent *error) {
  char buffer[500];

  XGetErrorText(display, error->error_code, buffer, sizeof(buffer));
  printf("%d:%s\n",error->error_code, buffer);

  return 0;
}

static PyObject * py_set_system_tray_filter(PyObject * self, PyObject * args) {
	PyObject * pyobj_win; // the GdkWindow
	PyObject * pyobj_display; // the GdkWindow

	
	if (!PyArg_ParseTuple(args, "OOO", &pyobj_win, &pyobj_display, &panel)) {
		return NULL;
	}

	if(!pygobject_check(pyobj_win, pygobject_lookup_class(GDK_TYPE_WINDOW))) {
		PyErr_SetString(PyExc_TypeError, "parameter must be A GdkWindow");
	}
	if(!pygobject_check(pyobj_display, pygobject_lookup_class(GDK_TYPE_DISPLAY))) {
		PyErr_SetString(PyExc_TypeError, "parameter must be A GdkDisplay");
	}


	GdkWindow * window = GDK_WINDOW(pygobject_get(pyobj_win));
	GdkDisplay * display = GDK_DISPLAY(pygobject_get(pyobj_display));
	guint32 timestamp = gdk_x11_get_server_time (window);


	GdkScreen * screen = gdk_display_get_default_screen (display);
	char * selection_atom_name = g_strdup_printf ("_NET_SYSTEM_TRAY_S%d",
					 gdk_screen_get_number (screen));


	GdkAtom selection_atom = gdk_atom_intern (selection_atom_name, FALSE);
	g_free (selection_atom_name);

	XSetErrorHandler(error_handler);
	
	/* Update Tray Manager */
	{
		Screen * xscreen = GDK_SCREEN_XSCREEN (screen);
		XClientMessageEvent xev;
	      
	      	xev.type = ClientMessage;
	      	xev.window = RootWindowOfScreen (xscreen);
	      	xev.message_type = gdk_x11_get_xatom_by_name_for_display (display,
			                                                "MANAGER");
	      	xev.format = 32;
	      	xev.data.l[0] = timestamp;
	      	xev.data.l[1] = gdk_x11_atom_to_xatom_for_display (display,
			                                         selection_atom);
	      	xev.data.l[2] = gdk_x11_window_get_xid(window);
	      	xev.data.l[3] = 0;	/* manager specific data */
	      	xev.data.l[4] = 0;	/* manager specific data */

	      	XSendEvent (GDK_DISPLAY_XDISPLAY (display),
			  RootWindowOfScreen (xscreen),
			  False, StructureNotifyMask, (XEvent *)&xev);
	}	

	/* Set filtering */
	Py_INCREF(panel);
	XSelectInput(GDK_DISPLAY_XDISPLAY (display), gdk_x11_window_get_xid(window), SubstructureNotifyMask);
	gdk_window_add_filter(window, &call_python_filter, (gpointer)panel);

	Py_RETURN_NONE;
}

/* XEMBED messages */
#define XEMBED_EMBEDDED_NOTIFY          0
#define XEMBED_WINDOW_ACTIVATE          1
#define XEMBED_WINDOW_DEACTIVATE        2
#define XEMBED_REQUEST_FOCUS            3
#define XEMBED_FOCUS_IN                 4
#define XEMBED_FOCUS_OUT                5
#define XEMBED_FOCUS_NEXT               6   
#define XEMBED_FOCUS_PREV               7
#define XEMBED_MODALITY_ON              10
#define XEMBED_MODALITY_OFF             11
#define XEMBED_REGISTER_ACCELERATOR     12
#define XEMBED_UNREGISTER_ACCELERATOR   13
#define XEMBED_ACTIVATE_ACCELERATOR     14

/* Details for XEMBED_FOCUS_IN */
#define XEMBED_FOCUS_CURRENT            0
#define XEMBED_FOCUS_FIRST              1
#define XEMBED_FOCUS_LAST               2



//void xembed_send(GdkDisplay * display, uint32_t win_id, uint32_t in1, uint32_t in2=0, uint32_t in3=0, uint32_t in4=0);

void xembed_send(GdkDisplay * display, uint32_t win_id, uint32_t in1, uint32_t in2, uint32_t in3, uint32_t in4)
{

	  XEvent ev;
		ev.xclient.type = ClientMessage;
		ev.xclient.window = win_id;
		ev.xclient.message_type = gdk_x11_get_xatom_by_name_for_display (display,
			                                                "_XEMBED");
		ev.xclient.format = 32;
		ev.xclient.data.l[0] = CurrentTime;
		ev.xclient.data.l[1] = in1;
		ev.xclient.data.l[2] = in2;
		ev.xclient.data.l[3] = in3;
		ev.xclient.data.l[4] = in4;



			XSendEvent (GDK_DISPLAY_XDISPLAY (display),
				  win_id,
				  False, NoEventMask, (XEvent *)&ev);
	}

static PyObject * py_move_tray(PyObject * self, PyObject * args) {
	PyObject * py_gdk_window;
	PyObject * pyobj_display; // the GdkWindow
	PyObject * py_win_id;
	PyObject * py_win_inter;
	PyObject * py_x;
	PyObject * py_y;
	PyObject * py_sz_x;
	PyObject * py_sz_y;

	if (!PyArg_ParseTuple(args, "OOOOOOOO", &py_gdk_window, &pyobj_display, &py_win_id, &py_win_inter, &py_x, &py_y, &py_sz_x, &py_sz_y))
		return NULL;

	if(!pygobject_check(py_gdk_window, pygobject_lookup_class(GDK_TYPE_WINDOW))) {
		PyErr_SetString(PyExc_TypeError, "parameter must be A Windo");
		return NULL;
	}

	if(!pygobject_check(pyobj_display, pygobject_lookup_class(GDK_TYPE_DISPLAY))) {
		PyErr_SetString(PyExc_TypeError, "parameter must be A GdkDisplay");
		return NULL;
	}

	if(!PyLong_Check(py_win_id)) {
		PyErr_SetString(PyExc_TypeError, "win_id parameter must be A Long");
		return NULL;
	}

	if(!PyLong_Check(py_win_inter)) {
		PyErr_SetString(PyExc_TypeError, "win_inter parameter must be A Long");
		return NULL;
	}

	if(!PyLong_Check(py_x)) {
		PyErr_SetString(PyExc_TypeError, "x parameter must be A Long");
		return NULL;
	}

	if(!PyLong_Check(py_y)) {
		PyErr_SetString(PyExc_TypeError, "y parameter must be A Long");
		return NULL;
	}

	if(!PyLong_Check(py_sz_x)) {
		PyErr_SetString(PyExc_TypeError, "sz_x parameter must be A Long");
		return NULL;
	}

	if(!PyLong_Check(py_sz_y)) {
		PyErr_SetString(PyExc_TypeError, "sz_y parameter must be A Long");
		return NULL;
	}
	long win_id = PyLong_AsLong(py_win_id);
	long win_inter = PyLong_AsLong(py_win_inter);
	long x = PyLong_AsLong(py_x);
	long y = PyLong_AsLong(py_y);
	long sz_x = PyLong_AsLong(py_sz_x);
	long sz_y = PyLong_AsLong(py_sz_y);
	GdkDisplay * display = GDK_DISPLAY(pygobject_get(pyobj_display));
	Display * xdisplay = GDK_DISPLAY_XDISPLAY (display);
	GdkScreen * screen = gdk_display_get_default_screen (display);
	Screen * xscreen = GDK_SCREEN_XSCREEN (screen);

printf("%d id:%d inter:%d\n",__LINE__, win_id, win_inter);
XMoveResizeWindow(xdisplay, win_id, 0,0,  sz_x, sz_y);
printf("%d\n",__LINE__);
XMoveResizeWindow(xdisplay,  win_inter, x, y, sz_x, sz_y);
printf("%d\n",__LINE__);
	//XResizeWindow(GDK_DISPLAY_XDISPLAY (display),  win_id, 1, 1);
	printf("%d\n",__LINE__);
	XMapWindow(xdisplay,  win_id);
   XMapWindow(xdisplay, win_inter); 



#if 1
{

	GdkWindow * wininter = gdk_x11_window_foreign_new_for_display(display, win_inter);
#if 1
	cairo_t * cr = gdk_cairo_create (wininter);
 	cairo_set_operator(cr, CAIRO_OPERATOR_SOURCE);
 	cairo_set_source_rgba(cr, 1, 0, 1, 0);
 	cairo_paint(cr);
	cairo_destroy(cr);    
#endif
	}
#endif

	printf("%d\n",__LINE__);

	Py_RETURN_NONE;
}

static PyObject * py_undock_tray(PyObject * self, PyObject * args) {
	PyObject * py_gdk_window;
	PyObject * pyobj_display; // the GdkWindow
	PyObject * py_win_id;
	PyObject * py_win_inter;

	if (!PyArg_ParseTuple(args, "OOOO", &py_gdk_window, &pyobj_display, &py_win_id, &py_win_inter))
		return NULL;

	if(!pygobject_check(py_gdk_window, pygobject_lookup_class(GDK_TYPE_WINDOW))) {
		PyErr_SetString(PyExc_TypeError, "parameter must be A Windo");
		return NULL;
	}

	if(!pygobject_check(pyobj_display, pygobject_lookup_class(GDK_TYPE_DISPLAY))) {
		PyErr_SetString(PyExc_TypeError, "parameter must be A GdkDisplay");
		return NULL;
	}

	if(!PyLong_Check(py_win_id)) {
		PyErr_SetString(PyExc_TypeError, "win_id parameter must be A Long");
		return NULL;
	}

	if(!PyLong_Check(py_win_inter)) {
		PyErr_SetString(PyExc_TypeError, "win_inter parameter must be A Long");
		return NULL;
	}

	long win_id = PyLong_AsLong(py_win_id);
	long win_inter = PyLong_AsLong(py_win_inter);
	GdkDisplay * display = GDK_DISPLAY(pygobject_get(pyobj_display));
	Display * xdisplay = GDK_DISPLAY_XDISPLAY (display);

	XDestroyWindow(xdisplay, win_inter);

	Py_RETURN_NONE;
}
static PyObject * py_dock_tray(PyObject * self, PyObject * args) {
	PyObject * py_gdk_window;
	PyObject * pyobj_display; // the GdkWindow
	PyObject * py_win_id;
	if (!PyArg_ParseTuple(args, "OOO", &py_gdk_window, &pyobj_display, &py_win_id))
		return NULL;

	if(!pygobject_check(py_gdk_window, pygobject_lookup_class(GDK_TYPE_WINDOW))) {
		return NULL;
	}

if(!pygobject_check(pyobj_display, pygobject_lookup_class(GDK_TYPE_DISPLAY))) {
		PyErr_SetString(PyExc_TypeError, "parameter must be A GdkDisplay");
		return NULL;
	}

	if(!PyLong_Check(py_win_id)) {
		return NULL;
	}
	long win_id = PyLong_AsLong(py_win_id);
	GdkWindow * window = GDK_WINDOW(pygobject_get(py_gdk_window));
	GdkDisplay * display = GDK_DISPLAY(pygobject_get(pyobj_display));
	GdkWindow * win_sub = gdk_x11_window_foreign_new_for_display(display, win_id);
	long systray_win_id = gdk_x11_window_get_xid(window);

printf("%d\n",__LINE__);
	    

	GdkScreen * screen = gdk_display_get_default_screen (display);
	Screen * xscreen = GDK_SCREEN_XSCREEN (screen);
	Display * xdisplay = GDK_DISPLAY_XDISPLAY (display);
printf("%d\n",__LINE__);


	XWindowAttributes panel_attributes;
	XGetWindowAttributes(xdisplay, systray_win_id, &panel_attributes);

	XWindowAttributes window_attributes;
	XGetWindowAttributes(xdisplay, win_id, &window_attributes);

	    
printf("DOCKING TRAY - %d %d depth:%d panel:%d\n",__LINE__, win_id, window_attributes.depth, panel_attributes.depth);
Window win_inter;

Colormap colormap = XCreateColormap(
            xdisplay,
            win_id,
            panel_attributes.visual, AllocNone );
	printf("%d\n",__LINE__);

{
XSetWindowAttributes wc;
    wc.override_redirect = True;
    wc.background_pixel = BlackPixelOfScreen(xscreen);
    wc.border_pixel = BlackPixelOfScreen(xscreen);
    wc.event_mask = ExposureMask;
	wc.colormap = colormap;

    win_inter = XCreateWindow(xdisplay, RootWindowOfScreen (xscreen), 0, 0, 1, 1, 0,
            panel_attributes.depth, InputOutput, panel_attributes.visual,
CWOverrideRedirect|CWBackPixel|CWBorderPixel|CWEventMask|CWColormap, &wc);
printf("%d\n",__LINE__);
	/*** GRRRR ***/
    Atom WM_TRANSIENT_FOR = XInternAtom(xdisplay, "WM_TRANSIENT_FOR", False);
    Atom WINDOW = XInternAtom(xdisplay, "WINDOW", False);
printf("%d\n",__LINE__);
    XChangeProperty(xdisplay, win_inter, WM_TRANSIENT_FOR, WINDOW, 32,
PropModeReplace, &systray_win_id, 1);
}

{
 int red_prec, green_prec, blue_prec, depth;
  GdkVisual *visual;
  visual = gdk_x11_screen_lookup_visual (screen,
                                         window_attributes.visual->visualid);

  gdk_visual_get_red_pixel_details (visual, NULL, NULL, &red_prec);
  gdk_visual_get_green_pixel_details (visual, NULL, NULL, &green_prec);
  gdk_visual_get_blue_pixel_details (visual, NULL, NULL, &blue_prec);
  depth = gdk_visual_get_depth (visual);

  int visual_has_alpha = red_prec + blue_prec + green_prec < depth;

	printf("visual_has_alpha: %d\n",visual_has_alpha);
}


	printf("OO: %d %d\n",win_id, systray_win_id);
	XUnmapWindow(GDK_DISPLAY_XDISPLAY (display),  win_id);
	printf("%d\n",__LINE__);
	    
	printf("%d\n",__LINE__);
	XSelectInput(GDK_DISPLAY_XDISPLAY (display), win_id, StructureNotifyMask | PropertyChangeMask| EnterWindowMask | FocusChangeMask);
	printf("%d\n",__LINE__);
	XReparentWindow(GDK_DISPLAY_XDISPLAY (display), win_id, win_inter, 0, 0);
	printf("%d\n",__LINE__);	    
	gdk_window_add_filter(win_sub, &call_python_filter, (gpointer)panel);

	XMapWindow(xdisplay,  win_id);
	XMapWindow(xdisplay, win_inter); 

///g_object_unref(win_sub) ???
	printf("%d\n",__LINE__);
	xembed_send(display, win_id, XEMBED_EMBEDDED_NOTIFY, 0, systray_win_id, 1);
	printf("%d\n",__LINE__);
	xembed_send(display, win_id, XEMBED_WINDOW_ACTIVATE, 0, 0, 0);
	printf("%d\n",__LINE__);
	xembed_send(display, win_id, XEMBED_FOCUS_IN, XEMBED_FOCUS_CURRENT, 0, 0);
	printf("%d\n",__LINE__);


//gdk_window_set_background_pattern (window, NULL);

	//Py_RETURN_NONE;
	return Py_BuildValue("l", win_inter);
}

#define TPL_FUNCTION(name) {#name, py_##name, METH_VARARGS, "Not documented"}
#define TPL_FUNCTION_DOC(name, doc) {#name, py_##name, METH_VARARGS, doc}

static PyMethodDef methods[] =
{
	TPL_FUNCTION_DOC(print, "simple print for testing"),
	TPL_FUNCTION_DOC(print_type, "Print the type of a GObject"),
	TPL_FUNCTION_DOC(set_strut, "set _NET_WM_STRUT"),
	TPL_FUNCTION_DOC(dock_tray, "dock tray"),
	TPL_FUNCTION_DOC(undock_tray, "undock tray"),
	TPL_FUNCTION_DOC(move_tray, "move tray "),
	TPL_FUNCTION_DOC(set_system_tray_filter, "implement X11 event filtering"),
	TPL_FUNCTION_DOC(set_system_tray_orientation, "set _NET_SYSTEM_TRAY_ORIENTATION"),
	TPL_FUNCTION_DOC(set_system_tray_visual, "set _NET_SYSTEM_TRAY_VISUAL"),
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
	return m;

}

