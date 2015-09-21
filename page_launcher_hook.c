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

#if 0
static GdkFilterReturn call_python_filter (GdkXEvent *gdkxevent, GdkEvent *event, gpointer data) {
	int retval;
  	PyGILState_STATE gstate;
	gstate = PyGILState_Ensure();
	 XEvent        *xevent = (XEvent *)gdkxevent;

	PyObject * panel = (PyObject*)data;
	PyObject * func = PyObject_GetAttrString(panel, "tray_filter"); 
	//PyObject * func_opcode = PyObject_GetAttrString(myobject, "tray_filter_opcode"); 
	//PyObject * func_message_data = PyObject_GetAttrString(myobject, "tray_filter_message_data"); 

	if (!PyCallable_Check(func)) {
		PyErr_SetString(PyExc_TypeError, "parameter tray_filter be callable");
		return GDK_FILTER_CONTINUE;
	}

	PyObject *py_event = pyg_boxed_new(GDK_TYPE_EVENT, event, FALSE, FALSE);
	PyObject* args = Py_BuildValue("(O)",py_event);	
 	PyObject* result = PyObject_CallObject(func, args);
	Py_DECREF(args);
	Py_DECREF(py_event);


   	 PyGILState_Release(gstate);
#endif



static void
tray_manager_handle_dock_request (PyObject       *manager,
				      XClientMessageEvent  *xevent)
{
printf("%s\n",__FUNCTION__);

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
#if  0
  GtkWidget *socket;
  Window *window;
  
  socket = gtk_socket_new ();
  
  /* We need to set the child window here
   * so that the client can call _get functions
   * in the signal handler
   */
  window = g_new (Window, 1);
  *window = xevent->data.l[2];
      
  g_object_set_data_full (G_OBJECT (socket),
			  "egg-tray-child-window",
			  window, g_free);
  g_signal_emit (manager, manager_signals[TRAY_ICON_ADDED], 0,
		 socket);

  /* Add the socket only if it's been attached */
  if (GTK_IS_WINDOW (gtk_widget_get_toplevel (GTK_WIDGET (socket))))
    {
      g_signal_connect (socket, "plug_removed",
			G_CALLBACK (egg_tray_manager_plug_removed), manager);
      
      gtk_socket_add_id (GTK_SOCKET (socket), xevent->data.l[2]);

      g_hash_table_insert (manager->socket_table, GINT_TO_POINTER (xevent->data.l[2]), socket);
    }
  else
    gtk_widget_destroy (socket);
#endif
}

static void
tray_manager_handle_message_data (PyObject       *manager,
				       XClientMessageEvent  *xevent)
{
printf("%s\n",__FUNCTION__);
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
printf("%s\n",__FUNCTION__);
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
printf("%s\n",__FUNCTION__);
}
static GdkFilterReturn call_python_filter (GdkXEvent *gdkxevent, GdkEvent *event, gpointer data) {
  XEvent *xevent = (GdkXEvent *)gdkxevent;
  PyObject *manager = data;
  GdkFilterReturn retval = GDK_FILTER_CONTINUE;

  	PyGILState_STATE gstate;
	gstate = PyGILState_Ensure();

printf("%s\n",__FUNCTION__);

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
  
 

   	 PyGILState_Release(gstate);
  return retval;
}

static PyObject * py_set_system_tray_filter(PyObject * self, PyObject * args) {
	PyObject * pyobj_win; // the GdkWindow
	PyObject * pyobj_display; // the GdkWindow
	PyObject * panel; // the function to call

	
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
	gdk_window_add_filter(window, &call_python_filter, (gpointer)panel);

	Py_RETURN_NONE;
}

#define XEMBED_EMBEDDED_NOTIFY 0

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
	long systray_win_id = gdk_x11_window_get_xid(window);

	printf("OO: %d %d\n",win_id, systray_win_id);
     //ewmh_set_wm_state(s->win, NormalState);

#if 0
	GdkAtom wm_state = gdk_atom_intern("WM_STATE", FALSE);
	unsigned char d[] = { NormalState, None };
    XChangeProperty(GDK_DISPLAY_XDISPLAY (display), win_id, gdk_x11_atom_to_xatom_for_display (display,wm_state),
                     gdk_x11_atom_to_xatom_for_display (display,wm_state), 32, PropModeReplace, d, 2);
#endif
	//GdkAtom cardinal = gdk_atom_intern("CARDINAL", FALSE);
	//GdkAtom _net_system_tray_orientation = gdk_atom_intern("_NET_SYSTEM_TRAY_ORIENTATION", FALSE);

	//gdk_property_change(GDK_WINDOW(pygobject_get(py_gdk_window)),
	//			wm_state, wm_state, 32, GDK_PROP_MODE_REPLACE,
	//			(guchar*) d, 2);

     XSelectInput(GDK_DISPLAY_XDISPLAY (display), win_id, StructureNotifyMask | PropertyChangeMask| EnterWindowMask | FocusChangeMask);
     XReparentWindow(GDK_DISPLAY_XDISPLAY (display), win_id, systray_win_id, 0, 0);

#if 0
	{

	  XEvent ev;
		ev.xclient.type = ClientMessage;
		ev.xclient.window = win_id;
		ev.xclient.message_type = gdk_x11_get_xatom_by_name_for_display (display,
			                                                "_XEMBED");
		ev.xclient.format = 32;
		ev.xclient.data.l[0] = gdk_x11_get_server_time (window);
		ev.xclient.data.l[1] = XEMBED_EMBEDDED_NOTIFY;
		ev.xclient.data.l[2] = 0;
		ev.xclient.data.l[3] = systray_win_id;
		ev.xclient.data.l[4] = 0; // version

			XSendEvent (GDK_DISPLAY_XDISPLAY (display),
				  win_id,
				  False, NoEventMask, (XEvent *)&ev);
	}
#endif

	Py_RETURN_NONE;
}

#define TPL_FUNCTION(name) {#name, py_##name, METH_VARARGS, "Not documented"}
#define TPL_FUNCTION_DOC(name, doc) {#name, py_##name, METH_VARARGS, doc}

static PyMethodDef methods[] =
{
	TPL_FUNCTION_DOC(print, "simple print for testing"),
	TPL_FUNCTION_DOC(print_type, "Print the type of a GObject"),
	TPL_FUNCTION_DOC(set_strut, "set _NET_WM_STRUT"),
	TPL_FUNCTION_DOC(dock_tray, "dock tray"),
	TPL_FUNCTION_DOC(set_system_tray_filter, "implement X11 event filtering"),
	TPL_FUNCTION_DOC(set_system_tray_orientation, "set _NET_SYSTEM_TRAY_ORIENTATION"),
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

