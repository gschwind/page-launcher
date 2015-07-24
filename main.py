#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
from math import *

from io import StringIO

from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GdkX11
from gi.repository import ClutterGdk
from gi.repository import Clutter
from gi.repository import GtkClutter
from gi.repository import Pango

import Xlib
from Xlib import display as D

from xdg.IconTheme import getIconPath
from xdg.DesktopEntry import DesktopEntry

import dbus
import dbus.service
from dbus.mainloop.glib import DBusGMainLoop

import shlex
import subprocess

font = "Sans 20"
color_apps = Clutter.Color.new(255,255,255,255)

font_entry = "Sans Bold 30"
color_entry = Clutter.Color.new(255,255,255,255) # red,green,blue,alpha


class apps_entry:
 def __init__(self, de):
  size = 128.0
  self.name = de.getName().lower()
  self.generic_name = de.getGenericName().lower()
  self.comment = de.getComment().lower()
  self.exe = re.sub(u"%\w*", u"", de.getExec())
  self.icon = self._find_icon(de.getIcon())
  self.icon.set_size(size,size)
  self.text = Clutter.Text.new_full(font, de.getName(), color_apps)
  self.text.set_width(size)
  self.text.set_ellipsize(Pango.EllipsizeMode.END)
  self.text.set_line_alignment(Pango.Alignment.CENTER)
  self.rect = Clutter.Rectangle.new()
  self.rect.set_size(128.0*1.2,128.0*1.2*1.5)
  self.rect.set_color(Clutter.Color.new(128,128,128,128))
  self.rect.set_reactive(True)
  self.rect.set_opacity(0)
  self.rect.connect("enter-event", self.enter_handler, self)
  self.rect.connect("leave-event", self.leave_handler, self)
  self.rect.connect("button-press-event", apps_entry.button_press_handler, self)
  self.fade_in_transition = Clutter.PropertyTransition.new("opacity")
  self.fade_in_transition.set_duration(100)
  self.fade_in_transition.set_to(255)
  self.fade_in_transition.connect("completed", self.fade_in_completed, self)
  self.fade_out_transition = Clutter.PropertyTransition.new("opacity")
  self.fade_out_transition.set_duration(2000)
  self.fade_out_transition.set_to(0)
  self.fade_out_transition.connect("completed", self.fade_out_completed, self)
  stage.add_child(self.rect)
  stage.add_child(self.icon)
  stage.add_child(self.text)
  self.hide()
  pass

 def enter_handler(self, widget, event, self1):
  if self.rect.get_transition("fade_out"):
   self.rect.remove_transition("fade_out")
  if not self.rect.get_transition("fade_in"):
   self.rect.add_transition("fade_in", self.fade_in_transition)
  return True
 def leave_handler(self, widget, event, self1):
  if self.rect.get_transition("fade_in"):
   self.rect.remove_transition("fade_in")
  if not self.rect.get_transition("fade_out"):
   self.rect.add_transition("fade_out", self.fade_out_transition)
  return True

 def _find_icon(self, icon_name, size = 128):
  icon_theme = Gtk.IconTheme.get_default()
  gtk_icon_info = icon_theme.lookup_icon(icon_name, 128, 0)
  if gtk_icon_info:
   icon_path = gtk_icon_info.get_filename()
  else:
   icon_path = getIconPath(icon_name, size)

  if (icon_path and re.match("^.*\.svg$", icon_path)) or not icon_path:
   gtk_icon_info = icon_theme.lookup_icon("exec", 128, 0)
   if gtk_icon_info:
    icon_path = gtk_icon_info.get_filename()
   else:
    icon_path = getIconPath(icon_name, size)
  return Clutter.Texture.new_from_file(icon_path)

 def show(self):
  self.rect.show()
  self.icon.show()
  self.text.show()
  pass
 def hide(self):
  self.rect.hide()
  self.icon.hide()
  self.text.hide()
  pass

 def set_position(self, x, y):
  self.rect.set_position(x-128.0*0.10,y-128.0*0.10)
  self.icon.set_position(x,y)
  self.text.set_position(x,y+128.0*1.10)
  pass

 def fade_in_completed(self, transition, self1):
  if self.rect.get_transition("fade_in"):
   self.rect.remove_transition("fade_in")
  self.rect.set_opacity(255)
  pass
 def fade_out_completed(self, transition, self1):
  if self.rect.get_transition("fade_out"):
   self.rect.remove_transition("fade_out")
  self.rect.set_opacity(0)
  pass

 def button_press_handler(widget, event, self):
  if event.button == Clutter.BUTTON_PRIMARY:
   self.call()
   stage.hide()
  pass

 def call(self):
  subprocess.Popen(shlex.split(self.exe))
  pass
 pass

class apps_handler:
 def _get_all_desktop_files(self, path):
  ret = list()
  for (rdir, dirs, files) in os.walk(path):
   for f in files:
    if self._rdsk.match(f):
     ret.append(os.path.join(rdir,f))
  return ret

 def __init__(self):
  self._apps = list()
  self._rdsk = re.compile(u".*\.desktop$")
  self._icon_actor_cache = dict()
  HOME = os.getenv("HOME")
  l = list()
  l += self._get_all_desktop_files(u'/usr/share/applications')
  if HOME:
   l += self._get_all_desktop_files(os.path.join(HOME, u'.local/share/applications'))
  for f in l:
   de = DesktopEntry(f)
   if de.getType() == u"Application" and not de.getHidden():
    self._apps.append(apps_entry(de))
  self._apps.sort(key=lambda x: x.name)
  pass

 def filter_apps(self, patern):
  p = re.compile(patern.lower())
  ret = list()
  for de in self._apps:
   if p.search(de.name):
    ret.append(de)
   elif p.search(de.generic_name):
    ret.append(de)
   elif p.search(de.comment):
    ret.append(de)
  return ret

 def hide_all(self):
  for x in self._apps:
   x.hide()

 pass


def key_press_handler(widget, event, data):
 if event.keyval == Clutter.KEY_Escape:
  stage.hide()
 elif event.keyval == Clutter.KEY_Return:
  if len(apps_list) == 1:
   apps_list[0].call()
   stage.hide()
 return False
 pass

def button_press_handler(widget, event, data):
 global apps_list
 widget = stage.get_actor_at_pos(Clutter.PickMode.ALL, event.x, event.y)
 if widget == intext:
  stage.set_key_focus(intext)
 else:
  stage.set_key_focus(intext)  


 pass

#def motion_handler(widget, event, data):
# global apps_list
# widget = stage.get_actor_at_pos(Clutter.PickMode.ALL, event.x, event.y)
# if type(widget) == type(intext):
#  widget.set_color(Clutter.Color.new(255,255,0,255))
# 
# layout = launcher_layout(len(apps_list))
# c = floor((event.x-layout.left_margin)/(layout.size*1.3))
# r = floor((event.y-layout.y_offset-layout.top_margin)/(layout.size*1.5*1.3))
# if c >= 0 and c < layout.columns and r >= 0 and r < layout.rows:
#  selected_rect.set_position(c*layout.size*1.3+layout.left_margin-layout.size*0.15,r*1.5*1.3*layout.size+layout.y_offset+layout.top_margin-layout.size*0.15)
# return False
# pass

current_actor = list()
apps_list = list()

class launcher_layout:
 def __init__(self, napps):
   self.size = 128.0
   self.y_offset = intext.get_height()
   self.width = stage.get_width()
   self.height = stage.get_height() - self.y_offset
   self.columns = int(floor(self.width/(self.size*1.3)))
   self.rows = int(floor(self.height/(self.size*1.5*1.3)))
   self.left_margin = (self.width-(self.columns*self.size*1.3))/2.0
   self.top_margin = (self.height-(self.rows*self.size*1.5*1.3))/2.0
   npages = int(floor(napps/self.columns*self.rows)+1.0)
 pass

def handle_text_changed(widget, data):
 global apps
 global apps_list
 global current_actor

 apps.hide_all()

 text = widget.get_text()
 if not text:
  text = u""
 apps_list = apps.filter_apps(text)
 layout = launcher_layout(len(apps_list))

 if text == u"":
  notext.show()
 else:
  notext.hide()

 current_actor = list()
 
 for i in range(0, layout.columns*layout.rows):
  if i >= len(apps_list):
   break
  c = i - floor(i / layout.columns)*layout.columns
  l = floor(i / layout.columns)
  a = apps_list[i]
  a.set_position(c*layout.size*1.3+layout.left_margin,l*1.5*1.3*layout.size+layout.y_offset+layout.top_margin)
  a.show()
  current_actor.append(a)
 pass

def activate_handler(widget, data):
 stage.set_key_focus(intext)
 pass

def desactivate_handler(widget, data):
 stage.hide()
 pass

def allocation_changed_handler(widget, box, flags, data):
 # indirectly update the layout:
 handle_text_changed(intext, data)
 pass

# dbus model : dbus have object and interface.
# object name has form : /x/y/object_name
# interface name has form: x.y.interface_name
# an object can implement severals interfases
# interfaces can be used by severrals object.
class DBusWidget(dbus.service.Object):
 interface_name = "org.page.launcher"
 object_name = "/org/page/launcher"
 def __init__(self):
  session_bus = dbus.SessionBus()
  dbus_name = dbus.service.BusName("org.page.launcher", session_bus)
  dbus.service.Object.__init__(self, dbus_name, "/org/page/launcher")

 @dbus.service.method("org.page.launcher", in_signature='', out_signature='')
 def map(self):
  intext.set_text(u"")
  stage.show()
  stage.set_key_focus(intext)
  pass
 
 @dbus.service.method("org.page.launcher", in_signature='', out_signature='')
 def quit(self):
  Clutter.main_quit()
  pass
 pass

if __name__ == '__main__':
 Clutter.set_windowing_backend(Clutter.WINDOWING_GDK)
 Clutter.init(sys.argv)
 
 # check if page-launcher is already running
 loop = DBusGMainLoop(set_as_default=True)
 bus = dbus.SessionBus()
 try:
  remote_object = bus.get_object(DBusWidget.interface_name, DBusWidget.object_name)
 except dbus.DBusException:
  remote_object = None

 # If the object exist just active the running launcher
 if remote_object:
  iface = dbus.Interface(remote_object, DBusWidget.interface_name)
  iface.map()
  sys.exit(0)

 # create the object dbus listenner
 dbus_launcher = DBusWidget()

 stage = Clutter.Stage()
 intext = Clutter.Text.new_full(font_entry, u"", color_entry)
 apps = apps_handler()
 notext = Clutter.Text.new_full(font_entry, u"Enter Text Here", Clutter.Color.new(128,128,128,255))
 
 #windows.set_type_hint(Gdk.WindowTypeHint.DOCK)
 #windows.stick()
 #geo = Gdk.Geometry()
 #geo.min_width=300
 #geo.min_height=800
 #geo.max_width=300
 #geo.max_height=800
 #geo.base_width=300
 #geo.base_height=800
 #windows.set_geometry_hints(geo, Gdk.WindowHints.MIN_SIZE | Gdk.WindowHints.MAX_SIZE | Gdk.WindowHints.BASE_SIZE)

 #print windows.get_xid()
 #left=0
 #right=300
 #top = 0 
 #bottom=800
 #_display = D.Display()
 #_win = _display.create_resource_object('window', windows.get_xid())
 #_win.change_property(_display.intern_atom('_NET_WM_STRUT'), _display.intern_atom('CARDINAL'),32, [left,right,top,bottom])
 #_win.change_property(_display.intern_atom('_NET_WM_DESKTOP'), _display.intern_atom('CARDINAL'),32, [0xffffffff])
 #_win.change_property(_display.intern_atom('_NET_WM_WINDOW_TYPE'), _display.intern_atom('CARDINAL'),32, [_display.intern_atom("_NET_WM_WINDOW_TYPE_DOCK")])
 #print(windows)
 #sys.exit(-1)
 #Clutter.main()

class BaseView:
#----------------------------------------------------------------------------
    #------------------------------------------------------
 def __init__(self, display, x, y, width, height):
        resolution = display.screen().root.get_geometry()
        self.screen_width_px = resolution.width
        self.screen_height_px = resolution.height        
        self.display = display
        self.x       = x
        self.y       = y
        self.width   = width
        self.height  = self.screen_height_px
        self._init_x_atoms()
        self._init_clutter()
        self.screen  = display.screen()
        self._set_props()
        self.set_struts(500,500,500,0,0,0,0,0,0,0,0,0)
        self.window.map()
        self.display.flush()

 def _init_clutter(self):
        stage.set_user_resizable(True)
        stage.set_title("page-launcher")
        stage.set_use_alpha(True)
        stage.set_opacity(128)
        stage.set_color(Clutter.Color.new(32,32,32,128))


        stage.add_child(notext)
        notext.show()
        intext.set_editable(True)
        intext.set_selectable(True)
        intext.set_activatable(True)
        intext.connect("key-press-event", key_press_handler, None)
        stage.add_child(intext)
        intext.show()

        selected_rect = Clutter.Rectangle.new()
        selected_rect.set_size(128.0*1.3,128.0*1.3*1.5)
        selected_rect.set_color(Clutter.Color.new(128,128,128,128))
        selected_rect.hide()
        stage.add_child(selected_rect)


        apps.hide_all()
        stage.set_key_focus(intext)

        #stage.connect('button-press-event', lambda x, y: print("pressed"))
        #stage.connect('button-release-event', button_press_handler, None)
        stage.connect('key-press-event', key_press_handler, None)
        #stage.connect('motion-event', motion_handler, None)
        stage.connect('destroy', lambda x: Clutter.main_quit())
        #stage.connect('deactivate', desactivate_handler, None)
        stage.connect('allocation-changed', allocation_changed_handler, None)
        #stage.connect('activate', activate_handler, None)
        intext.connect('text-changed', handle_text_changed, None)

        #toto= Clutter.get_default_backend()
        #print(toto)
        stage.show()
        #.create_resource_object('window', winId)

        self.windowsClutter = ClutterGdk.get_stage_window(stage)
        self.windowsClutter.move_resize(0,0,100,800)
        #print(windows.get_xid())
        self.window = self.display.create_resource_object('window', self.windowsClutter.get_xid())
    #-------------------------------------
 def _init_x_atoms(self):
    #-------------------------------------
        self._ABOVE                   = self.display.intern_atom("_NET_WM_STATE_ABOVE")
        self._BELOW                   = self.display.intern_atom("_NET_WM_STATE_BELOW")
        self._BLACKBOX                = self.display.intern_atom("_BLACKBOX_ATTRIBUTES")
        self._CHANGE_STATE            = self.display.intern_atom("WM_CHANGE_STATE")
        self._CLIENT_LIST             = self.display.intern_atom("_NET_CLIENT_LIST")
        self._CURRENT_DESKTOP         = self.display.intern_atom("_NET_CURRENT_DESKTOP")
        self._DESKTOP                 = self.display.intern_atom("_NET_WM_DESKTOP")
        self._DESKTOP_COUNT           = self.display.intern_atom("_NET_NUMBER_OF_DESKTOPS")
        self._DESKTOP_NAMES           = self.display.intern_atom("_NET_DESKTOP_NAMES")
        self._HIDDEN                  = self.display.intern_atom("_NET_WM_STATE_HIDDEN")
        self._ICON                    = self.display.intern_atom("_NET_WM_ICON")
        self._NAME                    = self.display.intern_atom("_NET_WM_NAME")
        self._RPM                     = self.display.intern_atom("_XROOTPMAP_ID")
        self._SHADED                  = self.display.intern_atom("_NET_WM_STATE_SHADED")
        self._SHOWING_DESKTOP         = self.display.intern_atom("_NET_SHOWING_DESKTOP")
        self._SKIP_PAGER              = self.display.intern_atom("_NET_WM_STATE_SKIP_PAGER")
        self._SKIP_TASKBAR            = self.display.intern_atom("_NET_WM_STATE_SKIP_TASKBAR")
        self._STATE                   = self.display.intern_atom("_NET_WM_STATE")
        self._STICKY                  = self.display.intern_atom("_NET_WM_STATE_STICKY")
        self._STRUT                   = self.display.intern_atom("_NET_WM_STRUT")
        self._STRUTP                  = self.display.intern_atom("_NET_WM_STRUT_PARTIAL")
        self._WMSTATE                 = self.display.intern_atom("WM_STATE")
        self._WIN_STATE               = self.display.intern_atom("_WIN_STATE")
        self._MOTIF_WM_HINTS          = self.display.intern_atom("_MOTIF_WM_HINTS")
        self._NET_WM_WINDOW_TYPE      = self.display.intern_atom("_NET_WM_WINDOW_TYPE")
        self._NET_WM_WINDOW_TYPE_DOCK = self.display.intern_atom("_NET_WM_WINDOW_TYPE_DOCK")

 def _set_props(self):
        print("OUOU")
        self.window.set_wm_hints(flags=(Xlib.Xutil.InputHint | Xlib.Xutil.StateHint), input=0, initial_state=1)
        self.window.set_wm_normal_hints(flags=(
                Xlib.Xutil.PPosition |
                Xlib.Xutil.PMaxSize |
                Xlib.Xutil.PMinSize),
            min_width=self.width, min_height=self.height,
            max_width=self.width, max_height=self.height)

        self.window.change_property(self._WIN_STATE, Xlib.Xatom.CARDINAL,32,[1])
        self.window.change_property(self._MOTIF_WM_HINTS, self._MOTIF_WM_HINTS, 32, [0x2, 0x0, 0x0, 0x0, 0x0])
        self.window.change_property(self._DESKTOP, Xlib.Xatom.CARDINAL, 32, [0xffffffff])
        self.window.change_property(self._NET_WM_WINDOW_TYPE, Xlib.Xatom.ATOM, 32, [self._NET_WM_WINDOW_TYPE_DOCK])

 def set_struts(self, left_start, left, left_end, right_start, right, right_end, top_start, top, top_end, bottom_start, bottom, bottom_end):
        self.window.change_property(self._STRUT, Xlib.Xatom.CARDINAL, 32, [left, right, top, bottom])
        self.window.change_property(self._STRUTP, Xlib.Xatom.CARDINAL, 32, [left, right, top, bottom, left_start, left_end, right_start, right_end, top_start, top_end, bottom_start, bottom_end])


 def run(self):
        Clutter.main()

launcher = BaseView(D.Display(), 0, 0 ,500,600)
launcher.run()

