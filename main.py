#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
from math import *

from io import StringIO

from gi.repository import Gtk
from gi.repository import Clutter
from gi.repository import GtkClutter
from gi.repository import Pango

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
  self.rect.connect("enter-event", apps_entry.enter_handler, self)
  self.rect.connect("leave-event", apps_entry.leave_handler, self)
  self.rect.connect("button-press-event", apps_entry.button_press_handler, self)
  self.fade_in_transition = Clutter.PropertyTransition.new("opacity")
  self.fade_in_transition.set_duration(100)
  self.fade_in_transition.set_to(255)
  self.fade_in_transition.connect("completed", apps_entry.fade_in_completed, self)
  self.fade_out_transition = Clutter.PropertyTransition.new("opacity")
  self.fade_out_transition.set_duration(2000)
  self.fade_out_transition.set_to(0)
  self.fade_out_transition.connect("completed", apps_entry.fade_out_completed, self)
  stage.add_child(self.rect)
  stage.add_child(self.icon)
  stage.add_child(self.text)
  self.hide()
  pass

 def enter_handler(widget, event, self):
  if self.rect.get_transition("fade_out"):
   self.rect.remove_transition("fade_out")
  if not self.rect.get_transition("fade_in"):
   self.rect.add_transition("fade_in", self.fade_in_transition)
  return True
 def leave_handler(widget, event, self):
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
  if not icon_path:
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

 def fade_in_completed(transition, self):
  if self.rect.get_transition("fade_in"):
   self.rect.remove_transition("fade_in")
  self.rect.set_opacity(255)
  pass
 def fade_out_completed(transition, self):
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
 stage.set_user_resizable(True)
 stage.set_title("page-launcher")
 stage.set_use_alpha(True)
 stage.set_opacity(128)
 stage.set_color(Clutter.Color.new(32,32,32,128))

 notext = Clutter.Text.new_full(font_entry, u"Enter Text Here", Clutter.Color.new(128,128,128,255))
 stage.add_child(notext)
 notext.show()
 intext = Clutter.Text.new_full(font_entry, u"", color_entry)
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

 apps = apps_handler()

 apps.hide_all()
 stage.set_key_focus(intext)

 #stage.connect('button-press-event', lambda x, y: print("pressed"))
 #stage.connect('button-release-event', button_press_handler, None)
 stage.connect('key-press-event', key_press_handler, None)
 #stage.connect('motion-event', motion_handler, None)
 stage.connect('destroy', lambda x: Clutter.main_quit())
 stage.connect('deactivate', desactivate_handler, None)
 stage.connect('allocation-changed', allocation_changed_handler, None)
 stage.connect('activate', activate_handler, None)
 intext.connect('text-changed', handle_text_changed, None)

 stage.show()
 Clutter.main()

