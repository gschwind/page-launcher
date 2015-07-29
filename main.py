#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys

import signal
import time

import gc
import pprint

from math import *

from io import StringIO


from gi.repository import GObject
from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GdkX11
from gi.repository import GdkPixbuf
from gi.repository import ClutterGdk
from gi.repository import Clutter
from gi.repository import GtkClutter
from gi.repository import Pango
from gi.repository import Wnck

import PageLauncherHook

from xdg.IconTheme import getIconPath
from xdg.DesktopEntry import DesktopEntry

import dbus
import dbus.service
from dbus.mainloop.glib import DBusGMainLoop

import shlex
import subprocess

font_dash = "Sans 12"
color_apps = Clutter.Color.new(255,255,255,255)

font_entry = "Sans Bold 30"
color_entry = Clutter.Color.new(255,255,255,255) # red,green,blue,alpha


font_menu_entry = "Sans Bold 10"
color_menu_entry = Clutter.Color.new(255,255,255,255) # red,green,blue,alpha

font_icon_entry = "Sans Bold 8"
color_icon_entry = Clutter.Color.new(128,128,128,255) # red,green,blue,alpha


def sig_int_handler(n):
 Clutter.main_quit()


class apps_entry:
 def __init__(self, parent, de):
  self.parent = parent
  self.ico_size = parent.ico_size
  self.name = de.getName().lower()
  self.generic_name = de.getGenericName().lower()
  self.comment = de.getComment().lower()
  self.exe = re.sub(u"%\w*", u"", de.getExec())
  self.icon_name = de.getIcon()
  self.icon = self._find_icon(de.getIcon())
  self.icon.set_size(self.ico_size,self.ico_size)
  self.text = Clutter.Text.new_full(font_dash, de.getName(), color_apps)
  self.text.set_width(self.ico_size*1.5)
  self.text.set_ellipsize(Pango.EllipsizeMode.END)
  self.text.set_line_alignment(Pango.Alignment.CENTER)
  self.rect = Clutter.Rectangle.new()
  self.rect.set_size(self.ico_size*1.2*1.5,self.ico_size*1.2*1.5)
  self.rect.set_color(Clutter.Color.new(255,255,255,128))
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
  self.parent.add_child(self.rect)
  self.parent.add_child(self.icon)
  self.parent.add_child(self.text)
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

 def get_icon(self, size = 128):
  return self._find_icon(self.icon_name, size)

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
  self.rect.set_position(x-self.ico_size*0.10,y-self.ico_size*0.10)
  self.icon.set_position(x,y)
  self.text.set_position(x,y+self.ico_size*1.10)
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
   self.parent.hide()
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

 def __init__(self, stage):
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
    self._apps.append(apps_entry(stage, de))
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

class launcher_layout:
 def __init__(self, stage, napps):
   self.size = stage.ico_size
   self.y_offset = stage.intext.get_height()
   self.width = stage.get_width()
   self.height = stage.get_height() - self.y_offset
   self.columns = int(floor(self.width/(self.size*1.5*1.3)))
   self.rows = int(floor(self.height/(self.size*1.5*1.3)))
   self.left_margin = (self.width-(self.columns*self.size*1.5*1.3))/2.0
   self.top_margin = (self.height-(self.rows*self.size*1.5*1.3))/2.0
   npages = int(floor(napps/self.columns*self.rows)+1.0)
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
  #intext.set_text(u"")
  #stage.show()
  #stage.set_key_focus(intext)
  pass
 
 @dbus.service.method("org.page.launcher", in_signature='', out_signature='')
 def quit(self):
  Clutter.main_quit()
  pass
 pass

class DashView(Clutter.Stage):
	def __init__(self, parent):
		super().__init__()
		self.is_grab = False
		self._create_dash_window()
		ClutterGdk.set_stage_foreign(self, self.window)
				
		self.ico_size = 104
		self.parent = parent
		self.set_user_resizable(False)
		self.set_title("page-dash")
		self.set_use_alpha(True)
		self.set_opacity(200)
		self.set_color(Clutter.Color.new(32,32,32,128))
		self.set_scale(1.0, 1.0)
		self.set_accept_focus(True)

		self.notext = Clutter.Text.new_full(font_entry, u"Enter Text Here",
		 Clutter.Color.new(255,255,255,128))
		self.add_child(self.notext)
		self.notext.show()
		
		
		self.intext = Clutter.Text.new_full(font_entry, u"", color_entry)
		self.apps = apps_handler(self)
		self.intext.set_editable(True)
		self.intext.set_selectable(True)
		self.intext.set_activatable(True)
		

		self.add_child(self.intext)
		self.intext.show()

		#self.selected_rect = Clutter.Rectangle.new()
		#self.selected_rect.set_size(self.ico_size*1.3,self.ico_size*1.3*1.5)
		#self.selected_rect.set_color(Clutter.Color.new(128,128,128,128))
		#self.selected_rect.hide()
		#self.add_child(self.selected_rect)

		self.apps.hide_all()
		self.set_key_focus(self.intext)

		self.connect('button-press-event', self.button_press_handler)
		
		# check for enter or Escape
		self.intext.connect("key-press-event", self.key_press_handler)
		self.connect('key-press-event', self.key_press_handler)
		self.connect('allocation-changed', self.allocation_changed_handler)
		self.intext.connect('text-changed', self.handle_text_changed)


	def show(self, time):
		print("Dash show")

		parent_window = ClutterGdk.get_stage_window(self.parent)
		self.window.set_transient_for(parent_window)
		root_height = self.window.get_screen().get_root_window().get_height()
		self.set_size(500, root_height)
		self.window.move(32, 0)
		
		if self.intext.get_text() == u"":
			self.notext.show()
		else:
			self.notext.hide()
		
		super().show()
		self.window.focus(time)
		self._start_grab(time)
		self.set_key_focus(self.intext)
		
	def _start_grab(self, time):
		if self.is_grab:
			return
		self.is_grab = True
		dpy = ClutterGdk.get_default_display()
		dm = dpy.get_device_manager()
		dev = dm.list_devices(Gdk.DeviceType.MASTER)
		for d in dev:
		 d.grab(self.window, Gdk.GrabOwnership.WINDOW, False,
		 Gdk.EventMask.BUTTON_PRESS_MASK
		 |Gdk.EventMask.BUTTON_RELEASE_MASK
		 |Gdk.EventMask.BUTTON_MOTION_MASK
		 |Gdk.EventMask.POINTER_MOTION_MASK
		 |Gdk.EventMask.KEY_PRESS_MASK
		 |Gdk.EventMask.KEY_RELEASE_MASK,
		 None,
		 time)
		 
	def _stop_grab(self):
		if not self.is_grab:
			return
		self.is_grab = False
		dpy = ClutterGdk.get_default_display()
		dm = dpy.get_device_manager()
		dev = dm.list_devices(Gdk.DeviceType.MASTER)
		for d in dev:
		 d.ungrab(Gdk.CURRENT_TIME)

	def key_press_handler(self, widget, event):
		if event.keyval == Clutter.KEY_Escape:
			self.hide()
			return True
		elif event.keyval == Clutter.KEY_Return:
			if len(self.apps_list) == 1:
				self.apps_list[0].call()
				self.hide()
			return True
		return False

	def button_press_handler(self, widget, event):
		widget = self.get_actor_at_pos(Clutter.PickMode.ALL, event.x, event.y)
		if widget == self.intext:
			self.set_key_focus(self.intext)
			return True
		elif not widget:
			self.hide()
			return True
		return False


	def handle_text_changed(self, data = None):
		self.apps.hide_all()
		text = self.intext.get_text()
		print("XXXX"+text)
		if not text:
			text = u""
		self.apps_list = self.apps.filter_apps(text)
		layout = launcher_layout(self, len(self.apps_list))
		if text == u"":
			self.notext.show()
		else:
			self.notext.hide()
		self.current_actor = list()
 
		for i in range(0, layout.columns*layout.rows):
			if i >= len(self.apps_list):
				break
			c = i - floor(i / layout.columns)*layout.columns
			l = floor(i / layout.columns)
			a = self.apps_list[i]
			a.set_position(c*layout.size*1.5*1.3+layout.left_margin,l*1.5*1.3*layout.size+layout.y_offset+layout.top_margin)
			a.show()
			self.current_actor.append(a)
		pass

	def activate_handler(self, data):
		self.set_key_focus(self.intext)
		pass

	def desactivate_handler(self, data):
		self.hide()
		pass

	def allocation_changed_handler(self, box, flags, data):
		# indirectly update the layout:
		self.handle_text_changed(data)
		pass
		
	def hide(self):
		print("Dash hide")
		self._stop_grab()
		super().hide()
		
		
	def _create_dash_window(self):
		display = ClutterGdk.get_default_display()
		root_height = display.get_default_screen().get_root_window().get_height()
	
		attr = Gdk.WindowAttr();
		attr.title = "page-dash"
		attr.width = 32
		attr.height = root_height
		attr.x = 32
		attr.y = 0
		attr.event_mask = 0
		attr.window_type = Gdk.WindowType.TOPLEVEL
		attr.visual = display.get_default_screen().get_rgba_visual()
		attr.override_redirect = True
		attr.type_hint = Gdk.WindowTypeHint.MENU
		self.window = Gdk.Window(None, attr,
		 Gdk.WindowAttributesType.TITLE
		|Gdk.WindowAttributesType.VISUAL
		|Gdk.WindowAttributesType.X
		|Gdk.WindowAttributesType.Y
		|Gdk.WindowAttributesType.NOREDIR
		|Gdk.WindowAttributesType.TYPE_HINT)



class SubWindow(Clutter.Stage):
	def __init__(self, parent, x, y, width, height):
		super().__init__()
		self.is_grab = False
		self._create_menu_window()
		ClutterGdk.set_stage_foreign(self, self.window)
		self.window.set_transient_for(parent.window)

		self.parent = parent
		self.set_user_resizable(False)
		self.set_title("sub-win")
		self.set_use_alpha(True)
		self.set_opacity(255)
		self.set_color(Clutter.Color.new(0,0,0,0))
		self.set_scale(1.0, 1.0)
		self.set_accept_focus(True)

	def _create_menu_window(self):
		display = ClutterGdk.get_default_display()
		self.root_height = display.get_default_screen().get_root_window().get_height()
	
		attr = Gdk.WindowAttr();
		attr.title = "sub-win"
		attr.width = 100
		attr.height = 100
		attr.x = 100
		attr.y = 100
		attr.event_mask = 0
		attr.window_type = Gdk.WindowType.TOPLEVEL
		attr.visual = display.get_default_screen().get_rgba_visual()
		attr.override_redirect = True
		attr.type_hint = Gdk.WindowTypeHint.MENU
		self.window = Gdk.Window(None, attr,
		 Gdk.WindowAttributesType.TITLE
		|Gdk.WindowAttributesType.VISUAL
		|Gdk.WindowAttributesType.X
		|Gdk.WindowAttributesType.Y
		|Gdk.WindowAttributesType.NOREDIR
		|Gdk.WindowAttributesType.TYPE_HINT)


class ItemMenu(Clutter.Rectangle):
	def __init__(self):
		super().__init__()
		self.set_color(		Clutter.Color.new(0,0,0,0))
		self.set_border_color(		Clutter.Color.new(100,100,100,255))
		self.set_opacity(100)
		self.set_border_width(2)
		self.set_reactive(True)
		self.connect("enter-event", ItemMenu.event_menu_enter_handler, self, self)
		self.connect("leave-event", ItemMenu.event_menu_leave_handler, self, self)
		
		self.fade_in_transition = Clutter.PropertyTransition.new("opacity")
		self.fade_in_transition.set_duration(100)
		self.fade_in_transition.set_to(255)
		self.fade_out_transition = Clutter.PropertyTransition.new("opacity")
		self.fade_out_transition.set_duration(500)
		self.fade_out_transition.set_to(100)
		self.fade_in_transition.connect("completed", ItemMenu.fade_in_completed, self, self)
		self.fade_out_transition.connect("completed", ItemMenu.fade_out_completed, self, self)

	def fade_in_completed(self, transition, item):
		if item.get_transition("fade_in"):
			item.remove_transition("fade_in")
		item.set_opacity(255)
		pass
	def fade_out_completed(self, transition, item):
		if item.get_transition("fade_out"):
			item.remove_transition("fade_out")
		item.set_opacity(100)
		pass
	def event_menu_enter_handler(widget, event, self,  item):
		print('ENTER')
		if item.get_transition("fade_out"):
			item.remove_transition("fade_out")
		if not item.get_transition("fade_in"):
			item.add_transition("fade_in", item.fade_in_transition)
		return True
	def event_menu_leave_handler(widget, event, self,  item):
		print('LEAVE')
		if item.get_transition("fade_in"):
			item.remove_transition("fade_in")
		if not item.get_transition("fade_out"):
			item.add_transition("fade_out", item.fade_out_transition)
		return True

class PanelMenu(SubWindow):
	def __init__(self, parent):
		super().__init__(parent, 0, 0, 100, 100)
		
		self.item_height = 32
		self.item_margin_y = 2
		self.item_margin_x = 20
		self.global_width = 0
		self.global_height = 0
		#self.connect("deactivate", PanelMenu.event_menu_focus_out, self)
	
		#elf.container = Clutter.Group()
			

	def event_menu_button_press(widget, event, self,  menu):
		print('PANEL KEYPRESS')
		menu['obj'].activate(event.time)
		self.hide_menu()

	def event_menu_focus_out(widget, event, self):
		print('FOCUS OUT')
		self.hide_menu()

	def show_menu(self, x, y, menulist, event_time):
		print('SHOW MENU')
		self.set_x(0)
		self.set_y(0)
		#self.rect.remove_all_children()
		self.destroy_all_children()
		#self.remove_all()

		self.rect = Clutter.Actor()
		self.rect.set_x(0)
		self.rect.set_y(0)
		self.rect.set_background_color(Clutter.Color.new(32,32,32,200))	

		self.add_child(self.rect)
		self.global_width = 0

		tmp_height = 0
		for menu in menulist:
			##print(menu['text'])
			text = Clutter.Text.new_full(font_menu_entry, menu['text'], Clutter.Color.new(255,255,255,255))
			#menu['txt'] = text
			#text.set_background_color(Clutter.Color.new(0,0,0,0))	

			text.set_x(self.item_margin_x)
			text.set_y(tmp_height+self.item_height/4)

			self.rect.add_child(text)
		
			self.global_width = max(self.global_width, text.get_width()+2*self.item_margin_x)
			tmp_height += self.item_height+self.item_margin_y
		self.global_height = tmp_height-+self.item_margin_y

		tmp_height = 0
		for menu in menulist:
			back = ItemMenu()
			back.connect("button-press-event", PanelMenu.event_menu_button_press, self, menu)

			back.set_x(0)
			back.set_y(tmp_height)
			back.set_size(self.global_width,self.item_height)

			tmp_height += self.item_height + self.item_margin_y
			self.rect.add_child(back)

		#self.window.set_size()
		#dir(self.window)
		print(self.global_width)
		print(self.global_height)

		if self.root_height < self.global_height+y:
			y=self.root_height-self.global_height
		
		self.window.resize(self.global_width, self.global_height)
		self.window.move(x,y)


		self.set_size(self.global_width, self.global_height)	
		self.rect.set_size(self.global_width, self.global_height)	
		
		self.window.focus(event_time)
		self.show_all()
		pass

	def hide_menu(self):
		self.hide()
		self.remove_all()

class PanelApp():
	def __init__(self, xid, name, group):
		self.xid = xid
		self.updated = 0
		self.name = name
		self.group_name = group.get_name()
		self.group = group
		# Add app to group
		self.group.add(self)

	def unregister(self):
		self.group = None

	def activate(self,event_time):
		print('ACTIVATE: '+str(self.xid))
		w=Wnck.Window.get(self.xid)
		if w!=None:
			w.activate(event_time)
		pass

	def update(self, update):
		self.updated = update

	def is_updated(self, update):
		return update == self.updated
	
	def get_xid(self):
		return self.xid
	def get_name(self):
		return self.name
	def get_group_name(self):
		return self.group_name
	def get_group(self):
		return self.group


class PanelIcon(Clutter.Group):
	def __init__(self, panel, icon, ico_size, margin = 2/3):
		super().__init__()
		self.panel = panel
		self.icon_size = ico_size
		self.sub_icon_size = ico_size*margin
		self.sub_offset = (self.icon_size-self.sub_icon_size)/2
		self.icon = icon
		self.icon.set_size(self.sub_icon_size,self.sub_icon_size)

		self.icon_back = ItemMenu()
		self.icon_back.set_size(ico_size,ico_size)
		self.icon_back.set_reactive(True)

		#Enable animation
		#self.save_easing_state();
		#self.set_easing_mode(Clutter.AnimationMode.EASE_IN_OUT_CUBIC);
		#self.set_easing_duration(100);

		self.add_child(self.icon_back)
		self.add_child(self.icon)

	def set_position(self,x, y):
		self.icon.set_position(x+self.sub_offset,y+self.sub_offset)
		self.icon_back.set_position(x,y)


class PanelShutdown(PanelIcon):
	def __init__(self, panel, ico_size):
		super().__init__(panel,  Clutter.Texture.new_from_file("./data/logout.png"), ico_size, 1)
		
		self.icon_back.connect("button-press-event", PanelShutdown.button_press_handler, self)
		

	def button_press_handler(widget, event, self):
		menu_list = [
			{'text':"Shutdown",'obj':Shutdown()},
			{'text':"Log Out",'obj':Logout()},
			{'text':"Sleep",'obj':Sleep()},
			{'text':"Reboot",'obj':Reboot()},
			]

		self.panel.panel_menu.show_menu(self.panel.panel_width, self.icon_back.get_y(), menu_list, event.time)	
		self.panel.panel_menu.window.focus(event.time)

class PanelGroupApp(PanelIcon):
	def __init__(self, panel, name, icon, ico_size):
		super().__init__(panel, icon, ico_size)
		self.app_list = {}
		self.name = name
		self.locked = False

		self.icon_back.connect("button-press-event", PanelGroupApp.button_press_handler, self)

		#self.icon_text = Clutter.Text.new_full(font_menu_entry, "", Clutter.Color.new(255,255,255,255))
		#self.add_child(self.icon_text)

	def set_position(self,x, y):
		super().set_position(x,y)
		#self.icon_text.set_text(str(len(self.app_list)))
		#self.icon_text.set_position(x+5,y+5)
		
	

	def unregister(self):
		print("UNREG")
		self.hide_all()
		self.destroy_all_children()
		self.get_parent().remove_child(self)
		print(self.app_list)
			


	def __del__(self):
		print("DESTROY")

	def add(self, iapp):
		print("Adding "+str(iapp.get_xid()) +" to "+ self.name)
		self.app_list[iapp.get_xid()] = iapp

	def remove(self, iapp):
		self.app_list.pop(iapp.get_xid())

	def is_empty(self):
		return len(self.app_list) == 0

	def is_locked(self):
		return self.locked

	def get_name(self):
		return self.name


	def button_press_handler(widget, event, self):
		if event.button == 1:
			if len(self.app_list) > 1:
				menu_list = []
				for k,iapp in self.app_list.items():
					menu = {}
					menu['text'] = iapp.get_name()
					menu['obj'] = iapp
					menu_list.append(menu)

				self.panel.panel_menu.show_menu(self.panel.panel_width, self.icon_back.get_y(), menu_list, event.time)	
				self.panel.panel_menu.window.focus(event.time)
			else:
				self.panel.panel_menu.hide_menu()
				for k,iapp in self.app_list.items():
					iapp.activate(event.time)
					break

	def __str__(self):
		#print(self.icon)
		tmp = "- "
		tmp += self.name
		tmp += ":\n"
		for k,iapp in self.app_list.items():
			tmp += "\t"			
			tmp += iapp.get_name()
			tmp += "\n"
		return tmp


class PanelView(Clutter.Stage):
	def __init__(self):
		super().__init__()

		# Manualy create the window to setup properties before mapping the window		
		self._create_panel_window()

		self.ico_size = 64
		self.margin = 2
		self.panel_width = self.ico_size+2*self.margin

		display = ClutterGdk.get_default_display()
		root_height = display.get_default_screen().get_root_window().get_height()

		# tricks to create the window
		ClutterGdk.set_stage_foreign(self, self.window)
		self.window.set_type_hint(Gdk.WindowTypeHint.DOCK)
		self.window.stick()
		PageLauncherHook.set_strut(self.window, [self.panel_width, 0, 0, 0])
		self.set_size(self.panel_width,root_height)
		self.set_user_resizable(False)
		self.set_title("page-panel")
		self.set_use_alpha(True)
		self.set_opacity(0)
		self.set_color(Clutter.Color.new(0,0,0,0))
		self.set_scale(1.0, 1.0)

		self.connect('button-press-event', PanelView.button_press_handler, self)
		# create dash view
		self.dash = DashView(self)
		self.panel_menu = PanelMenu(self)
		#
		

		# Dictionnary of apps
		self.dict_apps={}
		self.list_group_apps=[]
		self.list_sys_apps=[]

		self.list_sys_apps.append(PanelShutdown(self, self.ico_size))

		self.update_cnt = 0
		self.pos_offset = 0


		self.rect = Clutter.Actor()
		self.rect.set_x(0)
		self.rect.set_y(0)
		self.rect.set_size(self.panel_width,root_height)	
		self.rect.set_background_color(Clutter.Color.new(32,32,32,255))	
		#self.rect.set_background_color(Clutter.Color.new(255,255,0,0))
		self.add_child(self.rect)
		
		for app in self.list_sys_apps:
			self.rect.add_child(app)

		self.wnck_screen = Wnck.Screen.get_default()
		self.update_current_apps()
		self.show()

		GObject.timeout_add(1000, self.refresh_timer, self)		

	
	def refresh_timer(self, *arg):
		self.update_current_apps()
		return True

	def find_ico(self, name):
		ret = self.dash.apps.filter_apps(name)
		if ret:
			return ret[0].get_icon(self.ico_size)
		else:
			return None

	def update_current_apps(self):	
		self.wnck_screen.force_update()
		apps = self.wnck_screen.get_windows_stacked()
		#print(apps)
		#for app in apps:
		#	print('----')
		#	print(app.get_name())
		#	#print(app.get_icon_name())
		#	print(app.get_class_group_name())
		#	#print(app.get_class_instance_name())
		#	print(app.get_xid())
		#	##print(app.get_session_id_utf8())
		#	print(app.is_sticky())

		self.update_cnt += 1
		# Loop on Wnck apps
		for app in apps:
			if not app.is_sticky():
				#print(app.get_xid())
				xid = app.get_xid()
				name = app.get_name()
				group_name = app.get_application().get_name()

				# Check if app is already in list
				if xid in self.dict_apps:
					iapp = self.dict_apps[xid]
				# New app
				else:
					print('Create new app:' + str(xid) + "(" + group_name + ")")
					# Create group if does not exist
					grp = None
					for group in self.list_group_apps:
						if group_name == group.get_name():
							grp = group
							break

					if grp == None:
						#print(app.get_icon().get_width())
						# Get icon path
						ico = self.find_ico(app.get_application().get_icon_name())
						if ico==None:
							ico = self.find_ico(app.get_application().get_name())
						if ico==None:
							pix=app.get_icon()
							pix= pix.scale_simple(128,128,GdkPixbuf.InterpType.BILINEAR)
							ico = Clutter.Texture.new()
							data = pix.get_pixels()
							width = pix.get_width ()
							height = pix.get_height()
							if pix.get_has_alpha():
								bpp = 4
							else:
								bpp = 3
							rowstride = pix.get_rowstride()
							ico.set_from_rgb_data(data, pix.get_has_alpha(), width, height, rowstride, bpp,0 );
							#ico_data=  pix.get_pixels_array()

						print('Create new group:' + str(group_name))
						grp = PanelGroupApp(self,group_name,ico,self.ico_size)
						
						print(sys.getrefcount(grp))
						self.rect.add_child(grp)
						print(sys.getrefcount(grp))
						self.list_group_apps.append(grp)
						print(sys.getrefcount(grp))
	
					# Append app to dict
					print(sys.getrefcount(grp))					
					iapp=PanelApp(xid, name, grp)
					print(sys.getrefcount(grp))
					self.dict_apps[xid] = iapp
				
				
				iapp.update(self.update_cnt)

						
		# Delete app not updated
		list_del = []
		for xid, iapp in self.dict_apps.items():
			if not iapp.is_updated(self.update_cnt):
				# remove from group
				iapp.get_group().remove(iapp)
				iapp.unregister()
				list_del.append(xid)
				print('Deleting app:' + str(xid))

		for xid in list_del:			
			self.dict_apps.pop(xid)
			


		# Delete empty group		
		list_del = []
		for grp in self.list_group_apps:	
			if grp.is_empty() and not grp.is_locked():
				list_del.append(grp)
				print('Deleting group:' + str(grp.get_name()))
		for grp in list_del:
			
			print('Deleting group II :' + str(grp.get_name()))
			print(sys.getrefcount(grp))
			for r in gc.get_referrers(grp):
				pprint.pprint(r)	
			self.list_group_apps.remove(grp)	
			grp.unregister()
			print(sys.getrefcount(grp))
			for r in gc.get_referrers(grp):
				pprint.pprint(r)
	
		
		
		print('=====')
		for grp in self.list_group_apps:
			print(grp)

		# Update icon position
		pos_y = self.pos_offset
		for grp in self.list_group_apps:
			grp.set_position(self.margin, pos_y+self.margin)
			pos_y += self.ico_size+self.margin

		# Update icon position
		pos_y = self.get_height()
		for grp in self.list_sys_apps:
			pos_y -= grp.icon_size+self.margin
			print(pos_y)			
			grp.set_position(self.margin, pos_y+self.margin)
			

	def button_press_handler(widget, event, self):
		if event.button == 1:
			print ('pouet')
			#self.panel_menu.hide_menu()
			#self.dash.show(event.time)
			#self.dash.window.focus(event.time)
		elif event.button == 3:
			Clutter.main_quit()

	def run(self):
		self.show()
		self.window.move(0,0)
		Clutter.main()
		
	def _create_panel_window(self):
		display = ClutterGdk.get_default_display()
		root_height = display.get_default_screen().get_root_window().get_height()
	
		attr = Gdk.WindowAttr();
		attr.title = "page-panel"
		attr.width = 32
		attr.height = root_height
		attr.x = 0
		attr.y = 0
		attr.event_mask = 0
		attr.window_type = Gdk.WindowType.TOPLEVEL
		attr.visual = display.get_default_screen().get_rgba_visual()
		self.window = Gdk.Window(None, attr,
		 Gdk.WindowAttributesType.TITLE
		|Gdk.WindowAttributesType.VISUAL
		|Gdk.WindowAttributesType.X
		|Gdk.WindowAttributesType.Y)


#####
####
def dbus_login1(bus):
	bus_object = bus.get_object("org.freedesktop.login1", "/org/freedesktop/login1")
	iface = dbus.Interface(bus_object, 'org.freedesktop.login1.Manager')
	return iface
def dbus_upower(bus):
	bus_object = bus.get_object("org.freedesktop.UPower", "/org/freedesktop/UPower")
	iface = dbus.Interface(bus_object, 'org.freedesktop.UPower')
	return iface
def dbus_consolekit(bus):
	bus_object = bus.get_object("org.freedesktop.ConsoleKit", "/org/freedesktop/ConsoleKit/Manager")
	iface = dbus.Interface(bus_object, 'org.freedesktop.ConsoleKit.Manager')
	return iface
def dbus_lightdm(bus):
	seat_path=os.environ.get('XDG_SEAT_PATH')
	bus_object = bus.get_object("org.freedesktop.DisplayManager", seat_path)
	iface = dbus.Interface(bus_object, 'org.freedesktop.DisplayManager.Seat')
	return iface

class Sleep():
	def activate(self, event_time):
		bus = dbus.SystemBus()
		try:
			i = dbus_login1(bus)
			if i.CanHybridSleep()=="yes":
				print("sleep")
				i.HybridSleep(0)
			elif i.CanHibernate()=="yes":
				print("hibernate")
				i.Hibernate(0)
			elif i.CanSuspend()=="yes":
				print("suspend")
				i.Suspend(0)
			else:
				raise Exception("'org.freedesktop.login1' somehow doesn't work.")
		except:
			i=dbus_upower(bus)
			if i.HibernateAllowed():
				print ("hibernate")
				i.Hibernate(0)
			elif i.SuspendAllowed():
				print ("suspend")
				i.Suspend(0)
			else:
				print("Error: could not sleep, sorry.")

class Shutdown():
	def activate(self, event_time):
		bus = dbus.SystemBus()
		try:
			i = dbus_login1(bus)
			if i.CanPowerOff()=="yes":
				print("power off")
				i.PowerOff(0)
			else:
				raise Exception("'org.freedesktop.login1' somehow doesn't work.")
		except:
			i  = dbus_consolekit(bus)
			if i.CanStop():
				print("shutdown")
				i.Stop(0)
			else:
				raise Exception("'org.freedesktop.Consolekit' somehow doesn't work.")
class Reboot():
	def activate(self, event_time):
		bus = dbus.SystemBus()
		try:
			i = dbus_login1(bus)
			if i.CanReboot()=="no":
				print("reboot")
				i.Reboot(0)
			else:
				raise Exception("'org.freedesktop.login1' somehow doesn't work.")
		except:
			i  = dbus_consolekit(bus)
			if i.CanRestart():
				print("reboot")
				i.Restart(0)
			else:
				raise Exception("'org.freedesktop.Consolekit' somehow doesn't work.")
class Logout():
	def activate(self, event_time):
		bus = dbus.SystemBus()
		try:
			i= dbus_lightdm(bus)
			print("switch")
			i.SwitchToGreeter(0)
		except:
			raise Exception("logout somehow doesn't work.")

if __name__ == '__main__':
 Gdk.init(sys.argv)
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

 panel = PanelView()
 panel.run()

