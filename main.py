#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys

import datetime #Clock

import signal
import time

import gc # debug
import pprint # debug 

import configparser #Save Config

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


font_comment = "Sans 8"
color_comment = Clutter.Color.new(200,200,200,255)

font_dash = "Sans Bold 12"
color_apps = Clutter.Color.new(255,255,255,255)

font_entry = "Sans Bold 15"
color_entry = Clutter.Color.new(255,255,255,255) # red,green,blue,alpha


font_menu_entry = "Sans Bold 10"
color_menu_entry = Clutter.Color.new(255,255,255,255) # red,green,blue,alpha

font_icon_entry = "Sans Bold 8"
color_icon_entry = Clutter.Color.new(128,128,128,255) # red,green,blue,alpha

font_clock = "Sans Bold 12"
color_clock = Clutter.Color.new(255,255,255,255) # red,green,blue,alpha


def sig_int_handler(n):
 Clutter.main_quit()


class apps_entry:
 def __init__(self, parent, de):
  self.parent = parent
  self.ico_size = parent.ico_size
  self.name = de.getName().lower()
  self.generic_name = de.getGenericName().lower()
 # print("XX1"+self.generic_name)
  #print("XX e:"+de.getExec()+" g:"+self.generic_name+" i:"+de.getIcon()+" n:"+self.name)
  self.comment = de.getComment().lower()
  self.exe = re.sub(u"%\w*", u"", de.getExec())
  #print(self.exe)
  self.icon_name = de.getIcon().lower()
  self.icon = self._find_icon(de.getIcon(),self.ico_size)

  self.icon_offset_y = (parent.item_size_y-self.ico_size)/2
  self.icon_offset_x = parent.item_size_x-self.ico_size-self.icon_offset_y
  self.text_offset_x = self.icon_offset_y
  self.text_offset_y = self.icon_offset_y

  self.icon.set_size(parent.ico_size,parent.ico_size)
  #self.icon.set_size(self.ico_size,self.ico_size)

  self.text = Clutter.Text.new_full(font_dash, de.getName(), color_apps)
  self.text.set_width(self.icon_offset_x - self.icon_offset_y)
  self.text.set_ellipsize(Pango.EllipsizeMode.END)
  self.text.set_line_alignment(Pango.Alignment.LEFT)
  self.rect = ItemMenu()  
  #self.rect = Clutter.Rectangle.new()
  self.rect.set_size(parent.item_size_x,parent.item_size_y)
  #self.rect.set_color(Clutter.Color.new(255,255,255,128))
  #self.rect.set_reactive(True)
  #self.rect.set_opacity(0)
  #self.rect.connect("enter-event", self.enter_handler, self)
  #self.rect.connect("leave-event", self.leave_handler, self)
  self.rect.connect("button-press-event", apps_entry.button_press_handler, self)
  #self.fade_in_transition = Clutter.PropertyTransition.new("opacity")
  #self.fade_in_transition.set_duration(100)
  #self.fade_in_transition.set_to(255)
  #self.fade_in_transition.connect("completed", self.fade_in_completed, self)
  #self.fade_out_transition = Clutter.PropertyTransition.new("opacity")
  #self.fade_out_transition.set_duration(2000)
  #self.fade_out_transition.set_to(0)
  #self.fade_out_transition.connect("completed", self.fade_out_completed, self)


  self.text_comment_offset_x = 2*self.icon_offset_y
  self.text_comment_offset_y = self.icon_offset_y+self.text.get_height()+parent.margin
  self.text_comment = Clutter.Text.new_full(font_comment, self.comment, color_comment)
  self.text_comment.set_width(self.icon_offset_x - self.icon_offset_y - self.text_comment_offset_x)
  self.text_comment.set_height(self.ico_size-self.text_comment_offset_y+self.icon_offset_y)
  self.text_comment.set_line_wrap(True)

  self.parent.rect.add_child(self.rect)
  self.parent.rect.add_child(self.icon)
  self.parent.rect.add_child(self.text)
  self.parent.rect.add_child(self.text_comment)
  self.hide()
  pass

 #def enter_handler(self, widget, event, self1):
 # if self.rect.get_transition("fade_out"):
 #  self.rect.remove_transition("fade_out")
 # if not self.rect.get_transition("fade_in"):
 #  self.rect.add_transition("fade_in", self.fade_in_transition)
 # return True
 #def leave_handler(self, widget, event, self1):
 # if self.rect.get_transition("fade_in"):
 #  self.rect.remove_transition("fade_in")
 # if not self.rect.get_transition("fade_out"):
 #  self.rect.add_transition("fade_out", self.fade_out_transition)
 # return True

 def get_icon(self):
  return self._find_icon(self.icon_name, self.ico_size)

 def _find_icon(self, icon_name, size = 48):
  icon_theme = Gtk.IconTheme.get_default()
  gtk_icon_info = icon_theme.lookup_icon(icon_name, size, 0)
  if gtk_icon_info:
   icon_path = gtk_icon_info.get_filename()
  else:
   icon_path = getIconPath(icon_name, size)

  if (icon_path and re.match("^.*\.svg$", icon_path)) or not icon_path:
   gtk_icon_info = icon_theme.lookup_icon("exec", size, 0)
   if gtk_icon_info:
    icon_path = gtk_icon_info.get_filename()
   else:
    icon_path = getIconPath(icon_name, size)
  return Clutter.Texture.new_from_file(icon_path)

 def show(self):
  self.rect.show()
  self.icon.show()
  self.text.show()
  self.text_comment.show()
  pass
 def hide(self):
  self.rect.hide()
  self.icon.hide()
  self.text.hide()
  self.text_comment.hide()
  pass

 def set_position(self, x, y):
  self.rect.set_position(x,y)
  self.icon.set_position(x+self.icon_offset_x,y+self.icon_offset_y)
  self.text.set_position(x+self.text_offset_x,y+self.text_offset_y)
  self.text_comment.set_position(x+self.text_comment_offset_x,y+self.text_comment_offset_y)
  pass

 #def fade_in_completed(self, transition, self1):
 # if self.rect.get_transition("fade_in"):
 #  self.rect.remove_transition("fade_in")
 # self.rect.set_opacity(255)
 # pass
 #def fade_out_completed(self, transition, self1):
 # if self.rect.get_transition("fade_out"):
 #  self.rect.remove_transition("fade_out")
 # self.rect.set_opacity(0)
 # pass

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
  print("pattern:"+patern)
  p = re.compile(patern.lower())
  ret = list()
  for de in self._apps:
   if p.search(de.name):
    ret.append(de)
   elif p.search(de.generic_name):
    ret.append(de)
   elif p.search(de.icon_name):
    ret.append(de)
   elif p.search(de.comment):
    ret.append(de)
  return ret

 def match_apps(self, patern):
  print("pattern:"+patern)
  p = re.compile("^"+patern.lower()+"$")
  ret = list()
  for de in self._apps:
   if p.match(de.icon_name):
    ret.append(de)
   elif p.match(de.name):
    ret.append(de)
   elif p.match(de.exe):
    ret.append(de)
  return ret


 def hide_all(self):
  for x in self._apps:
   x.hide()

 pass

class launcher_layout:
 def __init__(self, stage, napps):
   self.size_x = stage.item_size_x
   self.size_y = stage.item_size_y + stage.margin
   self.y_offset = stage.y_offset
   self.width = stage.get_width()
   self.height = stage.get_height() - self.y_offset
   self.columns = int(floor(self.width/(self.size_x)))
   self.rows = int(floor(self.height/(self.size_y)))
   #self.left_margin = (self.width-(self.columns*self.size*1.5*1.3))/2.0
   self.top_margin = (self.height-(self.rows*self.size_y))
   #npages = int(floor(napps/self.columns*self.rows)+1.0)
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

class SubWindow(Clutter.Stage):
	def __init__(self, parent, x, y, width, height):
		super().__init__()
		self.is_grab = False
		self._create_menu_window(x, y, width, height)
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

	def _create_menu_window(self,x, y, width, height):
		display = ClutterGdk.get_default_display()
		self.root_height = display.get_default_screen().get_root_window().get_height()
	
		attr = Gdk.WindowAttr();
		attr.title = "sub-win"
		attr.width = width
		attr.height = height
		attr.x = x
		attr.y = y
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

class Slide(SubWindow):
	def __init__(self, parent, offset_x, size_x):
		self.parent = parent
		display = ClutterGdk.get_default_display()
		self.root_height = display.get_default_screen().get_root_window().get_height()
		print(size_x)
		print(self.root_height)
		super().__init__(parent, offset_x, 0, size_x, self.root_height)


	def show(self, event_time):
		print("Dash show")
		parent_window = ClutterGdk.get_stage_window(self.parent)
		self.window.set_transient_for(parent_window)
		self.reset()
		super().show()
		self.window.focus(event_time)
		self.set_move(0)
		


class DashSlide(Slide):
	def __init__(self, parent, offset_x):
		self.slide_size = 300
		super().__init__(parent, offset_x, self.slide_size)

		self.margin_x = 10
		self.text_size = 64
		self.margin = 2
		self.y_offset = self.text_size + 2*self.margin
		self.item_size_x = self.slide_size
		self.item_size_y = 64
		self.ico_size = 48

		self.rect = Clutter.Actor()
		self.rect.set_x(-self.slide_size)
		self.rect.set_y(0)
		self.rect.set_size(self.slide_size,self.root_height)
		self.rect.set_background_color(Clutter.Color.new(32,32,32,240))	
		
		#Enable animation
		self.rect.save_easing_state();
		self.rect.set_easing_mode(Clutter.AnimationMode.EASE_IN_OUT_CUBIC);
		self.rect.set_easing_duration(1000);

		self.add_child(self.rect)

		self.pos = 200

		#init slide 
		self.back_text = ItemMenu()
		self.back_text.set_x(0)
		self.back_text.set_y(self.root_height-self.text_size-self.margin)
		self.back_text.set_size(self.slide_size, self.text_size)

		self.intext = Clutter.Text.new_full(font_entry, u"", color_entry)
		self.apps = apps_handler(self)
		self.intext.set_editable(True)
		self.intext.set_selectable(True)
		self.intext.set_activatable(True)

		self.notext = Clutter.Text.new_full(font_entry, u"Enter Text Here ...",
Clutter.Color.new(255,255,255,128))
		self.text_offset = (self.text_size-self.notext.get_height())/2

		self.notext.set_x(self.margin_x)
		self.notext.set_y(self.root_height-self.text_size + self.text_offset-self.margin)
		self.notext.set_width(self.slide_size-self.margin_x*2)

		self.intext.set_x(self.margin_x)
		self.intext.set_y(self.root_height-self.text_size + self.text_offset-self.margin)
		self.intext.set_width(self.slide_size-self.margin_x*2)		

		self.notext.show()
		self.intext.show()

		self.rect.add_child(self.notext)
		self.rect.add_child(self.intext)
		self.rect.add_child(self.back_text)

		self.apps.hide_all()
		self.set_key_focus(self.intext)

		
		# check for enter or Escape
		self.intext.connect("key-press-event", self.key_press_handler)
		self.intext.connect('text-changed', self.handle_text_changed)

		self.connect('button-press-event', self.button_press_handler)		
		self.connect('key-press-event', self.key_press_handler)
		self.connect('allocation-changed', self.allocation_changed_handler)
		#self.show()

	def reset(self):
		self.rect.restore_easing_state();
		self.rect.set_x(-self.slide_size)
		self.rect.save_easing_state();

	def set_move(self, x):
		self.set_key_focus(self.intext)
		self.rect.set_x(x)

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
		#print("XXXX"+text)
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
			a.set_position(c*layout.size_x,(layout.rows-l-1)*layout.size_y+self.margin+layout.top_margin)
			a.show()
			self.current_actor.append(a)
		pass

	def allocation_changed_handler(self, box, flags, data):
		# indirectly update the layout:
		self.handle_text_changed(data)
		pass
		
	def _start_grab(self, time):
		if self.is_grab:
			return
		print("START GRABBING")
		self.is_grab = True
		dpy = ClutterGdk.get_default_display()
		dm = dpy.get_device_manager()
		dev = dm.list_devices(Gdk.DeviceType.MASTER)
		for d in dev:
		 # grab keyboard until the dash is hiden
		 if d.get_source() == Gdk.InputSource.KEYBOARD:
		  d.grab(self.window, Gdk.GrabOwnership.WINDOW, True,
		  Gdk.EventMask.KEY_PRESS_MASK
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
		 
	def show(self, time):
		super().show(time)
		self._start_grab(time)
		
	def hide(self):
		self._stop_grab()
		super().hide()
		
		
class PanelMenu(SubWindow):
	def __init__(self, parent):
		super().__init__(parent, 0, 0, 100, 100)
		
		self.is_grab = False
		self.item_height = 31
		self.item_margin_y = 2
		self.item_margin_x = 20
		self.global_width = 0
		self.global_height = 0
		#self.connect("deactivate", PanelMenu.event_menu_focus_out, self)
	
		#elf.container = Clutter.Group()
			

	def event_menu_button_press(widget, event, self,  menu):
		print('PANEL KEYPRESS')
		if 'cb' in menu:
			menu['cb'](event.time)
		else:
			menu['obj'].activate(event.time)
		self.hide_menu()
		pass

	def event_menu_focus_out(widget, event, self):
		print('FOCUS OUT')
		pass

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
		#print(self.global_width)
		#print(self.global_height)

		if self.root_height < self.global_height+y:
			y=self.root_height-self.global_height
		
		self.window.resize(self.global_width, self.global_height)
		self.window.move(x,y)


		self.set_size(self.global_width, self.global_height)	
		self.rect.set_size(self.global_width, self.global_height)	
		
		self.show_all()
		# Focus can only apply on visible window /!\
		self.window.focus(event_time)
		pass

	def hide_menu(self):
		self.hide()
		self.remove_all()


class PanelApp():
	def __init__(self, xid, name, group ,pid):
		self.xid = xid
		self.updated = 0
		self.name = name
		self.pid = pid
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

	def kill(self):
		os.kill(int(self.pid), signal.SIGTERM)
	

	def set_name(self, name):
		self.name = name

	def get_xid(self):
		return self.xid
	def get_name(self):
		return self.name
	def get_group_name(self):
		return self.group_name
	def get_group(self):
		return self.group


class PanelIcon(Clutter.Group):
	def __init__(self, panel, icon, ico_size, sub_ico_size):
		super().__init__()
		self.panel = panel
		self.icon_size_x = ico_size
		self.icon_size_y = ico_size
		self.sub_icon_size = sub_ico_size
		self.sub_offset = (self.icon_size_x-self.sub_icon_size)/2
		self.icon = icon
		self.icon.set_size(self.sub_icon_size,self.sub_icon_size)

		self.icon_back = ItemMenu()
		self.icon_back.set_size(self.icon_size_x,self.icon_size_y)
		self.icon_back.set_reactive(True)
		

		self.icon_back.connect("button-press-event", self.button_press_handler)
		
		self.icon.set_position(self.sub_offset,self.sub_offset)
		self.icon_back.set_position(0,0)


		self.add_child(self.icon_back)
		self.add_child(self.icon)
		#Enable animation
		self.save_easing_state();
		self.set_easing_mode(Clutter.AnimationMode.EASE_IN_OUT_CUBIC);
		self.set_easing_duration(100);


class PanelClock(Clutter.Group):
	def __init__(self, panel, ico_size):
		super().__init__()
		self.panel = panel
		self.icon_size_x = ico_size
		self.icon_size_y = ico_size/2
		self.sub_offset = 2
		#self.sub_icon_size = ico_size*margin
		#self.sub_offset = (self.icon_size-self.sub_icon_size)/2
		self.text = Clutter.Text.new_full(font_clock, u"12:30",
color_clock)
		#self.text.set_line_alignment(Pango.Alignment.CENTER)
		#self.text.set_size(ico_size, ico_size/2)

		self.icon_back = ItemMenu()
		self.icon_back.set_size(self.icon_size_x,self.icon_size_y)
		self.icon_back.set_position(0,0)

		self.add_child(self.icon_back)
		self.add_child(self.text)

		
	
	def set_position(self,x, y):
		super().set_position(x,y)
		now = datetime.datetime.now()
		self.text.set_text(now.strftime('%H:%M'))
		self.sub_offset_x = (self.icon_size_x-self.text.get_width())/2
		self.sub_offset_y = (self.icon_size_y-self.text.get_height())/2
		self.text.set_position(self.sub_offset_x,self.sub_offset_y)


class PanelApps(PanelIcon):
	def __init__(self, panel, ico_size):
		super().__init__(panel,  Clutter.Texture.new_from_file("./data/open.svg"), ico_size, 48)
		
	
	def button_press_handler(self, widget, event):
		self.panel.sub_dash(event.time)

class PanelShutdown(PanelIcon):
	def __init__(self, panel, ico_size):
		super().__init__(panel,  Clutter.Texture.new_from_file("./data/shutdown.svg"), ico_size, 48)
		

	def button_press_handler(self, widget, event):
		menu_list = [
			{'text':"Shutdown",'obj':Shutdown()},
			{'text':"Log Out",'obj':Logout()},
			{'text':"Lock",'obj':Lock()},
			{'text':"Sleep",'obj':Sleep()},
			{'text':"Reboot",'obj':Reboot()},
			]

		self.panel.sub_menu(self.get_y(), menu_list, event.time)	



class PanelGroupApp(PanelIcon):
	def __init__(self, panel, name, icon, ico_size, new_process, locked):
		self.app_list = {}
		self.name = name
		self.locked = locked
		self.icon_1px = None
		self.cb_new_process = new_process
		#self.icon_1px.set_depth(100)
		

		super().__init__(panel, icon, ico_size, 48)
		
		#self.icon_text = Clutter.Text.new_full(font_menu_entry, "", Clutter.Color.new(255,255,255,255))
		#self.add_child(self.icon_text)

	def set_background(self, px1_ico):
		self.icon_1px = px1_ico
		self.icon_1px.set_size(self.icon_size_x,self.icon_size_x)
		self.icon_1px.set_opacity(100)
		self.icon_1px.hide()
		self.insert_child_above(self.icon_1px, self.icon_back)

	def get_background(self):
		return self.icon_1px

	def lock(self, event_time):
		if self.cb_new_process:
			self.locked = True
		self.panel.config_save()

	def unlock(self, event_time):
		self.locked = False

	def process_new(self, event_time):
		self.cb_new_process()
	def process_close(self, event_time):
		for k,iapp in self.app_list.items():
			iapp.kill()


	def set_position(self,x, y):
		super().set_position(x,y)
		if self.icon_1px:
			self.icon_1px.set_position(0,0)
		#self.icon_text.set_text(str(len(self.app_list)))
		#self.icon_text.set_position(x+5,y+5)
		
	

	def unregister(self):
		print("UNREG")
		self.hide_all()
		self.destroy_all_children()
		self.get_parent().remove_child(self)
		#print(self.app_list)
			


	def __del__(self):
		print("DESTROY")

	def add(self, iapp):
		print("Adding "+str(iapp.get_xid()) +" to "+ self.name)
		self.app_list[iapp.get_xid()] = iapp
		if self.icon_1px:
			self.icon_1px.show()

	def remove(self, iapp):
		self.app_list.pop(iapp.get_xid())
		if self.is_empty():
			if self.icon_1px:
				self.icon_1px.hide()
		

	def is_empty(self):
		return len(self.app_list) == 0

	def is_locked(self):
		return self.locked

	def get_name(self):
		return self.name

	def button_press_handler(self, widget, event):
		print('PanelGroupApp.button_press_handler', event)
		if event.button == 1:
			if len(self.app_list) > 1:
				menu_list = []
				for k,iapp in self.app_list.items():
					menu = {}
					menu['text'] = iapp.get_name()
					menu['obj'] = iapp
					menu_list.append(menu)
				self.panel.sub_menu(self.get_y(), menu_list, event.time)	
			elif len(self.app_list) == 1:
				self.panel.sub_reset(event.time)
				for k,iapp in self.app_list.items():
					iapp.activate(event.time)
					return True
			else:
				self.process_new(event.time)
				return True
		elif event.button == 3:
			menu_list = []
			if self.locked:
				menu_list.append({'text':"Unlock", 'cb':self.unlock})
			else:				
				menu_list.append({'text':"Lock",'cb':self.lock})

			menu_list.append({'text':"New Instance",'cb':self.process_new})
			if len(self.app_list) >= 1:
				menu_list.append({'text':"Terminate",'cb':self.process_close})
				
			self.panel.sub_menu(self.get_y(), menu_list, event.time)
			return True
		return False


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
		
		screen = Wnck.Screen.get_default()
		screen.connect("active-window-changed", self.on_active_window_change)
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
		#self.dash = DashView(self)
		self.panel_menu = PanelMenu(self)
		self.dash_slide = DashSlide(self, self.panel_width)
		

		# Dictionnary of apps
		self.dict_apps={}
		self.list_group_apps=[]
		self.list_sys_apps=[]


		self.list_sys_apps.append(PanelClock(self, self.ico_size))
		self.list_sys_apps.append(PanelApps(self, self.ico_size))
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

	
		self.config_file =os.path.join(os.path.expanduser("~"),'.page-launcher.cfg')
	
	
		self.config_read()
		self.update_current_apps()
		self.show()

		GObject.timeout_add(1000, self.refresh_timer, self)		

	def config_save(self):
		config = configparser.ConfigParser()
		config.add_section('Launcher')
		apps_locked_list = []
		for group in self.list_group_apps:
			if group.is_locked():
				apps_locked_list.append(group.get_name())

		config.set('Launcher','apps' , ','.join(apps_locked_list))
		# Writing our configuration file to 'example.cfg'
		with open(self.config_file, 'w') as configfile:
			config.write(configfile)

	def config_read(self):
		config = configparser.ConfigParser()
		config.read(self.config_file)
		if config.has_section('Launcher'):
			apps_locked_list = config.get('Launcher','apps').split(',')
			for group_name in apps_locked_list:
				self.create_app(group_name, None, True)
			
			

	def refresh_timer(self, *arg):
		self.update_current_apps()
		return True

	def find_app(self, name):
		ret = self.dash_slide.apps.match_apps(name)
		if ret:
			#print(ret[0].name)
			return ret[0]
		else:
			return None

	def create_app(self, group_name, pix_icon, locked):
		#print(app.get_icon().get_width())
		# Get icon path
		appdict = self.find_app(group_name);
		if appdict:
			cb_new_process = appdict.call
			ico = appdict.get_icon()
		else:
			ico = None
			cb_new_process = None
		#if ico==None:
		#	ico = self.find_ico(app.get_application().get_name())
		if ico==None and pix_icon:
			pix= pix_icon.scale_simple(64,64,GdkPixbuf.InterpType.HYPER)
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
			#ico.set_width(48,48)
			#ico_data=  pix.get_pixels_array()
		assert(ico != None)
	

		#print('Create new group:' + str(group_name))
		grp = PanelGroupApp(self,group_name,ico,self.ico_size, cb_new_process, locked)
		
		#print(sys.getrefcount(grp))
		self.rect.add_child(grp)
		#print(sys.getrefcount(grp))
		self.list_group_apps.append(grp)
		#print(sys.getrefcount(grp))

		return grp


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
				group_name = app.get_class_group_name()
				#dir(app)	
				#print('---')
				#print(app.get_icon_name())
				#print(app.get_application().get_name())
				#print(app.get_application().get_pid())
				#print(app.get_class_group_name())
				#print(app.get_class_instance_name())
				#print(group_name)

				# Check if app is already in list
				if xid in self.dict_apps:
					iapp = self.dict_apps[xid]
					iapp.set_name(app.get_name())
				# New app
				else:
					print('Create new app:' + str(xid) + "(" + group_name + ")")
					# Check group if it does exist
					grp = None
					for group in self.list_group_apps:
						if group_name == group.get_name():
							grp = group
							break
					# Create group
					if grp == None:
						grp = self.create_app(group_name, app.get_icon(), False)
					# Update background if needed
					if grp.get_background() == None:
						px_ico = Clutter.Texture.new()
						px_pix= app.get_icon().scale_simple(1,1,GdkPixbuf.InterpType.HYPER)
						if px_pix.get_has_alpha():
							px_bpp = 4
						else:
							px_bpp = 3
						px_ico.set_from_rgb_data(px_pix.get_pixels(), px_pix.get_has_alpha(), px_pix.get_width(), px_pix.get_height(), px_pix.get_rowstride(), px_bpp,0 );
						grp.set_background(px_ico)

					# Append app to dict
					#print(sys.getrefcount(grp))					
					iapp=PanelApp(xid=xid, name=name, group=grp, pid=app.get_application().get_pid())
					#print(sys.getrefcount(grp))
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
			
			#print('Deleting group II :' + str(grp.get_name()))
			#print(sys.getrefcount(grp))
			#for r in gc.get_referrers(grp):
			#	pprint.pprint(r)	
			self.list_group_apps.remove(grp)	
			grp.unregister()
			#print(sys.getrefcount(grp))
			#for r in gc.get_referrers(grp):
			#	pprint.pprint(r)
	
		
		
		#print('=====')
		#for grp in self.list_group_apps:
		#	print(grp)

		# Update icon position
		pos_y = self.pos_offset
		for grp in self.list_group_apps:
			grp.set_position(self.margin, pos_y+self.margin)
			pos_y += self.ico_size+self.margin

		# Update icon position
		pos_y = self.get_height()-self.margin
		for grp in self.list_sys_apps:
			pos_y -= grp.icon_size_y+self.margin
			#print(pos_y)			
			grp.set_position(self.margin, pos_y+self.margin)
			

	def button_press_handler(self, event, data = None):
		print('PanelView.button_press_handler', event)
		if event.button == 1:
			#print ('pouet')
			#self.panel_menu.hide_menu()
			#self.dash.show(event.time)
			#self.dash.window.focus(event.time)
			#self.dash_slide.show(event.time)
			#self.panel_menu.hide_menu()
			pass
		elif event.button == 3:
			#Clutter.main_quit()
			pass
		pass
			
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

	def on_active_window_change(self, screen, window):
		if(self.window.get_xid() != screen.get_active_window().get_xid()):
			# TODO HIDE ALL SUB WINDOWS
			self.sub_reset()

	def sub_reset(self, event_time = 0):
		self.dash_slide.hide()
		self.panel_menu.hide_menu()
		
		
	def sub_dash(self, event_time):
		self.dash_slide.show(event_time)
		self.panel_menu.hide_menu()

	def sub_menu(self, offset_y, menu_list, event_time):
		self.raise_window(event_time)
		self.dash_slide.hide()
		self.panel_menu.show_menu(self.panel_width, offset_y, menu_list, event_time)	
		self.panel_menu.window.focus(event_time)

	def raise_window(self, time):
		self.window.show()
		w = Wnck.Window.get(self.window.get_xid())
		if w != None:
			w.activate(time)

#####
####

def dbus_mate(bus):
	bus_object = bus.get_object("org.mate.sessionmanager", "/org/mate/SessionManager")
	iface = dbus.Interface(bus_object, 'org.mate.SessionManager')
	return iface
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
			i= dbus_mate(bus)
			print("lock")
			i.Logout(1)
		except:
			raise Exception("logout somehow doesn't work.")

class Lock():
	def activate(self, event_time):
		bus = dbus.SystemBus()
		try:
			i= dbus_lightdm(bus)
			print("switch")
			i.SwitchToGreeter(0)
		except:
			raise Exception("lock somehow doesn't work.")

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

