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
   if de.getType() == u"Application":
    self._apps.append(de)
  self._apps.sort(key=lambda x: x.getName().lower())
  pass

 def filter_apps(self, patern):
  p = re.compile(patern.lower())
  ret = list()
  for de in self._apps:
   if p.search(de.getName().lower()):
    ret.append((de.getName(),self.get_icon_actor(de)))
   elif p.search(de.getGenericName().lower()):
    ret.append((de.getName(),self.get_icon_actor(de)))
   elif p.search(de.getComment().lower()):
    ret.append((de.getName(),self.get_icon_actor(de)))
  return ret

 def get_icon_actor(self, de, size = 128):
  icon_name = de.getIcon()
  icon_path = getIconPath(icon_name, size)
  if not icon_path:
   icon_path = "apps.svg"
  if not de in self._icon_actor_cache:
   self._icon_actor_cache[de] = Clutter.Texture.new_from_file(icon_path)
  return self._icon_actor_cache[de]
 pass


def key_press_handler(widget, event, data):
 print(event.keyval)
 if event.keyval == 65307:
  Clutter.main_quit()
 pass

def button_press_handler(widget, event, data):
 widget = stage.get_actor_at_pos(Clutter.PickMode.ALL, event.x, event.y)
 if widget == intext:
  stage.set_key_focus(intext)
 else:
  stage.set_key_focus(intext)
 pass

current_actor = list()
apps_list = list()

def handle_text_changed(widget, data):
 global apps
 global apps_list
 global current_actor

 y_offset = intext.get_height()

 size = 128.0
 #outtext.set_text(get_apps_string(widget.get_text()))

 for a in current_actor:
  stage.remove_actor(a)

 width = stage.get_width()
 height = stage.get_height() - y_offset

 columns = int(floor(width/(size*1.3)))
 rows = int(floor(height/(size*1.5*1.3)))

 npages = int(floor(len(apps_list)/columns*rows)+1.0)

 current_actor = list()
 apps_list = apps.filter_apps(widget.get_text())

 for i in range(0, columns*rows):
  if i >= len(apps_list):
   break
  c = i - floor(i / columns)*columns
  l = floor(i / columns)
  name, icon = apps_list[i]
  icon_actor = icon
  icon_actor.set_position(c*size*1.3,l*1.5*1.3*size+y_offset)
  icon_actor.set_size(size,size)
  text_actor = Clutter.Text.new_full(font, name, Clutter.Color.new(0,255,0,255))
  text_actor.set_position(c*size*1.3,l*1.5*1.3*size+size+y_offset)
  text_actor.set_width(size)
  text_actor.set_ellipsize(Pango.EllipsizeMode.END)
  stage.add_actor(icon_actor)
  stage.add_actor(text_actor)
  current_actor.append(icon_actor)
  current_actor.append(text_actor)
 pass

if __name__ == '__main__':
 Clutter.init(sys.argv)
 
 stage = Clutter.Stage()
 stage.set_user_resizable(True)
 stage.set_title("page-launcher")
 stage.set_use_alpha(True)
 stage.set_opacity(0)
 stage.set_color(Clutter.Color.new(0,0,0,128))
 
 font = "Sans 20"
 green = Clutter.Color.new(255,0,0,255) # red,green,blue,alpha
 intext = Clutter.Text.new_full(font, "", green)
 intext.set_editable(True)
 intext.set_selectable(True)
 intext.set_activatable(True)
 Clutter.Container.add_actor(stage, intext)

 apps = apps_handler()

 handle_text_changed(intext, None)

 # Adding a rectangle
 #transparentBlue = Clutter.Color.new(0,0,255,100)
 #rectangle = Clutter.Rectangle.new_with_color(transparentBlue)
 #rectangle.set_size(150,50)
 #Clutter.Container.add_actor(stage, rectangle)

 # Adding a texture
 #picture = Clutter.Texture.new_from_file("flor.jpg")
 #picture.set_size( 400,400) 
 #Clutter.Container.add_actor( stage, picture)

 stage.set_key_focus(intext)

 #stage.connect('button-press-event', lambda x, y: print("pressed"))
 stage.connect('button-release-event', button_press_handler, None)
 stage.connect('key-press-event', key_press_handler, None)
 #stage.connect('motion-event', lambda x, y: print("motion"))
 stage.connect('destroy', lambda x: Clutter.main_quit())

 intext.connect('text-changed', handle_text_changed, None)

 stage.show_all()
 Clutter.main()

