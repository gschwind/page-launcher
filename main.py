#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys

import datetime  # Clock

import signal
import time

import gc  # debug
import pprint  # debug

import configparser  # Save Config

from math import *

from io import StringIO

import gi

gi.require_version('ClutterGdk', '1.0')
gi.require_version('GtkClutter', '1.0')
gi.require_version('Wnck', '3.0')
gi.require_version('Gtk', '3.0')

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
color_comment = Clutter.Color.new(200, 200, 200, 255)

font_dash = "Sans Bold 12"
color_apps = Clutter.Color.new(255, 255, 255, 255)

font_entry = "Sans Bold 15"
color_entry = Clutter.Color.new(255, 255, 255, 255)  # red,green,blue,alpha

font_menu_entry = "Sans Bold 10"
color_menu_entry = Clutter.Color.new(255, 255, 255, 255)  # red,green,blue,alpha

font_icon_entry = "Sans Bold 8"
color_icon_entry = Clutter.Color.new(128, 128, 128, 255)  # red,green,blue,alpha

font_clock = "Sans Bold 12"
color_clock = Clutter.Color.new(255, 255, 255, 255)  # red,green,blue,alpha


def sig_int_handler(n):
    Clutter.main_quit()


class apps_entry:
    def __init__(self, parent, de):
        self.parent = parent
        self.ico_size = parent.ico_size
        self.name = de.getName().lower()
        self.generic_name = de.getGenericName().lower()
        # print("XX1"+self.generic_name)
        # print("XX e:"+de.getExec()+" g:"+self.generic_name+" i:"+de.getIcon()+" n:"+self.name)
        self.comment = de.getComment().lower()
        self.exe = re.sub(u"%\w*", u"", de.getExec())
        # print(self.exe)
        self.icon_name = de.getIcon().lower()
        self.icon = self._find_icon(de.getIcon(), self.ico_size)

        self.icon_offset_y = (parent.item_size_y - self.ico_size) / 2
        self.icon_offset_x = parent.item_size_x - self.ico_size - self.icon_offset_y
        self.text_offset_x = self.icon_offset_y
        self.text_offset_y = self.icon_offset_y

        self.icon.set_size(parent.ico_size, parent.ico_size)
        # self.icon.set_size(self.ico_size,self.ico_size)

        self.text = Clutter.Text.new_full(font_dash, de.getName(), color_apps)
        self.text.set_width(self.icon_offset_x - self.icon_offset_y)
        self.text.set_ellipsize(Pango.EllipsizeMode.END)
        self.text.set_line_alignment(Pango.Alignment.LEFT)
        self.rect = ItemMenu()
        # self.rect = Clutter.Rectangle.new()
        self.rect.set_size(parent.item_size_x, parent.item_size_y)
        # self.rect.set_color(Clutter.Color.new(255,255,255,128))
        # self.rect.set_reactive(True)
        # self.rect.set_opacity(0)
        # self.rect.connect("enter-event", self.enter_handler, self)
        # self.rect.connect("leave-event", self.leave_handler, self)
        self.rect.connect("button-press-event", apps_entry.button_press_handler, self)

        # self.fade_in_transition = Clutter.PropertyTransition.new("opacity")
        # self.fade_in_transition.set_duration(100)
        # self.fade_in_transition.set_to(255)
        # self.fade_in_transition.connect("completed", self.fade_in_completed, self)
        # self.fade_out_transition = Clutter.PropertyTransition.new("opacity")
        # self.fade_out_transition.set_duration(2000)
        # self.fade_out_transition.set_to(0)
        # self.fade_out_transition.connect("completed", self.fade_out_completed, self)


        self.text_comment_offset_x = 2 * self.icon_offset_y
        self.text_comment_offset_y = self.icon_offset_y + self.text.get_height() + parent.margin
        self.text_comment = Clutter.Text.new_full(font_comment, self.comment, color_comment)
        self.text_comment.set_width(self.icon_offset_x - self.icon_offset_y - self.text_comment_offset_x)
        self.text_comment.set_height(self.ico_size - self.text_comment_offset_y + self.icon_offset_y)
        self.text_comment.set_line_wrap(True)

        self.parent.rect.add_child(self.rect)
        self.parent.rect.add_child(self.icon)
        self.parent.rect.add_child(self.text)
        self.parent.rect.add_child(self.text_comment)
        self.hide()
        pass

    # def enter_handler(self, widget, event, self1):
    # if self.rect.get_transition("fade_out"):
    #  self.rect.remove_transition("fade_out")
    # if not self.rect.get_transition("fade_in"):
    #  self.rect.add_transition("fade_in", self.fade_in_transition)
    # return True
    # def leave_handler(self, widget, event, self1):
    # if self.rect.get_transition("fade_in"):
    #  self.rect.remove_transition("fade_in")
    # if not self.rect.get_transition("fade_out"):
    #  self.rect.add_transition("fade_out", self.fade_out_transition)
    # return True

    def get_icon(self):
        return self._find_icon(self.icon_name, self.ico_size)

    def _find_icon(self, icon_name, size=48):
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
        # TODO: use a default icon.
        if not os.path.exists(icon_path):
            icon_path = '/usr/share/icons/gnome/48x48/apps/gnome-terminal.png'
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
        self.rect.set_position(x, y)
        self.icon.set_position(x + self.icon_offset_x, y + self.icon_offset_y)
        self.text.set_position(x + self.text_offset_x, y + self.text_offset_y)
        self.text_comment.set_position(x + self.text_comment_offset_x, y + self.text_comment_offset_y)
        pass

    # def fade_in_completed(self, transition, self1):
    # if self.rect.get_transition("fade_in"):
    #  self.rect.remove_transition("fade_in")
    # self.rect.set_opacity(255)
    # pass
    # def fade_out_completed(self, transition, self1):
    # if self.rect.get_transition("fade_out"):
    #  self.rect.remove_transition("fade_out")
    # self.rect.set_opacity(0)
    # pass

    def button_press_handler(widget, event, self):
        if event.button == Clutter.BUTTON_PRIMARY:
            self.call()
            self.rect.get_parent().hide()
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
                    ret.append(os.path.join(rdir, f))
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
        print("pattern:" + patern)
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
        print("pattern:" + patern)
        p = re.compile("^" + patern.lower() + "$")
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
        self.columns = int(floor(self.width / (self.size_x)))
        self.rows = int(floor(self.height / (self.size_y)))
        # self.left_margin = (self.width-(self.columns*self.size*1.5*1.3))/2.0
        self.top_margin = (self.height - (self.rows * self.size_y))

    # npages = int(floor(napps/self.columns*self.rows)+1.0)
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
        # intext.set_text(u"")
        # stage.show()
        # stage.set_key_focus(intext)
        pass

    @dbus.service.method("org.page.launcher", in_signature='', out_signature='')
    def quit(self):
        Clutter.main_quit()
        pass

    pass


class SubWindow(Clutter.Stage):
    def __init__(self, parent, x, y, width, height):
        super().__init__()
        self.parent = parent

        self.is_grab = False
        self._create_menu_window(x, y, width, height)

        # bind this stage to the window.
        ClutterGdk.set_stage_foreign(self, self.window)

        # Tell the window manager to put this window above the parent.
        self.window.set_transient_for(parent.window)

        self.set_user_resizable(False)
        self.set_title("sub-win")
        self.set_use_alpha(True)
        self.set_opacity(255)
        self.set_color(Clutter.Color.new(0, 0, 0, 0))
        self.set_scale(1.0, 1.0)
        self.set_accept_focus(True)

    def _create_menu_window(self, x, y, width, height):
        display = ClutterGdk.get_default_display()

        attr = Gdk.WindowAttr();
        attr.title = "sub-win"
        attr.width = width
        attr.height = height
        attr.x = x
        attr.y = y
        attr.event_mask = 0
        attr.window_type = Gdk.WindowType.TEMP
        attr.visual = display.get_default_screen().get_rgba_visual()
        attr.override_redirect = True
        attr.type_hint = Gdk.WindowTypeHint.MENU

        # parent is set here if you want create a nested window.
        self.window = Gdk.Window(None, attr,
                                 Gdk.WindowAttributesType.TITLE
                                 | Gdk.WindowAttributesType.VISUAL
                                 | Gdk.WindowAttributesType.X
                                 | Gdk.WindowAttributesType.Y
                                 | Gdk.WindowAttributesType.NOREDIR
                                 | Gdk.WindowAttributesType.TYPE_HINT)

    def show(self):
        super().show()
        self.window.show()

    def hide(self):
        self.window.hide()
        super().hide()


class ItemMenu(Clutter.Rectangle):
    def __init__(self):
        super().__init__()
        self.set_color(Clutter.Color.new(0, 0, 0, 0))
        self.set_border_color(Clutter.Color.new(100, 100, 100, 255))
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

    def event_menu_enter_handler(widget, event, self, item):
        print('ENTER')
        if item.get_transition("fade_out"):
            item.remove_transition("fade_out")
        if not item.get_transition("fade_in"):
            item.add_transition("fade_in", item.fade_in_transition)
        return True

    def event_menu_leave_handler(widget, event, self, item):
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
        root_width = display.get_default_screen().get_root_window().get_width()
        super().__init__(parent, offset_x, 0, size_x, self.root_height)

    def show(self, event_time):
        print("Slide.show: Dash show")
        parent_window = ClutterGdk.get_stage_window(self.parent)
        self.window.set_transient_for(parent_window)
        self.reset()
        super().show()
        self.window.focus(event_time)
        self.set_move(self.pos_x_show)


class DashSlide(Slide):
    def __init__(self, parent, offset_x, ico_size, slide_size):
        self.slide_size = slide_size
        super().__init__(parent, offset_x, self.slide_size)

        self.margin_x = 10
        self.text_size = 64
        self.margin = 2
        self.y_offset = self.text_size + 2 * self.margin
        self.item_size_x = self.slide_size
        self.item_size_y = 64
        self.ico_size = ico_size

        self.rect = Clutter.Actor()
        if parent.side == 'left':
            self.pos_x_hide = -self.slide_size
            self.pos_x_show = 0
        else:
            self.pos_x_hide = self.slide_size
            self.pos_x_show = 0

        self.rect.set_x(self.pos_x_hide)
        self.rect.set_y(0)
        self.rect.set_size(self.slide_size, self.root_height)
        self.rect.set_background_color(Clutter.Color.new(32, 32, 32, 240))

        # Enable animation
        self.rect.save_easing_state();
        self.rect.set_easing_mode(Clutter.AnimationMode.EASE_IN_OUT_CUBIC);
        self.rect.set_easing_duration(1000);

        self.add_child(self.rect)

        self.pos = 200

        # init slide
        self.back_text = ItemMenu()
        self.back_text.set_x(0)
        self.back_text.set_y(self.root_height - self.text_size - self.margin)
        self.back_text.set_size(self.slide_size, self.text_size)

        self.intext = Clutter.Text.new_full(font_entry, u"", color_entry)
        self.apps = apps_handler(self)
        self.intext.set_editable(True)
        self.intext.set_selectable(True)
        self.intext.set_activatable(True)

        self.notext = Clutter.Text.new_full(font_entry, u"Enter Text Here ...",
                                            Clutter.Color.new(255, 255, 255, 128))
        self.text_offset = (self.text_size - self.notext.get_height()) / 2

        self.notext.set_x(self.margin_x)
        self.notext.set_y(self.root_height - self.text_size + self.text_offset - self.margin)
        self.notext.set_width(self.slide_size - self.margin_x * 2)

        self.intext.set_x(self.margin_x)
        self.intext.set_y(self.root_height - self.text_size + self.text_offset - self.margin)
        self.intext.set_width(self.slide_size - self.margin_x * 2)

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

    # self.show()

    def reset(self):
        self.rect.restore_easing_state();
        self.rect.set_x(self.pos_x_hide)
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

    def leave_notify(self, widget, event):
        print("LEAVE")

    def enter_notify(self, widget, event):
        print("ENTER")

    def key_focus_in(self, event, data=None):
        print("focus_in #DashView")

    def key_focus_out(self, event, date=None):
        print("focus_out #DashView")

    def button_press_handler(self, widget, event):
        print("button press")
        print(event)
        widget = self.get_actor_at_pos(Clutter.PickMode.ALL, event.x, event.y)
        if widget == self.intext:
            self.set_key_focus(self.intext)
            return True
        elif not widget:
            self.hide()
            return True
        return False

    def handle_text_changed(self, data=None):
        self.apps.hide_all()
        text = self.intext.get_text()
        # print("XXXX"+text)
        if not text:
            text = u""
        self.apps_list = self.apps.filter_apps(text)
        layout = launcher_layout(self, len(self.apps_list))
        if text == u"":
            self.notext.show()
        else:
            self.notext.hide()
        self.current_actor = list()

        for i in range(0, layout.columns * layout.rows):
            if i >= len(self.apps_list):
                break
            c = i - floor(i / layout.columns) * layout.columns
            l = floor(i / layout.columns)
            a = self.apps_list[i]
            a.set_position(c * layout.size_x, (layout.rows - l - 1) * layout.size_y + self.margin + layout.top_margin)
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
                       | Gdk.EventMask.KEY_RELEASE_MASK,
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

    def event_menu_button_press(widget, event, self, menu):
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

    def show_menu(self, x, y, menulist, event_time, halign_right):
        print('PanelMenu.show_menu: SHOW MENU')
        self.set_x(0)
        self.set_y(0)
        # self.rect.remove_all_children()
        self.destroy_all_children()
        # self.remove_all()

        self.rect = Clutter.Actor()
        self.rect.set_x(0)
        self.rect.set_y(0)
        self.rect.set_background_color(Clutter.Color.new(32, 32, 32, 200))

        self.add_child(self.rect)
        self.global_width = 0

        tmp_height = 0
        for menu in menulist:
            ##print(menu['text'])
            text = Clutter.Text.new_full(font_menu_entry, menu['text'], Clutter.Color.new(255, 255, 255, 255))
            # menu['txt'] = text
            # text.set_background_color(Clutter.Color.new(0,0,0,0))

            text.set_x(self.item_margin_x)
            text.set_y(tmp_height + self.item_height / 4)

            self.rect.add_child(text)

            self.global_width = max(self.global_width, text.get_width() + 2 * self.item_margin_x)
            tmp_height += self.item_height + self.item_margin_y
        self.global_height = tmp_height - +self.item_margin_y

        tmp_height = 0
        for menu in menulist:
            back = ItemMenu()
            back.connect("button-press-event", PanelMenu.event_menu_button_press, self, menu)

            back.set_x(0)
            back.set_y(tmp_height)
            back.set_size(self.global_width, self.item_height)

            tmp_height += self.item_height + self.item_margin_y
            self.rect.add_child(back)

        # self.window.set_size()
        # dir(self.window)
        # print(self.global_width)
        # print(self.global_height)

        # if self.root_height < self.global_height+y:
        #	y=self.root_height-self.global_height

        self.window.resize(self.global_width, self.global_height)
        if halign_right:
            self.window.move(x, y)
        else:
            self.window.move(x - self.global_width, y)

        self.set_size(self.global_width, self.global_height)
        self.rect.set_size(self.global_width, self.global_height)

        super().show()
        # Focus can only apply on visible window /!\
        self.window.focus(event_time)
        pass

    def hide_menu(self):
        print('PanelMenu.hide_menu')
        super().hide()
        self.remove_all()


class PanelApp():
    def __init__(self, xid, name, group, pid):
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

    def activate(self, event_time):
        print('ACTIVATE: ' + str(self.xid))
        w = Wnck.Window.get(self.xid)
        if w != None:
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
        self.sub_offset = (self.icon_size_x - self.sub_icon_size) / 2
        self.icon = icon
        self.icon.set_size(self.sub_icon_size, self.sub_icon_size)

        self.icon_back = ItemMenu()
        self.icon_back.set_size(self.icon_size_x, self.icon_size_y)
        self.icon_back.set_reactive(True)
        self.icon_back.connect("button-press-event", self.button_press_handler)
        self.icon_back.connect("button-release-event", self.button_release_handler)

        self.icon.set_position(self.sub_offset, self.sub_offset)
        self.icon_back.set_position(0, 0)

        self.add_child(self.icon_back)
        self.add_child(self.icon)
        # Enable animation
        self.save_easing_state();
        self.set_easing_mode(Clutter.AnimationMode.EASE_IN_OUT_CUBIC);
        self.set_easing_duration(100);

    def get_size_y(self):
        return self.icon_size_y

    def button_release_handler(self, widget, event):
        print("button_release_handler\n")
        return True

        # def motion_handler(self, widget, event):
        #	print("motion_handler\n")
        #	return True


class PanelClock(Clutter.Group):
    def __init__(self, panel, ico_size):
        super().__init__()
        self.panel = panel
        self.icon_size_x = ico_size
        self.icon_size_y = ico_size / 2
        self.sub_offset = 2
        # self.sub_icon_size = ico_size*margin
        # self.sub_offset = (self.icon_size-self.sub_icon_size)/2
        self.text = Clutter.Text.new_full(font_clock, u"",
                                          color_clock)
        # self.text.set_line_alignment(Pango.Alignment.CENTER)
        # self.text.set_size(ico_size, ico_size/2)
        # self.text.set_width(ico_size)

        self.icon_back = ItemMenu()
        self.icon_back.set_position(0, 0)

        self.add_child(self.icon_back)
        self.add_child(self.text)

    def get_size_y(self):
        return self.icon_size_y

    def set_position(self, x, y):
        super().set_position(x, y)
        now = datetime.datetime.now()
        self.text.set_text(now.strftime('%H:%M'))

        if self.text.get_width() < self.icon_size_x:
            self.sub_offset_x = (self.icon_size_x - self.text.get_width()) / 2
            self.sub_offset_y = (self.icon_size_y - self.text.get_height()) / 2
        # self.text.set_position(self.sub_offset_x,self.sub_offset_y)
        else:
            # if self.text.get_height() > self.icon_size_x:
            #	self.text.set_text(now.strftime('%H:%M'))
            self.icon_size_y = ceil(self.text.get_width() / 32) * 32
            self.text.set_rotation(Clutter.RotateAxis.Z_AXIS, -90, 0, 0, 0)
            self.text.set_translation(0, self.text.get_width(), 0)
            self.sub_offset_y = (self.icon_size_y - self.text.get_width()) / 2
            self.sub_offset_x = (self.icon_size_x - self.text.get_height()) / 2
            print([self.text.get_width(), self.text.get_height(), self.sub_offset_x, self.sub_offset_y])
        self.text.set_position(self.sub_offset_x, self.sub_offset_y)
        self.icon_back.set_size(self.icon_size_x, self.icon_size_y)


class PanelApps(PanelIcon):
    def __init__(self, panel, ico_size):
        super().__init__(panel, Clutter.Texture.new_from_file("./data/app.svg"), ico_size, 3 * ico_size / 4)

    def button_press_handler(self, widget, event):
        self.panel.sub_dash(event.time)
        return True


class PanelShutdown(PanelIcon):
    def __init__(self, panel, ico_size):
        super().__init__(panel, Clutter.Texture.new_from_file("./data/shutdown.svg"), ico_size, 3 * ico_size / 4)

    def button_press_handler(self, widget, event):
        menu_list = [
            {'text': "Shutdown", 'obj': Shutdown()},
            {'text': "Log Out", 'obj': Logout()},
            {'text': "Lock", 'obj': Lock()},
            {'text': "Sleep", 'obj': Sleep()},
            {'text': "Reboot", 'obj': Reboot()},
        ]

        self.panel.sub_menu(self.get_y(), menu_list, event.time)
        return True


class PanelGroupApp(PanelIcon):
    def __init__(self, panel, name, icon, ico_size, new_process, locked):
        self.app_list = {}
        self.name = name
        self.locked = locked
        self.icon_1px = None
        self.cb_new_process = new_process
        self.enabled = True
        # self.icon_1px.set_depth(100)


        super().__init__(panel, icon, ico_size, 3 * ico_size / 4)

        self.arrows = [Clutter.Texture.new_from_file("./data/launcher_arrow_ltr_19.svg"),
                       Clutter.Texture.new_from_file("./data/launcher_arrow_ltr_19.svg"),
                       Clutter.Texture.new_from_file("./data/launcher_arrow_ltr_19.svg")]

        for arrow in self.arrows:
            self.insert_child_above(arrow, self.icon_back)
            arrow.set_size(ico_size / 5, ico_size / 5)
            arrow.hide()
        self.arrow_size = self.arrows[0].get_height()

    # self.icon_text = Clutter.Text.new_full(font_menu_entry, "", Clutter.Color.new(255,255,255,255))
    # self.add_child(self.icon_text)

    def update_arrows(self):
        ind = 0
        margin = -1 * self.arrow_size / 3
        nb_app = min(len(self.app_list), 3)
        for arrow in self.arrows:
            if ind < nb_app:
                arrow.set_y(self.icon_size_x / 2 - (nb_app * arrow.get_height() + (nb_app - 1) * margin) / 2 + ind * (
                    arrow.get_height() + margin))
                arrow.show()
            else:
                print
                arrow.hide()
            ind = ind + 1
            # print(ind)
            # print("nb_app:"+str(nb_app))

    def set_background(self, px1_ico):
        self.icon_1px = px1_ico
        self.icon_1px.set_size(self.icon_size_x, self.icon_size_x)
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
        for k, iapp in self.app_list.items():
            iapp.kill()

    def set_position(self, x, y):
        super().set_position(x, y)
        self.update_arrows()
        if self.icon_1px:
            self.icon_1px.set_position(0, 0)
            # self.icon_text.set_text(str(len(self.app_list)))
            # self.icon_text.set_position(x+5,y+5)

    def unregister(self):
        # print("UNREG")
        self.hide_all()
        self.destroy_all_children()
        self.get_parent().remove_child(self)

    # print(self.app_list)



    def __del__(self):
        print("DESTROY")

    def add(self, iapp):
        print("Adding " + str(iapp.get_xid()) + " to " + self.name)
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

    def set_enabled(self, value):
        self.enabled = value

    def button_press_handler(self, widget, event):
        print('PanelGroupApp.button_press_handler', event)
        if event.button == 1:
            self.sliding_icons = True
            self.sliding_start = self.get_x()

    def button_release_handler(self, widget, event):
        print('PanelGroupApp.button_release_handler', event)
        if not self.enabled:
            return False

        if event.button == 1:
            if len(self.app_list) > 1:
                menu_list = []
                for k, iapp in self.app_list.items():
                    menu = {}
                    menu['text'] = iapp.get_name()
                    menu['obj'] = iapp
                    menu_list.append(menu)
                self.panel.sub_menu(self.get_y(), menu_list, event.time)
                return False
            elif len(self.app_list) == 1:
                self.panel.sub_reset(event.time)
                for k, iapp in self.app_list.items():
                    iapp.activate(event.time)
                    return False
            else:
                self.process_new(event.time)
                return False
        elif event.button == 3:
            menu_list = []
            if self.locked:
                menu_list.append({'text': "Unlock", 'cb': self.unlock})
            else:
                menu_list.append({'text': "Lock", 'cb': self.lock})

            menu_list.append({'text': "New Instance", 'cb': self.process_new})
            if len(self.app_list) >= 1:
                menu_list.append({'text': "Terminate", 'cb': self.process_close})

            self.panel.sub_menu(self.get_y(), menu_list, event.time)
            return True
        return False

    def __str__(self):
        # print(self.icon)
        tmp = "- "
        tmp += self.name
        tmp += ":\n"
        for k, iapp in self.app_list.items():
            tmp += "\t"
            tmp += iapp.get_name()
            tmp += "\n"
        return tmp


class PanelTray(Clutter.Group):
    def __init__(self, panel, ico_size):
        super().__init__()
        self.panel = panel
        self.icon_size_x = ico_size
        self.icon_size_y = ico_size / 2

        self.sz_x_ico = 24
        self.sz_y_ico = 24

        self.max_col = int(self.icon_size_x / 32)
        # print(self.max_col)
        self.margin_ico_y = 5
        self.sub_offset = 2
        self.window = panel.window
        self.dock_list = {}
        # self.sub_icon_size = ico_size*margin
        # self.sub_offset = (self.icon_size-self.sub_icon_size)/2
        # self.text = Clutter.Text.new_full(font_clock, u"12:30",color_clock)
        # self.text.set_line_alignment(Pango.Alignment.CENTER)
        # self.text.set_size(ico_size, ico_size/2)

        self.icon_back = ItemMenu()
        self.icon_back.set_size(self.icon_size_x, self.icon_size_y)
        self.icon_back.set_position(0, 0)
        # self.icon_back.set_color(Clutter.Color.new(1,0,0,255))

        self.add_child(self.icon_back)
        # self.add_child(self.text)

        display = ClutterGdk.get_default_display()
        root_height = display.get_default_screen().get_root_window().get_height()
        root_width = display.get_default_screen().get_root_window().get_width()

        screen = Wnck.Screen.get_default()

        # display = Gdk.screen_get_display (screen)
        selection_atom_name = "_NET_SYSTEM_TRAY_S%d" % screen.get_number()
        selection_atom = Gdk.atom_intern(selection_atom_name, False);

        self.atom_opcode = GdkX11.x11_atom_to_xatom_for_display(display,
                                                                Gdk.atom_intern("_NET_SYSTEM_TRAY_OPCODE", False))
        self.atom_message_data = GdkX11.x11_atom_to_xatom_for_display(display,
                                                                      Gdk.atom_intern("_NET_SYSTEM_TRAY_MESSAGE_DATA",
                                                                                      False))
        timestamp = GdkX11.x11_get_server_time(self.window)
        # toto = Gdk.selection_owner_get_for_display(display, selection_atom)
        # print(toto)
        res = Gdk.selection_owner_set_for_display(display,
                                                  self.window,
                                                  selection_atom,
                                                  timestamp,
                                                  True)
        # print(selection_atom_name )
        if not res:
            print("Unable to set systray !!")
        else:
            res_get = Gdk.selection_owner_get_for_display(display, selection_atom)
            if res_get != self.window:
                print("Unable to get systray !!")
            else:
                print("Owner is ok")

            PageLauncherHook.set_system_tray_orientation(self.window, True)
            PageLauncherHook.set_system_tray_visual(self.window, display)
            PageLauncherHook.set_system_tray_filter(self.window, display, self)
            # self.window.add_filter(self.toto)

    def get_size_y(self):
        sz = int(ceil(len(self.dock_list) / self.max_col) * (self.sz_y_ico + self.margin_ico_y) + self.margin_ico_y)
        return sz

    def update_pos_x(self, nb_trays):
        self.margin_ico_x = (self.icon_size_x - nb_trays * self.sz_x_ico) / (nb_trays + 1)
        pos_x = [self.margin_ico_x + self.panel.window.get_position()[0]]
        tmp_x = pos_x[0]

        for ind in range(nb_trays):
            tmp_x = tmp_x + self.margin_ico_x + self.sz_x_ico
            pos_x.append(tmp_x)

        # print(pos_x)
        return pos_x

    def set_position(self, x, y):
        super().set_position(x, y)
        self.icon_back.set_height(self.get_size_y())

        ind = 0
        tmp_y = self.margin_ico_y
        pos_x = self.update_pos_x(min(len(self.dock_list) - ind, self.max_col))
        for win_id, win_inter in self.dock_list.items():

            PageLauncherHook.move_tray(self.window,
                                       ClutterGdk.get_default_display(),
                                       win_id,
                                       win_inter,
                                       int(pos_x[ind % self.max_col] + self.sub_offset),
                                       int(tmp_y + y),
                                       self.sz_x_ico,
                                       self.sz_y_ico)
            if ind % self.max_col == self.max_col - 1:
                tmp_y = tmp_y + self.sz_y_ico + self.margin_ico_y
                pos_x = self.update_pos_x(min(len(self.dock_list) - ind - 1, self.max_col))
            ind += 1

    def undock_request(self, socket_id, window):
        print('undock request: ' + str(socket_id))
        # print(socket_id)
        # print(window)
        if socket_id in self.dock_list:
            win_inter = self.dock_list[socket_id]
            PageLauncherHook.undock_tray(self.window, ClutterGdk.get_default_display(), socket_id, win_inter)
            del self.dock_list[socket_id]
        else:
            print('unknown id:' + str(socket_id))

    def dock_request(self, socket_id, window):
        print('dock request')
        # print(socket_id)
        # print(window)
        win_inter = PageLauncherHook.dock_tray(self.window, ClutterGdk.get_default_display(), socket_id)
        self.dock_list[socket_id] = win_inter

    def message_begin(self, socket_id, window):
        print('message begin')

    def message_cancel(self, socket_id, window):
        print('message cancel')

    def tray_filter(self, a):
        # print(a.type)
        return Gdk.FilterReturn.CONTINUE

        # print(b)


class PanelView(Clutter.Stage):
    def __init__(self):
        super().__init__()

        # Manualy create the window to setup properties before mapping the window
        self.ico_size = 64
        self.margin = 2
        self.side = 'left'

        self.config_file = os.path.join(os.environ['HOME'], '.page-launcher.cfg')
        self.config_read()
        self.ico_size = max(32, self.ico_size)
        self.panel_width = self.ico_size + 2 * self.margin

        self._create_panel_window()

        display = ClutterGdk.get_default_display()
        root_height = display.get_default_screen().get_root_window().get_height()
        root_width = display.get_default_screen().get_root_window().get_width()

        screen = Wnck.Screen.get_default()
        screen.connect("active-window-changed", self.on_active_window_change)
        # tricks to create the window
        ClutterGdk.set_stage_foreign(self, self.window)
        self.window.set_type_hint(Gdk.WindowTypeHint.DOCK)
        self.window.stick()
        if self.side == 'left':
            PageLauncherHook.set_strut(self.window, [self.panel_width, 0, 0, 0])
        else:
            PageLauncherHook.set_strut(self.window, [0, self.panel_width, 0, 0])

        self.set_size(self.panel_width, root_height)
        self.set_user_resizable(False)
        self.set_title("page-panel")
        self.set_use_alpha(True)
        self.set_opacity(0)
        self.set_color(Clutter.Color.new(0, 0, 0, 0))
        self.set_scale(1.0, 1.0)
        # print(self.window.get_position())

        self.sliding_icons = False
        self.sliding_start = 0
        self.sliding_current = 0;
        self.connect('button-press-event', self.button_press_handler)
        self.connect('button-release-event', self.button_release_handler)
        self.connect("motion-event", self.motion_handler)
        self.connect("leave-event", self.leave_handler)
        # create dash view
        # self.dash = DashView(self)
        self.panel_menu = PanelMenu(self)

        if self.side == 'left':
            self.dash_slide = DashSlide(self, self.panel_width, self.panel_width, 300)
        else:
            self.dash_slide = DashSlide(self, root_width - self.panel_width - 300, self.panel_width, 300)

        # Dictionnary of apps
        self.dict_apps = {}
        self.list_group_apps = []
        self.list_sys_apps = []

        self.list_sys_apps.append(PanelClock(self, self.ico_size))
        self.list_sys_apps.append(PanelTray(self, self.ico_size))
        self.list_sys_apps.append(PanelApps(self, self.ico_size))
        self.list_sys_apps.append(PanelShutdown(self, self.ico_size))

        self.update_cnt = 0
        self.pos_offset = 0

        self.rect = Clutter.Actor()
        self.rect.set_x(0)
        self.rect.set_y(0)
        self.rect.set_size(self.panel_width, root_height)
        self.rect.set_background_color(Clutter.Color.new(32, 32, 32, 255))
        # self.rect.set_background_color(Clutter.Color.new(255,255,0,0))
        self.add_child(self.rect)

        for app in self.list_sys_apps:
            self.rect.add_child(app)

        self.wnck_screen = Wnck.Screen.get_default()

        self.config_init_apps()
        self.update_current_apps()

        GObject.timeout_add(1000, self.refresh_timer, self)

    def config_save(self):
        config = configparser.ConfigParser()
        config.add_section('Launcher')
        apps_locked_list = []
        for group in self.list_group_apps:
            if group.is_locked():
                apps_locked_list.append(group.get_name())

        config.set('Launcher', 'apps', ','.join(apps_locked_list))
        config.set('Launcher', 'side', self.side)
        config.set('Launcher', 'width', str(self.ico_size))

        # Writing our configuration file to 'example.cfg'
        with open(self.config_file, 'w') as configfile:
            config.write(configfile)

    def config_read(self):
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file)
        if self.config.has_section('Launcher'):
            if self.config.has_option('Launcher', 'width'):
                self.side = self.config.get('Launcher', 'side')
            if self.config.has_option('Launcher', 'width'):
                self.ico_size = int(self.config.get('Launcher', 'width'))

    def config_init_apps(self):
        if self.config.has_section('Launcher'):
            apps_locked_list = self.config.get('Launcher', 'apps').split(',')
            for group_name in apps_locked_list:
                self.create_app(group_name, None, True)

    def refresh_timer(self, *arg):
        self.update_current_apps()
        return True

    def find_app(self, name):
        ret = self.dash_slide.apps.match_apps(name)
        if ret:
            # print(ret[0].name)
            return ret[0]
        else:
            return None

    def create_app(self, group_name, pix_icon, locked):
        # print(app.get_icon().get_width())
        # Get icon path
        appdict = self.find_app(group_name);
        if appdict:
            cb_new_process = appdict.call
            ico = appdict.get_icon()
        else:
            ico = None
            cb_new_process = None
        # if ico==None:
        #	ico = self.find_ico(app.get_application().get_name())
        if ico == None and pix_icon:
            pix = pix_icon.scale_simple(self.ico_size, self.ico_size, GdkPixbuf.InterpType.HYPER)
            ico = Clutter.Texture.new()
            data = pix.get_pixels()
            width = pix.get_width()
            height = pix.get_height()
            if pix.get_has_alpha():
                bpp = 4
            else:
                bpp = 3
            rowstride = pix.get_rowstride()
            ico.set_from_rgb_data(data, pix.get_has_alpha(), width, height, rowstride, bpp, 0);
        # ico.set_width(48,48)
        # ico_data=  pix.get_pixels_array()
        assert (ico != None)

        # print('Create new group:' + str(group_name))
        grp = PanelGroupApp(self, group_name, ico, self.ico_size, cb_new_process, locked)

        # print(sys.getrefcount(grp))
        self.rect.add_child(grp)
        # print(sys.getrefcount(grp))
        self.list_group_apps.append(grp)
        # print(sys.getrefcount(grp))

        return grp

    def update_current_apps(self):
        self.wnck_screen.force_update()
        apps = self.wnck_screen.get_windows_stacked()
        # print(apps)
        # for app in apps:
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
                # print(app.get_xid())
                xid = app.get_xid()
                name = app.get_name()
                group_name = app.get_class_group_name()
                # dir(app)
                # print('---')
                # print(app.get_icon_name())
                # print(app.get_application().get_name())
                # print(app.get_application().get_pid())
                # print(app.get_class_group_name())
                # print(app.get_class_instance_name())
                # print(group_name)

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
                        px_pix = app.get_icon().scale_simple(1, 1, GdkPixbuf.InterpType.HYPER)
                        if px_pix.get_has_alpha():
                            px_bpp = 4
                        else:
                            px_bpp = 3
                        px_ico.set_from_rgb_data(px_pix.get_pixels(), px_pix.get_has_alpha(), px_pix.get_width(),
                                                 px_pix.get_height(), px_pix.get_rowstride(), px_bpp, 0);
                        grp.set_background(px_ico)

                    # Append app to dict
                    # print(sys.getrefcount(grp))
                    iapp = PanelApp(xid=xid, name=name, group=grp, pid=app.get_application().get_pid())
                    # print(sys.getrefcount(grp))
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
            # print('Deleting group II :' + str(grp.get_name()))
            # print(sys.getrefcount(grp))
            # for r in gc.get_referrers(grp):
            #	pprint.pprint(r)
            self.list_group_apps.remove(grp)
            grp.unregister()
        # print(sys.getrefcount(grp))
        # for r in gc.get_referrers(grp):
        #	pprint.pprint(r)

        self.update_sys_icons_pos()
        if not self.sliding_icons:
            self.update_apps_icons_pos(self.sliding_current)


            # print('=====')
            # for grp in self.list_group_apps:
            #	print(grp)

    def update_apps_icons_pos(self, offset):
        # Update icon position
        pos_y = self.pos_offset

        for grp in self.list_group_apps:
            pos_tmp = pos_y + self.margin + offset
            if pos_tmp < 0:
                ico_offset = max(-pos_tmp, 0)
                ico_opacity = max(0, min(255 * ((pos_tmp + self.ico_size) / self.ico_size), 255))
            elif pos_tmp > self.ico_sys_limit:
                ico_offset = min(self.ico_sys_limit - pos_tmp, 0)
                ico_opacity = max(0, min(255 * ((self.ico_size - (pos_tmp - self.ico_sys_limit)) / self.ico_size), 255))
            else:
                ico_offset = 0
                ico_opacity = 255
            grp.set_opacity(ico_opacity)
            grp.set_position(self.margin, pos_tmp + ico_offset)
            if ico_opacity:
                grp.show()
            else:
                grp.hide()
            pos_y += self.ico_size + self.margin

        # print(self.sliding_current)
        # print(pos_y)
        if self.ico_sys_limit > pos_y:
            self.sliding_next = 0
        else:
            self.sliding_next = offset

    def update_sys_icons_pos(self):
        # Update icon position
        pos_y = self.get_height() - self.margin
        for grp in self.list_sys_apps:
            pos_y -= grp.get_size_y() + self.margin
            grp.set_position(self.margin, pos_y + self.margin)
        self.ico_sys_limit = pos_y - self.ico_size

    def motion_handler(self, widget, event):
        if self.sliding_icons:
            # Update icons position according to mouse pos
            self.update_apps_icons_pos(self.sliding_current + event.y - self.sliding_start)
            # Disable press action of apps
            for grp in self.list_group_apps:
                grp.set_enabled(False)
        pass

    def leave_handler(self, widget, event):
        print('PanelView.leave_handler', event)
        # self.button_release_handler(widget,event)
        pass

    def button_release_handler(self, widget, event):
        self.sliding_current = min(self.sliding_next, 0)
        self.sliding_icons = False
        self.update_apps_icons_pos(self.sliding_current)
        # Enable press action on app
        for grp in self.list_group_apps:
            grp.set_enabled(True)
        print('PanelView.button_release_handler', event)
        return False
        pass

    def button_press_handler(self, widget, event):
        print('PanelView.button_press_handler', event)

        if event.button == 1:
            self.sliding_icons = True
            self.sliding_start = event.y
            # print ('pouet')
            # self.panel_menu.hide_menu()
            # self.dash.show(event.time)
            # self.dash.window.focus(event.time)
            # self.dash_slide.show(event.time)
            # self.panel_menu.hide_menu()
            pass

        elif event.button == 3:
            self.sub_menu(event.y, [{'text': 'Quit launcher', 'cb': self.main_quit}], event.time)
            return True
            pass

        # hide all sub windows on panel click if
        # no action happenned
        self.sub_reset()

        return False

    def main_quit(self, time):
        Clutter.main_quit()

    def run(self):
        self.show()
        Clutter.main()

    def _create_panel_window(self):
        display = ClutterGdk.get_default_display()
        self.root_height = display.get_default_screen().get_root_window().get_height()
        self.root_width = display.get_default_screen().get_root_window().get_width()

        attr = Gdk.WindowAttr();
        attr.title = "page-panel"
        attr.width = self.panel_width
        attr.height = self.root_height
        attr.x = 0  # root_width-attr.width
        attr.y = 0
        # print(attr.x)
        attr.event_mask = 0
        attr.window_type = Gdk.WindowType.TOPLEVEL
        attr.visual = display.get_default_screen().get_rgba_visual()
        self.window = Gdk.Window(None, attr,
                                 Gdk.WindowAttributesType.TITLE
                                 | Gdk.WindowAttributesType.VISUAL
                                 | Gdk.WindowAttributesType.X
                                 | Gdk.WindowAttributesType.Y)

    def key_focus_in(self, event, data=None):
        print("focus_in")

    def key_focus_out(self, event, date=None):
        print("focus_out")

    def xxx_activate(self, event, date=None):
        print("activate #PanelView")

    def xxx_deactivate(self, event, date=None):
        print("deactivate #PanelView")

    def on_active_window_change(self, screen, window):
        if (self.window.get_xid() != screen.get_active_window().get_xid()):
            # TODO HIDE ALL
            self.dash.hide()

    def on_active_window_change(self, screen, window):
        if (self.window.get_xid() != screen.get_active_window().get_xid()):
            # TODO HIDE ALL SUB WINDOWS
            self.sub_reset()

    def sub_reset(self, event_time=0):
        self.dash_slide.hide()
        self.panel_menu.hide_menu()

    def sub_dash(self, event_time):
        self.dash_slide.show(event_time)
        self.panel_menu.hide_menu()

    def sub_menu(self, offset_y, menu_list, event_time):
        self.raise_window(event_time)
        self.dash_slide.hide()
        if self.side == 'left':
            self.panel_menu.show_menu(self.panel_width, offset_y, menu_list, event_time, True)
        else:
            self.panel_menu.show_menu(self.root_width - self.panel_width, offset_y, menu_list, event_time, False)
        self.panel_menu.window.focus(event_time)

    def raise_window(self, time):
        self.window.show()
        w = Wnck.Window.get(self.window.get_xid())
        if w != None:
            w.activate(time)

    def show(self):
        super().show()
        self.window.show()


#####
####

def dbus_mate(bus):
    bus_object = bus.get_object("org.mate.SessionManager", "/org/mate/SessionManager")
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
    seat_path = os.environ.get('XDG_SEAT_PATH')
    bus_object = bus.get_object("org.freedesktop.DisplayManager", seat_path)
    iface = dbus.Interface(bus_object, 'org.freedesktop.DisplayManager.Seat')
    return iface


class Sleep():
    def activate(self, event_time):
        bus = dbus.SystemBus()
        try:
            i = dbus_login1(bus)
            if i.CanHybridSleep() == "yes":
                print("sleep")
                i.HybridSleep(0)
            elif i.CanHibernate() == "yes":
                print("hibernate")
                i.Hibernate(0)
            elif i.CanSuspend() == "yes":
                print("suspend")
                i.Suspend(0)
            else:
                raise Exception("'org.freedesktop.login1' somehow doesn't work.")
        except:
            i = dbus_upower(bus)
            if i.HibernateAllowed():
                print("hibernate")
                i.Hibernate(0)
            elif i.SuspendAllowed():
                print("suspend")
                i.Suspend(0)
            else:
                print("Error: could not sleep, sorry.")


class Shutdown():
    def activate(self, event_time):
        bus = dbus.SystemBus()
        try:
            i = dbus_login1(bus)
            if i.CanPowerOff() == "yes":
                print("power off")
                i.PowerOff(0)
            else:
                raise Exception("'org.freedesktop.login1' somehow doesn't work.")
        except:
            i = dbus_consolekit(bus)
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
            if i.CanReboot() == "no":
                print("reboot")
                i.Reboot(0)
            else:
                raise Exception("'org.freedesktop.login1' somehow doesn't work.")
        except:
            i = dbus_consolekit(bus)
            if i.CanRestart():
                print("reboot")
                i.Restart(0)
            else:
                raise Exception("'org.freedesktop.Consolekit' somehow doesn't work.")


class Logout():
    # dbus-send --session --type=method_call --print-reply --dest=org.gnome.SessionManager /org/gnome/SessionManager org.gnome.SessionManager.Logout uint32:1
    def activate(self, event_time):
        bus = dbus.SessionBus()
        try:
            i = dbus_mate(bus)
            print("logout")
            i.Logout(1)
        except:
            raise Exception("logout somehow doesn't work.")


class Lock():
    def activate(self, event_time):
        bus = dbus.SystemBus()
        try:
            i = dbus_lightdm(bus)
            print("lock")
            i.SwitchToGreeter(0)
        except:
            raise Exception("lock somehow doesn't work.")


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

    panel = PanelView()
    panel.run()
