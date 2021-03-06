# -*- coding: utf-8 -*-
# Python 3 code
#from distutils.core import setup, Extension
import subprocess

from numpy.distutils.core import setup, Extension

def pkgconfig(*packages, **kw):
 flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
 for token in subprocess.check_output(["pkg-config", "--libs", "--cflags"] + list(packages)).decode('UTF-8').split():
  if token[:2] in flag_map:
   kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
  else: # throw others to extra_link_args
   kw.setdefault('extra_link_args', []).append(token)
 try:
  # python 2.x
  for k, v in kw.iteritems(): # remove duplicated
   kw[k] = list(set(v))
 except AttributeError:
  # python 3.x
  for k, v in kw.items(): # remove duplicated
   kw[k] = list(set(kw[k]))
 return kw

params = pkgconfig('pygobject-3.0', 'clutter-glx-1.0', 'gdk-x11-3.0')
params['extra_compile_args'] = ['-std=c99 -ggdb']
module1 = Extension('PageLauncherHook', sources = ['page_launcher_hook.c'], **params)

setup (name = 'PageLauncherHook',
		version = '1.0',
		description = 'Low level fonction not avalaible in python',
		ext_modules = [module1])

