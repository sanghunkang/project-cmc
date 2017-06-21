#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import 3rd-party packages
import os, subprocess, sys

DIR_DATA = "/usr/local/dev/project-cucm/data_light/"

print(sys.platform)



if "linux" in sys.platform:
	# convert /usr/local/dev/project-cucm/data_light/I0000001 /usr/local/dev/project-cucm/data_light/I0000001.BMP
	cmd = "convert " + DIR_DATA + "I0000001 " + DIR_DATA + "I0000001.BMP"
	subprocess.call([cmd], shell=True)
	# subprocess.call(["convert " + DIR_DATA + "I0000002 " + DIR_DATA + "I0000002.BMP"], shell=True)
