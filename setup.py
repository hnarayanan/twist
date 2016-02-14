#!/usr/bin/env python

from distutils.core import setup

setup(name = "CBC.Solve",
      version = "0.1.0-dev",
      description = "CBC Solver Collection",
      author = "Simula Research Laboratory and <authors>",
      author_email = "logg@simula.no",
      url = "http://www.fenics.org/wiki/CBC.Solve/",
      packages = ["cbc",
                  "cbc.common",
                  "cbc.twist"],
      package_dir = {"cbc": "cbc"},
      scripts = [],
      data_files = [])
