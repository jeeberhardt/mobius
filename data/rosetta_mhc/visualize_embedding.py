#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Visualize MD trajectories using Pymol and Bokeh """

from __future__ import print_function

import os
import sys
import random
import argparse
import warnings
import subprocess
import numpy as np
import pandas as pd

from xmlrpc.client import ServerProxy
from bokeh.layouts import row, column
from bokeh.models import HoverTool, ColumnDataSource, BoxSelectTool, LassoSelectTool
from bokeh.plotting import curdoc, figure
from bokeh.io import show
from bokeh.models import CheckboxButtonGroup, RangeSlider, MultiSelect

warnings.filterwarnings("ignore")

__author__ = "Jérôme Eberhardt"
__copyright__ = "Copyright 2016, Jérôme Eberhardt"

__lience__ = "MIT"
__maintainer__ = "Jérôme Eberhardt"
__email__ = "qksoneo@gmail.com"


class Visualize():

    def __init__(self, covid_directory, config_file):

        # Start Bokeh server
        #if not self.is_screen_running("visu_bokeh"):
        #    print("Error: Bokeh is not running")
        #    sys.exit(1)

        # Start PyMOL
        #if not self.is_screen_running("visu_pymol"):
        #    print("Error: Pymol is not running")
        #    sys.exit(1)

        self._rpc_port = 9123
        self._pymol = ServerProxy(uri="http://localhost:%s/RPC2" % self._rpc_port)
        self._controls = {}
        self._source = None

        self._mol_directory = mol_directory
        self._df = pd.read_csv(config_file)

        self._df_tmp = self._df.copy()

    def is_screen_running(self, sname):
        output = subprocess.check_output(["screen -ls; true"], shell=True, universal_newlines=True)
        return [l for l in output.split("\n") if sname in l]

    def update_pymol(self, indices):
        if indices:
            try:
                self._pymol.do("delete *_s")

                molecules = self._df_tmp.iloc[indices]

                reference = molecules['peptide'].values[0]

                print(molecules['peptide'].values)

                for m in molecules.itertuples():
                    molecule_file = '%s/%s-ensemble.pdb' % (self._mol_directory, m.peptide)
                    self._pymol.do('load %s, %s_s' % (molecule_file, m.peptide))
                    self._pymol.do('super %s_s, %s_s' % (m.peptide, reference))

                self._pymol.do("zoom %s_s, 5" % m.peptide)
                self._pymol.do("show lines")
                self._pymol.do("hide (h.)")
                self._pymol.do('hide cartoon, resi 181-189')
                self._pymol.do('hide lines, resi 181-189')
            except:
                print("Connection issue with PyMol! (Cmd: pymol -R)")

    def get_selected_frames(self, attr, old, new):
        self.update_pymol(new)

    def update(self):
        self._df_tmp = self.select_molecules()

        data = dict(x=self._df_tmp['x'], y=self._df_tmp['y'],
                    molecule=self._df_tmp['peptide'])
        self._source.data = data

    def show(self):
        # Store some informations
        title = ""
        TOOLS = "wheel_zoom,box_zoom,undo,redo,box_select,save,reset,hover,crosshair,tap,pan"

        # Figure
        p = figure(plot_width=1500, plot_height=1500, tools=TOOLS, title=title, output_backend="webgl")
        p.title.text_font_size = '20pt'
        p.select(BoxSelectTool).select_every_mousemove = False
        p.select(LassoSelectTool).select_every_mousemove = False

        data  = dict(x=self._df['x'], y=self._df['y'], molecule=self._df['peptide'])
        self._source = ColumnDataSource(data=data)
        s = p.scatter(x="x", y="y", source=self._source, line_color="black", size=10)

        # Selection
        s.data_source.selected.on_change('indices', self.get_selected_frames)

        # Create Hovertools
        tooltips = [("(X, Y)", "(@x @y)"), 
                    ("Peptide", "@molecule")]

        hover = p.select({"type": HoverTool})
        hover.tooltips = tooltips

        # Go!
        layout = column(row(p,), sizing_mode="scale_both")
        curdoc().add_root(layout)


def parse_options():
    parser = argparse.ArgumentParser(description="visu 2D configuration")
    parser.add_argument("-d", "--mol_dir", dest="mol_directory", required=True,
                        action="store", type=str,
                        help="molecule directory")
    parser.add_argument("-c", "--configuration", dest="config_file",
                        required=True, action="store", type=str,
                        help="configuration file")

    args = parser.parse_args()

    return args


options = parse_options()
mol_directory = options.mol_directory
config_file = options.config_file

V = Visualize(mol_directory, config_file)
V.show()
