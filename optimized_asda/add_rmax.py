#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 05 2025

Name: add_rmax.py

Purpose: add the property of rmax into the vortex structure

This code is based on code from https://github.com/PyDL/ASDA/blob/master/asda/add_rmax.py.

Original license: 'GPLv2'
Copyright: 'Copyright 2017, The Solar Physics and Space Plasma ' + \
                'Research Center (SP2RC)'
@author: Jaijia Liu at University of Science and Technology of China      
          
Modifications by: Quan Xie, xq30@mail.ustc.edu.cn
Modified on: 2025-08-05
Description: revise the file to adapt Optimized ASDA. 
"""

__author__ = 'Jiajia Liu (modified by Quan Xie)'
__copyright__ = 'Copyright 2025, University of Sci. & Tech. China'
__license__ = 'GPLv3'
__version__ = '1.1.0'
__date__ = '2025/08/05'
__maintainor__ = 'Quan Xie'
__email__ = 'xq30@mail.ustc.edu.cn'

import numpy as np
from optimized_asda.vortex import read_vortex, save_vortex

def add_rmax(im_path, filename='vortex.npz'):

    ds_dt = np.load(im_path + 'ds_dt.npz')
    nt = len(ds_dt['dt'])

    for i in range(nt):
        current = im_path + '{:d}'.format(i) + '/'
        vortex = read_vortex(filename=current + filename)
        center = vortex['center']
        edge = vortex['edge']
        n = len(center)
        rmax = []
        for j in np.arange(n):
            e = edge[j]
            c = center[j]
            d = np.linalg.norm(np.subtract(e, c), axis=1)
            rmax.append(np.max(d))
        vortex['rmax'] = rmax
        save_vortex(vortex, filename=current + filename)


if __name__ == '__main__':
#    prefix = './SST/'
#    im_paths = [prefix + 'Swirl/6563core/',
#                prefix + 'Swirl/8542core/',
#                prefix + 'Swirl/6302wb/']
    prefix = './SOT/'
    im_paths = [prefix + 'Swirl/FG-blue/',
                prefix + 'Swirl/CaII-H/']
    for im_path in im_paths:
        add_rmax(im_path, filename='vortex.npz')