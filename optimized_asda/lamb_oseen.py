# -*- coding: utf-8 -*-
"""
Created on Tue Aug 05 2025

Name: lamb_oseen.py

This code is based on code from https://github.com/PyDL/ASDA/blob/master/asda/lamb_oseen.py.

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
import matplotlib.pyplot as plt


class Lamb_Oseen:
    def __init__(self, vmax=1, rmax=1, gamma=None, rcore=None,
                 ratio_vradial=0):
        '''
        Initialization of the Lamb Oseen vortex
        vmax: maximum value of v_theta, negative value for clockwise vortex
        rmax: radius of the position where v_theta reaches vmax
        ratio_vradial: ratio between expanding/shrinking speed and
                       rotating speed
        '''
        self.vmax = vmax
        self.rmax = rmax
        self.alpha = 1.256430
        self.rcore = self.rmax / np.sqrt(self.alpha)  # Core radius
        # gamma is the circulation of the vortex
        # in real world, gamma has unit of m^2 s^{-1}
        self.gamma = 2 * np.pi * self.vmax * self.rmax * (1 + 1/(2*self.alpha))
        if gamma is not None and rcore is not None:
            self.gamma = gamma
            self.rcore = rcore
            self.rmax = self.rcore * np.sqrt(self.alpha)
            self.vmax = self.gamma / (2 * np.pi
                                      * self.rmax * (1 + 1/(2*self.alpha)))
        self.vcore = (1 - np.exp(-1.0)) * self.gamma / (2 * np.pi * self.rcore)
        self.ratio_vradial = ratio_vradial

    def get_grid(self, xrange=[], yrange=[]):
        '''
        Return meshgrid of the coordinate of the vortex
        '''
        if len(xrange) != 2:
            self.xrange = [0-self.rmax, self.rmax]
        else:
            self.xrange = xrange
        if len(yrange) != 2:
            self.yrange = [0-self.rmax, self.rmax]
        else:
            self.yrange = yrange
        x = np.arange(self.xrange[0], self.xrange[1])
        y = np.arange(self.yrange[0], self.yrange[1])
        xx, yy = np.meshgrid(x, y)

        self.xx = xx
        self.yy = yy

        return xx, yy

    def get_vtheta(self, r=0):
        '''
        Return v_theta at radius of r
        '''
        r = r + 1e-10
        vtheta = self.gamma \
                 *(1.0-np.exp(0-np.square(r)/np.square(self.rcore))) \
                 /(2*np.pi*r)
        return vtheta


    def get_vradial(self, r=0):

        r = r + 1e-10
        vtheta = self.get_vtheta(r)
        vradial = vtheta * self.ratio_vradial

        return vradial

    def get_vxvy(self, x=None, y=None, xrange=[], yrange=[]):
        '''
        calculate vx and vy value at point (x, y)
        '''
        if x is None or y is None:
            x, y = self.get_grid(xrange=xrange, yrange=yrange)
        r = np.sqrt(np.square(x) + np.square(y)) + 1e-10
        vtheta = self.get_vtheta(r)
        vradial = self.get_vradial(r)
        vector = [0 - vtheta * y + vradial * x, vtheta * x + vradial * y]
        vx = vector[0] / r
        vy = vector[1] / r
        return vx, vy


if __name__ == '__main__':
    lo = Lamb_Oseen(vmax=0.4, rmax=10)
    xx, yy = lo.get_grid(xrange=[], yrange=[])
    vx, vy = lo.get_vxvy(xrange=[], yrange=[])
    plt.streamplot(xx, yy, vx, vy)
    print(lo.rcore, lo.vcore)
    print(xx,yy)
    print(vx,vy)
    print(vx[0,0],vx[0,1],vx[1,1])
    plt.show()