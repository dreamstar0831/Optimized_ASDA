# -*- coding: utf-8 -*-
"""
Created on Mon Aug 4 2025

Name: vortex.py

Original Author: [Jiajia Liu]  
Original Repository: https://github.com/PyDL/ASDA

Modified by: [Quan Xie]  
Modified Date: 2025-08-04  
Repository: https://github.com/dreamstar0831/Optimized_ASDA

Discription: 
        This module is based on the original implementation from Jiajia Liu,
        with the following modifications:
        - gamma_values_parallel: add the option of using variable gamma calculating method to calculate gamma 
        - gamma_values: add the option of using variable gamma calculating method to calculate gamma
        - center_edge: modify the value of gamma_min from 0.89 to 0.63 (see Xie et al. 2025, A&A)
@author: Quan Xie


"""
__author__ = 'Jiajia Liu (modified by Quan Xie)'
__copyright__ = 'Copyright 2025, University of Sci. & Tech. China'
__license__ = 'GPLv3'
__version__ = '1.1.0'
__date__ = '2025/08/04'
__maintainor__ = 'Quan Xie'
__email__ = 'xq30@mail.ustc.edu.cn'

import numpy as np
from optimized_asda.points_in_poly import points_in_poly
from scipy.interpolate import interp2d
from itertools import product
from skimage import measure
import h5py
import multiprocessing as mp
from copy import copy


def reform2d(array, factor=1):
    """
    Reform a 2d array by a given factor

    Parameters
    ----------
    array : `numpy.ndarray`
        2d array to be reformed

    factor : `int`
        The array is going to be magnified by the factor. Default is 1.

    Returns
    -------
        `numpy.ndarray`
        reformed array

    """
    if not isinstance(factor, int):
        raise ValueError("Parameter 'factor' must be an integer!")

    if len(np.shape(array)) != 2:
        raise ValueError("Input array must be 2d!")

    if factor > 1:
        congridx = interp2d(np.arange(0, array.shape[0]),
                            np.arange(0, array.shape[1]), array.T)
        array = congridx(np.arange(0, array.shape[0], 1/factor),
                         np.arange(0, array.shape[1], 1/factor)).T

    return array


def remove_duplicate(edge):
    """
    Remove duplicated points in a the edge of a polygon

    Parameters
    ----------
    edge : `list` or `numpy.ndarray`
        n x 2 list, defines all points at the edge of a polygon

    Returns
    -------
        `list`
        same as edge, but with duplicated points removed
    """

    shape = np.shape(edge)
    if shape[1] != 2:
        raise ValueError("Polygon must be defined as a n x 2 array!")

    new_edge = []
    for i in range(shape[0]):
        p = edge[i]
        if not isinstance(p, list):
            p = p.tolist()
        if p not in new_edge:
            new_edge.append(p)

    return new_edge


def gamma_values_parallel(vx, vy, r, factor=1, ncpus=None):
    '''
    Purpose: calculate gamma1 and gamma2 values of velocity field vx, vy
             using multiprocessing parallelisation

    Inputs:
        vx - velocity field in x direction
        vy - velocity field in y direction
        r - maximum distance of neighbour points from target point
            if r='variable', employ the variable gamma calculating method (r=5, 7, 9, and 11) to calculate gamma
            if r is a single number, employ a single r to calculate gamma  
        factor - default is 2. Magnify the original data to find sub-grid
                 vortex center and boundary
        ncpus - number of cpus to be used, default is the maximum number of
                cpus available
    Outputs:
        gamma - tuple in form of (gamma1, gamma2), where gamma1 is useful in finding
              vortex centers and gamma2 is useful in finding vortex edges

    '''
    # do basic checks
    vx = np.array(vx, dtype=np.float32)
    vy = np.array(vy, dtype=np.float32)
    if vx.shape != vy.shape:
        print("Velocity field vx and vy do not match!")
        return None
    if r == 'adaptive': # employ the VGCM method (r=5, 7, 9, and 11) to calculate gamma
        r_values = [5, 7, 9, 11]
        gamma1_list = []
        gamma2_list = []

        for r_val in r_values:
            g = gamma_values_parallel(vx, vy, r=r_val, factor=1, ncpus=ncpus)
            gamma1_list.append(g[0])
            gamma2_list.append(g[1])

        gamma1_stack = np.stack(gamma1_list, axis=-1)
        gamma2_stack = np.stack(gamma2_list, axis=-1)

        idx1 = np.argmax(np.abs(gamma1_stack), axis=-1)
        idx2 = np.argmax(np.abs(gamma2_stack), axis=-1)

        ny, nx = vx.shape
        max_gamma1 = gamma1_stack[np.arange(ny)[:, None], np.arange(nx)[None, :], idx1]
        max_gamma2 = gamma2_stack[np.arange(ny)[:, None], np.arange(nx)[None, :], idx2]

        return np.array([max_gamma1, max_gamma2])
    
    elif isinstance(r, (int, float)): # employ a single r to calculate gamma
        r = int(r)
        factor = int(factor)
        if factor > 1:
            vx = reform2d(vx, factor=factor)
            vy = reform2d(vy, factor=factor)
        ny = vx.shape[1]
        factor = 1
        # initialise gamma
        gamma = np.array([np.zeros_like(vx),
                        np.zeros_like(vy)])
        # number of threads to be used
        if ncpus is None or ncpus <= 0:
            ncpus = mp.cpu_count()

        if ny > ncpus:
            # initialise the multiprocess
            manager = mp.Manager()
            return_dict = manager.dict()
            jobs = []
            # slit the data
            step = int(int(ny) / int(ncpus))
            i0_ext = np.arange(ncpus, dtype=int) * step
            i1_ext = (np.arange(ncpus, dtype=int) + 1) * step
            i1_ext[-1] = ny
            i0 = copy(i0_ext)
            i1 = copy(i1_ext)
            i0_ext[1:] = i0_ext[1:] - r
            i1_ext[0:-1] = i1_ext[0:-1] + r
            # do multiprocessing
            for i in range(ncpus):
                p = mp.Process(target=gamma_values, args=(vx[:, i0_ext[i]:i1_ext[i]],
                                                        vy[:, i0_ext[i]:i1_ext[i]],
                                                        r, factor, return_dict,
                                                        i))
                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()

            # join the data
            for i in range(ncpus):
                if i == 0:
                    gamma[:, :, i0[i]:i1[i]] = return_dict[i][:, :, 0:-r]
                elif i == ncpus-1:
                    gamma[:, :, i0[i]:i1[i]] = return_dict[i][:, :, r:]
                else:
                    gamma[:, :, i0[i]:i1[i]] = return_dict[i][:, :, r:-r]
        else:
            gamma = gamma_values(vx, vy, r=r, factor=1)

        return gamma


def gamma_values(vx, vy, r, factor=1, return_values=None, procnum=None):
    '''
    Purpose: calculate gamma1 and gamma2 values of velocity field vx and vy
             Fomulars can be found in Graftieaux et al. 2001
    Inputs:
        vx - velocity field in x direction
        vy - velocity field in y direction
        r - maximum distance of neighbour points from target point
            if r='variable', employ the variable gamma calculating method (r=5, 7, 9, and 11) to calculate gamma
            if r is a single number, employ a single r to calculate gamma  
        factor - default is 2. Magnify the original data to find sub-grid
                 vortex center and boundary
        return_values, procnum - inputs designed to be used in
                gamma_values_parallel().
    Outputs:
        gamma - tuple in form of (gamma1, gamma2), where gamma1 is useful in finding
              vortex centers and gamma2 is useful in finding vortex edges
    '''

    vx = np.array(vx, dtype=np.float32)
    vy = np.array(vy, dtype=np.float32)
    if vx.shape != vy.shape:
        print("Velocity field vx and vy do not match!")
        return None
    if r == 'adaptive': # employ the VGCM method (r=5, 7, 9, and 11) to calculate gamma
        r_values = [5, 7, 9, 11]
        gamma1_list = []
        gamma2_list = []

        for r_val in r_values:
            g = gamma_values(vx, vy, r=r_val, factor=1)
            gamma1_list.append(g[0])
            gamma2_list.append(g[1])

        gamma1_stack = np.stack(gamma1_list, axis=-1)
        gamma2_stack = np.stack(gamma2_list, axis=-1)

        idx1 = np.argmax(np.abs(gamma1_stack), axis=-1)
        idx2 = np.argmax(np.abs(gamma2_stack), axis=-1)

        ny, nx = vx.shape
        max_gamma1 = gamma1_stack[np.arange(ny)[:, None], np.arange(nx)[None, :], idx1]
        max_gamma2 = gamma2_stack[np.arange(ny)[:, None], np.arange(nx)[None, :], idx2]

        return np.array([max_gamma1, max_gamma2])
    elif isinstance(r, (int, float)): # employ a single r to calculate gamma       
        r = int(r)
        factor = int(factor)
        if factor > 1:
            vx = reform2d(vx, factor=factor)
            vy = reform2d(vy, factor=factor)
        nx = vx.shape[0]
        ny = vx.shape[1]

        gamma = np.array([np.zeros_like(vx),
                        np.zeros_like(vy)])
        # pm vectors, see equation (8) in Graftieaux et al. 2001 or Equation
        # (1) in Liu et al. 2019
        pm = np.array([[i, j]
                        for i in np.arange(-r, r + 1)
                        for j in np.arange(-r, r + 1)], dtype=float)

        # mode of vector pm
        pnorm = np.linalg.norm(pm, axis=1)

        # Number of points in the concerned region
        N = (2 * r + 1) ** 2

        # Create index array
        index = np.array([[i, j]
                            for i in np.arange(r, ny - r)
                            for j in np.arange(r, nx - r)])

        # Transpose index
        index = index.T

        # Generate velocity field
        vel = gen_vel(vx, vy, index[1], index[0], r=r)

        # Iterate over the array gamma
        for d, (i, j) in enumerate(product(np.arange(r, ny - r, 1),
                                            np.arange(r, nx - r, 1))):

            gamma[0, j, i], gamma[1, j, i] = calc_gamma(pm, vel[..., d], pnorm, N)

        if (procnum is not None) and (return_values is not None):
            return_values[procnum] = gamma
            
        return gamma


def center_edge(gamma1, gamma2, factor=1, rmin=1, gamma_min=0.63):
    '''
    Find vortices from gamma1, and gamma2
    Output:
        center: center location of vortices
        edge: edge location of vortices
        points: all points within vortices
        peak: maximum/minimum gamma1 value in vortices
        radius: equivalent radius of vortices
        All in pixel coordinates
    '''
    # ------------ deprecated ------------------------------------------------
    # matplotlib.interactive(False)
    # plt.subplots()
    # cs = plt.contour(gamma2.T, levels=[-2 / np.pi, 2 / np.pi])
    # plt.close()

    edge = ()
    center = ()
    points = ()
    peak = ()
    radius = ()
    cs = np.array(measure.find_contours(gamma2, -2 / np.pi),dtype=object)
    cs_pos = np.array(measure.find_contours(gamma2, 2 / np.pi),dtype=object)
    # fix bug when len(cs) or len(cs_pos) is 1
    if len(cs) == 1:
        cs = [np.squeeze(cs), [[]]]
    if len(cs_pos) == 1:
        cs_pos = [np.squeeze(cs_pos), [[]]]

    if len(cs) == 0:
        cs = cs_pos
    elif len(cs_pos) != 0:
        cs = np.append(cs, cs_pos, 0)
    for i in range(np.shape(cs)[0]):
        v = np.rint(cs[i])
        if np.shape(v)[1] == 2:
            v = remove_duplicate(v)
            ps = points_in_poly(v)
            dust = []
            for p in ps:
                dust.append(gamma1[int(p[0]), int(p[1])])
            if len(dust) > 1:
                re = np.sqrt(np.array(ps).shape[0]/np.pi) / factor
                if np.max(np.fabs(dust)) >= gamma_min and re >= rmin :
                    # allow some error around 0.9
                    # vortex with radius less than 4 pixels is not reliable
                    idx = np.where(np.fabs(dust) == np.max(np.fabs(dust)))
                    idx = idx[0][0]
                    center = center + (np.array(ps[idx])/factor, )
                    edge = edge + (np.array(v)/factor, )
                    points = points + (np.array(ps)/factor, )
                    peak = peak + (dust[idx], )
                    radius = radius + (re, )

    return (center, edge, points, peak, radius)


def vortex_property(centers, edges, points, vx, vy, image=None):
    '''
    Calculate expanding, rotational speed, equivalent radius and average
        intensity of given swirls.
    Output:
        ve: expanding speed, pixel/frame
        vr: rotational speed, pixel/frame
        vc: velocity of the center, pixel/frame
        ia: average the observation values (intensity or magnetic field)
            within the vortices
    '''
    vx = np.array(vx)
    vy = np.array(vy)
    n_swirl = len(centers)
    ve = ()
    vr = ()
    vc = ()
    ia = ()

    for i in range(n_swirl):
        cen = centers[i]
        edg = edges[i]
        pnt = np.array(points[i], dtype=int)
        x0 = int(round(cen[0]))
        y0 = int(round(cen[1]))
        vcen = [vx[x0, y0], vy[x0, y0]]
        vc = vc + (vcen, )
        if image is not None:
            image = np.array(image)
            value = 0
            for pos in pnt:
                value = value + image[pos[0], pos[1]]
            value = value * 1.0 / pnt.shape[0]
        else:
            value = None
        ia = ia + (value, )
        ve0 = []
        vr0 = []
        for j in range(edg.shape[0]):
            idx = [edg[j][0], edg[j][1]]
            pm = [idx[0]-cen[0], idx[1]-cen[1]]
            tn = [cen[1]-idx[1], idx[0]-cen[0]]
            idx = np.array(idx, dtype=int)
            v = [vx[idx[0], idx[1]], vy[idx[0], idx[1]]]
            ve0.append(np.dot(v, pm)/np.linalg.norm(pm))
            vr0.append(np.dot(v, tn)/np.linalg.norm(tn))
        ve = ve + (np.nanmean(ve0), )
        vr = vr + (np.nanmean(vr0), )
        #vemax = np.nanmax(ve0)
        #vrmax = np.nanmax(vr0)

    return (ve, vr, vc, ia)


def gen_vel(vx, vy, i, j, r=3):
    """
    Given a point [i, j], generate a velocity field which contains
    a region with a size of (2r+1) x (2r+1) centered at [i, j] from
    the original velocity field vx and vy.

    Parameters
    ----------
    i : `int`
        first dimension of the pixel position of a target point.
    j : `int`
        second dimension of the pixel position of a target point.

    Returns:
    -------
        `numpy.ndarray`
        the first dimension is a velocity field which contains a
        region with a size of (2r+1) x (2r+1) centered at [i, j] from
        the original velocity field vx and vy.
        the second dimension is similar as the first dimension, but
        with the mean velocity field substracted from the original
        velocity field.
    """

    vel = np.array([[vx[i + im, j + jm], vy[i + im, j + jm]]
                    for im in np.arange(-r, r + 1)
                    for jm in np.arange(-r, r + 1)])

    return np.array([vel, vel - vel.mean(axis=0)])

def calc_gamma(pm, vel, pnorm, N):
    """
    Calculate Gamma values, see equation (8) in Graftieaux et al. 2001
    or Equation (1) in Liu et al. 2019

    Parameters
    ----------
        pm : `numpy.ndarray`
            vector from point p to point m
        vel : `numpy.ndarray`
            velocity vector
        pnorm : `numpy.ndarray`
            mode of pm
        N : `int`
            number of points

    Returns
    -------
        `float`
        calculated gamma values for velocity vector vel
    """

    cross = np.cross(pm, vel)
    vel_norm = np.linalg.norm(vel, axis=2)
    sint = cross / (pnorm * vel_norm + 1e-10)

    return np.nansum(sint, axis=1) / N


def save_vortex(vortex, filename='vortex.h5'):
    '''
    Save a vortex dictionary into a hdf5 file or npz file
    written on March 2021 by Jiajia Liu (j.liu@qub.ac.uk)
    Parameters
    ----------
    vortex : dictonary
        dictionary that contains the vortex detection result, should have
        9 keys: center, edge, points, peak, radius, ve, vr, vc, ia
    filename : string, optional
        file to be saved, extension must be .h5 or npz

    Returns
    -------
    None.

    '''
    # test keys of vortex
    keys = vortex.keys()
    if 'center' not in keys or 'edge' not in keys or 'points' not in keys \
        or 'radius' not in keys or 've' not in keys or 'vr' not in keys \
        or 'vc' not in keys:
            raise RuntimeError('input dictionary must contain all of' +
                               ' the following keys: center, edge, points,' +
                               ' peak, radius, ve, vr, vc')
    # check extension
    ext = filename[filename.rfind('.'):]
    if ext == '.h5':
        # number of vortices
        nvortex = len(vortex['vr'])
        # create hdf5 file
        file = h5py.File(filename, 'w')
        # because points and edge are irregular in shape, they need to be groups
        file.create_group('points')
        file.create_group('edge')
        for i in range(nvortex):
            name = '{:d}'.format(i)
            file['points'][name] = vortex['points'][i]
            file['edge'][name] = vortex['edge'][i]
        # all other keys can be datasets
        file['center'] = np.array(vortex['center'])
        file['radius'] = np.array(vortex['radius'])
        file['peak'] = np.array(vortex['peak'])
        file['ve'] = np.array(vortex['ve'])
        file['vr'] = np.array(vortex['vr'])
        file['vc'] = np.array(vortex['vc'])

        # ia is optional
        if 'ia' in keys:
            if None not in vortex['ia']:
                if len(vortex['ia']) > 0:
                    file['ia'] = np.array(vortex['ia'])
        # rmax is optional
        if 'rmax' in keys:
            file['rmax'] = np.array(vortex['rmax'])
    elif ext == '.npz':
        np.savez(filename, **vortex)
    else:
        raise RuntimeError('File extension must be npz or h5')


def read_vortex(filename='vortex.h5'):
    '''
    Read a vortex dictionary from a hdf5 file or npz file
    written on March 2021 by Jiajia Liu (j.liu@qub.ac.uk)
    Parameters
    ----------
    filename : string
        file to be read, extension must be h5 or npz

    Returns
    -------
    vortex : dictonary
        dictionary that contains the vortex detection result

    '''
    # check extension
    ext = filename[filename.rfind('.'):]
    if ext == '.npz':
        vortex = dict(np.load(filename, allow_pickle=True))
    elif ext == '.h5':
        vortex = {}
        # load hdf5 file
        file = h5py.File(filename, 'r')
        # all other keys are datasets
        vortex['center'] = np.array(file['center'])
        vortex['radius'] = np.array(file['radius'])
        vortex['peak'] = np.array(file['peak'])
        vortex['ve'] = np.array(file['ve'])
        vortex['vr'] = np.array(file['vr'])
        vortex['vc'] = np.array(file['vc'])
        # number of vortices
        nvortex = len(vortex['vr'])
        # because points and edge are irregular in shape, they need to be groups
        edge = ()
        points = ()
        for i in range(nvortex):
            name = '{:d}'.format(i)
            points = points + (np.array(file['points'][name], dtype=int), )
            edge = edge + (np.array(file['edge'][name], dtype=int), )

        vortex['edge'] = edge
        vortex['points'] = points
        # ia is optional
        keys = list(file.keys())
        if 'ia' in keys:
            vortex['ia'] = np.array(file['ia'])
        else:
            vortex['ia'] = None
        # rmax is optional
        if 'rmax' in keys:
            vortex['rmax'] = np.array(file['rmax'])
    else:
        raise RuntimeError('File extension must be npz or h5')

    return vortex