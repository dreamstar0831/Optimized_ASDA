# -*- coding: utf-8 -*-
"""
Created on Friday May 1 2026

Name: VTA.py

Purpose: Using the Vortex Tracking Algorithm to track vortices 



"""
__author__ = 'Quan Xie'
__license__ = 'GPLv3'
__date__ = '2026/05/01'
__maintainor__ = 'Quan Xie'
__email__ = 'xq30@mail.ustc.edu.cn'

import numpy as np
from matplotlib.path import Path

def select_candidate(vortex, pvortex, plabel, dmax, p2vortex=None, p2label=None, three_frame=False):
    """
    For each vortex in the current frame, candidate matches are searched for in the previous frame (or the two preceding frames).
    """
    if p2vortex is None:
        p2vortex = {'center': [], 'vr': []}
    if p2label is None:
        p2label = []

    n_curr = len(vortex['radius'])
    candidates = []

    for i in range(n_curr):
        center = np.array(vortex['center'][i])
        vr = vortex['vr'][i]
        points = vortex['points'][i]
        found = False

        # 1. 先在前一帧找（优先匹配）
        for j in range(len(plabel)):
            pcenter = np.array(pvortex['center'][j])
            pvr = pvortex['vr'][j]
            ppoints = pvortex['points'][j]
            if np.sign(vr) == np.sign(pvr):
                d = np.linalg.norm(center - pcenter)
                if d <= dmax:
                    count = len(
                        set(tuple(p) for p in points) & 
                        set(tuple(p) for p in ppoints)
                    )
                    candidates.append({
                        'curr_idx': i,
                        'prev_label': plabel[j],
                        'dist': d,
                        'count': count,
                        'source': 'prev'
                    })
                    found = True

        # if not found and three_frames = True，then search for cadidates in the previous two frames
        if three_frame:
            for k in range(len(p2label)):
                ppcenter = np.array(p2vortex['center'][k])
                ppvr = p2vortex['vr'][k]
                pppoints = p2vortex['points'][k]
                ppvortex_label = p2label[k]
                if np.sign(vr) == np.sign(ppvr):
                    d = np.linalg.norm(center - ppcenter)
                    if d <= dmax:  
                        skip = False
                        for m in range(len(plabel)):
                            pvortex_label = plabel[m]
                            if ppvortex_label == pvortex_label:
                                skip = True
                                break
                        if not skip:
                            count = len(
                                set(tuple(p) for p in points) & 
                                set(tuple(p) for p in pppoints)
                            )
                            candidates.append({
                                'curr_idx': i,
                                'prev_label': p2label[k],
                                'dist': d,
                                'count': count,
                                'source': 'p2'
                            })

    return candidates


def get_label(vortex, pvortex, plabel, dmax, three_frame=False, p2vortex=None, p2label=None):
    
    p2label = list(p2label) if p2label is not None else []
    
    n_curr = len(vortex['radius'])
    if n_curr == 0:
        return np.array([], dtype=int)
   
    plabel = np.array(plabel) if len(plabel) > 0 else np.array([], dtype=int)
    
    candidates = select_candidate(
        vortex, pvortex, plabel, dmax,
        p2vortex=p2vortex, p2label=p2label,
        three_frame=three_frame
    )
    
    label = np.full(n_curr, -1, dtype=int)
    used_labels = set()
    assigned_curr = set()
    
    candidates.sort(key=lambda x: (x['source'] != 'prev', x['dist']))
    
    for cand in candidates:
        i = cand['curr_idx']
        if i in assigned_curr:
            continue
            
        candidates_i = [c for c in candidates if c['curr_idx'] == i]
        if not candidates_i:
            continue
            
        target = min(candidates_i, key=lambda x: (-x['count'], x['dist']))
        
        if target['prev_label'] in used_labels:
            continue
            
        label[i] = target['prev_label']
        assigned_curr.add(i)
        used_labels.add(target['prev_label'])
    
    max_prev = np.max(plabel) if len(plabel) > 0 else 0
    max_p2 = np.max(p2label) if p2label and len(p2label) > 0 else 0
    max_existing = max(max_prev, max_p2, np.max(label) if n_curr > 0 else 0)
    next_new_label = int(max_existing) + 1
    
    for i in range(n_curr):
        if label[i] == -1:
            label[i] = next_new_label
            next_new_label += 1
    
    return label


def group_swirls(im_path, label_name='label', dmax=5.0, three_frame=False):
    '''
    Assign group IDs to vortices in different frames, where the same group ID indicates the same vortex.
    '''
    nt =  # total frames
    
    vortex_file = im_path + '0/vortex.npz'
    vortex = dict(np.load(vortex_file, allow_pickle=True))['vortex'].item()
    n_vortex = len(vortex['radius'])
    label = np.arange(0, n_vortex)  # 0,1,2,...,n_vortex-1
    np.save(im_path + '0/' + label_name + '.npy', label)
    
    for i in range(1, nt):
        vortex_file = im_path + f"{i}/vortex.npz"
        pvortex_file = im_path + f"{i-1}/vortex.npz"
        plabel_file = im_path + f"{i-1}/" + label_name + '.npy'
        
        vortex = dict(np.load(vortex_file, allow_pickle=True))['vortex'].item()
        pvortex = dict(np.load(pvortex_file, allow_pickle=True))['vortex'].item()
        plabel = np.load(plabel_file)
        
        if i == 1:  
            label = get_label(vortex, pvortex, plabel, dmax, three_frame=False)
        else:  
            p2vortex_file = im_path + f"{i-2}/vortex.npz"
            p2label_file = im_path + f"{i-2}/" + label_name + '.npy'
            
            p2vortex = dict(np.load(p2vortex_file, allow_pickle=True))['vortex'].item()
            p2label = np.load(p2label_file)
            
            if three_frame:
                label = get_label(vortex, pvortex, plabel, dmax, three_frame=True, p2vortex=p2vortex, p2label=p2label)
            else:
                label = get_label(vortex, pvortex, plabel, dmax, three_frame=False)
        
        np.save(im_path + f"{i}/" + label_name + '.npy', label)


def label_frames(im_path, label_name='label', info_name='label_info'):
    '''
    Generate a label information file that records the frames in which each label appears.
    '''
    nt = # total frames

    labels = {}
    for i in range(nt):
        label_file = im_path + f"{i}/" + label_name + '.npy'
        labels[i] = np.load(label_file)
    
    all_labels = set()
    for i in range(nt):
        all_labels.update(labels[i])
    n_label = max(all_labels) + 1 if all_labels else 0
    
    label_info = {}
    for j in range(n_label):
        frames = [i for i in range(nt) if j in labels[i]]
        label_info[f"{j}"] = frames
    
    info_file = im_path + info_name + '.npz'
    np.savez(info_file, **label_info)
    return label_info


if __name__ == '__main__':
    prefix = # your path to store the information of vortices
    im_paths = [prefix]
    label_name = 'label'
    info_name = 'label_info'
    three_frame = True  
    ds, dt = # pixel size and cadence of your data
    dmax = 7 * dt / ds  # assume the maximum movement speed of vortices to be 7 km/s
    print(dmax)
    if three_frame:
        label_name = label_name + '3_impro'
        info_name = info_name + '3_impro'   
    else:
        label_name = label_name + '_impro'
        info_name = info_name + '_impro' 
    for im_path in im_paths:
        group_swirls(im_path, label_name=label_name, dmax=dmax, three_frame=three_frame)
        label_info = label_frames(im_path, label_name=label_name, info_name=info_name)