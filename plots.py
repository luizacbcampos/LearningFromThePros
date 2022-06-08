import re
import json

import numpy as np
import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import cv2


from mplsoccer import VerticalPitch


mpii_edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], 
              [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], 
              [6, 8], [8, 9]]


# ---- aux ----

def ImageID(df, array_id):
    #Get photo id's of poses
    return df.loc[array_id, 'file']

def plot_rectangle(points, bbox):
    '''
        Plots the skeleton points and the minimum rectangle
    '''
    plt.scatter(points[:,0], points[:,1])
    plt.fill(bbox[:,0], bbox[:,1], alpha=0.2)
    plt.axis('equal')
    plt.show()

def plot3D(ax, points, edges, marker_size = 100):
    ax.grid(False)
    oo = 1e10
    xmax,ymax,zmax = -oo,-oo,-oo
    xmin,ymin,zmin = oo, oo, oo
    #edges = mpii_edges
    c, marker = 'b', 'o'
    points = points.reshape(-1, 3)
    x, y, z = np.zeros((3, points.shape[0]))

    for j in range(points.shape[0]):
        x[j] = points[j, 0].copy()
        y[j] = points[j, 2].copy()
        z[j] = -points[j, 1].copy()
        xmax = max(x[j], xmax)
        ymax = max(y[j], ymax)
        zmax = max(z[j], zmax)
        xmin = min(x[j], xmin)
        ymin = min(y[j], ymin)
        zmin = min(z[j], zmin)

    ax.scatter(x, y, z, s = marker_size, c = c, marker = marker)
    
    for e in edges:
        ax.plot(x[e], y[e], z[e], c = c)
    
    max_range = np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xmax+xmin)
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ymax+ymin)
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zmax+zmin)
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')


def plot2D(ax, pose_3d, mpii_edges):
	'''
		2D plot of the 3D body pose in the x-y plane (ignoring z-axis)
	'''
	for e in range(len(mpii_edges)):
		ax.plot(pose_3d[mpii_edges[e]][:, 0], -1*pose_3d[mpii_edges[e]][:, 1])

	ax.set_xlabel('x')
	ax.set_ylabel('y')


def clusterExamples(k, n_examples, path, model_clusters, pose_df, pose_arr, mpii_edges, save):
    ax_array = np.linspace(1, k * 2 * n_examples - (k * 2 - 1), n_examples).astype(int)
    fig = plt.figure(figsize=(15, 15))

    for a in ax_array:
        addition = 0
        for cluster in range(k):
            arr_id = np.random.choice(np.where(model_clusters == cluster)[0])
            photo_id = ImageID(pose_df, arr_id)
            ax = fig.add_subplot(n_examples, k*2, a + addition)
            ax.imshow(importImage(path + photo_id))
            ax.set_xticks([])
            ax.set_yticks([])
            addition += 1
            
            ax = fig.add_subplot(n_examples, k*2, a+addition)
            plot2D(ax, pose_to_matrix(pose_arr[arr_id][:-1]), mpii_edges)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            if a == 1:
                ax.set_title('Cluster ' + str(cluster), position=(-0.1, 1), size=16)
            addition += 1

    plt.tight_layout()
    plt.savefig('viz/' + save + '.png')
    plt.show()


def plotXSMap(xs_map, num_clusters, cluster_names):
    
    pitch = VerticalPitch(half=True, goal_type='box', pad_bottom=-30, 
                          pad_left=-10, pad_right=-10, line_color='black')
    
    fig, ax = pitch.draw(figsize=(10, 5), nrows=1, ncols=num_clusters, tight_layout=True)
    
    for i in range(num_clusters):
        im = ax[i].imshow(xs_map[i], cmap=plt.cm.Greens, interpolation='none', 
                       vmin=xs_map.min(), vmax=xs_map.max(), extent=[0,80,120,90])
        ax[i].set_title('Cluster ' + str(i) + ': ' + cluster_names[i])
    
    cax = plt.axes([1, 0.3, 0.05, 0.4])
    plt.colorbar(im, cax=cax)


def plotBestTechniqueUp(xs_map, xs_map_up, cluster_name):
    '''
    	Best technique to use
    '''

    pitch = VerticalPitch(half=True, goal_type='box', pad_bottom=-37, 
                          pad_left=-15, pad_right=-15, line_color='black',
                          orientation='horizontal')
    fig, ax = pitch.draw(figsize=(10,5), nrows=1, ncols=2)
    
    cmap = plt.cm.tab20
    
    im = ax[0].imshow(np.argmax(xs_map, axis=0), cmap=cmap,
                      interpolation='none', extent=[0,80,120,90])
    
    im = ax[1].imshow(np.argmax(xs_map_up, axis=0), cmap=cmap,
                      interpolation='none', extent=[0,80,120,90])
    
    ax[0].set_title('Striker Not Under Pressure')
    ax[1].set_title('Striker Under Pressure')
    
    custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                    Line2D([0], [0], color=cmap(0.33), lw=4),
                    Line2D([0], [0], color=cmap(0.66), lw=4),
                    Line2D([0], [0], color=cmap(1.), lw=4)]
    
    ax[1].legend(custom_lines, cluster_name, loc=1, bbox_to_anchor=(1, 0.38))
    plt.tight_layout()


def plotDoubleXSMap(xs_map, xs_map_up, cluster_names, num_clusters=4):

    pitch = VerticalPitch(half=True, goal_type='box', pad_bottom=-30, 
                          pad_left=-10, pad_right=-10, line_color='black')
    
    fig, ax = pitch.draw(figsize=(10, 5), nrows=2, ncols=num_clusters, tight_layout=True)
    
    min_v = np.min([xs_map.min(),xs_map_up.min()])
    max_v = np.max([xs_map.max(),xs_map_up.max()])
    
    for i in range(num_clusters):
        im = ax[0, i].imshow(xs_map[i], cmap=plt.cm.Greens, interpolation='none', 
                       vmin=min_v, vmax=max_v, extent=[0,80,120,90])
        ax[0, i].set_title('Cluster ' + str(i) + ': ' + cluster_names[i])
        if i == 0:
            ax[0,i].set_ylabel('No Pressure', rotation=0, labelpad=33)
    
    for i in range(num_clusters):
        im = ax[1, i].imshow(xs_map_up[i], cmap=plt.cm.Greens, interpolation='none', 
                       vmin=min_v, vmax=max_v, extent=[0,80,120,90])
        #ax[1, i].set_title('Cluster ' + str(i) + ': ' + cluster_names[i])
        if i == 0:
            ax[1,i].set_ylabel('Pressure', rotation=0, labelpad=33)
    
    cax = plt.axes([1, 0.3, 0.05, 0.4])
    cax.set_title('xSAA')
    plt.colorbar(im, cax=cax)
    plt.tight_layout()


if __name__ == '__main__':
	print("main")