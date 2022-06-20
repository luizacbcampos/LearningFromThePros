import re
import cv2
import json

import numpy as np
import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


import seaborn as sns
from mplsoccer import VerticalPitch
import auxi

sns.set()

mpii_edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], [6, 8], [8, 9]]

altura, largura = 244, 732 # goal dimensions
positions = {
    "bottom left corner" : [(1,1), largura/3, altura/2, "green"],
    "bottom right corner" : [(2*largura/3, 1), largura/3, altura/2, "pink"],
    "centre of the goal" : [(largura/3, 1), largura/3, altura/2, "orange"],
    "top right corner" : [(2*largura/3, altura/2), largura/3, altura/2, "red"],
    "top left corner" : [(1, altura/2), largura/3, altura/2, "blue"],
    "top centre of the goal" : [(largura/3, altura/2), largura/3, altura/2, "purple"],

    "the bar" : None, "left post" : None, "left" : None,
    "just a bit too high" : None,
    "too high" : None, "right post" : None, "right" : None,
}

# pose converter

def pose_to_matrix(pose):
    if len(pose) == 48:
        pose_matrix = pose.reshape(16, 3)
    else:
        pose_matrix = pose.reshape(16, 2)
    return pose_matrix


# ---- aux ----
def importImage(img):
    '''
        Import image
    '''
    image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    return image

def ImageID(df, array_id):
    '''
        Get photo id's of poses
    '''
    return df.loc[array_id, 'file']

def remove_ax_ticks(ax, s=2):
    '''
        Removes the ticks from s axis 
    '''
    ax.set_xticks([])
    ax.set_yticks([])
    if s==3:
        ax.set_zticks([])


def remove_labels(ax, s=2):
    '''
        Removes ax label
    '''
    ax.set_xlabel('')
    ax.set_ylabel('')
    if s==3:
        ax.set_zlabel('')

# Plots

def plot_rectangle(points, bbox, show=False):
    '''
        Plots the skeleton points and the minimum rectangle
    '''
    plt.scatter(points[:,0], points[:,1])
    plt.fill(bbox[:,0], bbox[:,1], alpha=0.2)
    plt.axis('equal')

    if show:
        plt.show()

def plot_camera_view_invariance(sets_3d, set_3d_df, sets_3d_cvi, pose_id=319, path='images/1v1_images/', show=False):

    photo_id = set_3d_df.loc[pose_id,'file']
    fig = plt.figure(figsize=(20, 4))
    ax = fig.add_subplot(1, 5, 1)
    ax.imshow(importImage(path + photo_id))
    remove_ax_ticks(ax, s=2)

    ax = fig.add_subplot(1, 5, 2, projection='3d')
    plot3D(ax, pose_to_matrix(sets_3d[pose_id]))
    ax.set_title('Raw 3D Pose', fontsize=18, pad=25)
    remove_ax_ticks(ax, s=3)
    
    ax = fig.add_subplot(1, 5, 4, projection='3d')
    plot3D(ax, pose_to_matrix(sets_3d_cvi[pose_id][:-1]))
    ax.set_title('View-invariant 3D Pose', fontsize=18, pad=25)
    remove_ax_ticks(ax, s=3)
    
    ax = fig.add_subplot(1, 5, 3)
    plot2D(ax, pose_to_matrix(sets_3d[pose_id]))
    remove_labels(ax, s=2)
    remove_ax_ticks(ax, s=2)
    ax.set_title('Raw 2D Projection', fontsize=18)
    
    ax = fig.add_subplot(1, 5, 5)
    plot2D(ax, pose_to_matrix(sets_3d_cvi[pose_id][:-1]))
    remove_labels(ax, s=2)
    remove_ax_ticks(ax, s=2)
    ax.set_title('View-invariant 2D Projection', fontsize=18)
    plt.tight_layout()
    if show:
        plt.show()

def plot3D(ax, points, marker_size=100):
    ax.grid(False)
    oo = 1e10
    xmax,ymax,zmax = -oo,-oo,-oo
    xmin,ymin,zmin = oo, oo, oo

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
    
    for e in mpii_edges:
        ax.plot(x[e], y[e], z[e], c = c)
    
    max_range = np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xmax+xmin)
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ymax+ymin)
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zmax+zmin)
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

def plot2D(ax, pose_3d):
    '''
    2D plot of the 3D body pose in the x-y plane (ignoring z-axis)
    '''
    for e in range(len(mpii_edges)):
        ax.plot(pose_3d[mpii_edges[e]][:, 0], -1*pose_3d[mpii_edges[e]][:, 1], label=e)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # ax.legend(loc=1, bbox_to_anchor=(1, 0.38))

def clusterExamples(k, n_examples, path, model_clusters, pose_df, pose_arr, save, show=False):
    ax_array = np.linspace(1, k * 2 * n_examples - (k * 2 - 1), n_examples).astype(int)
    fig = plt.figure(figsize=(15, 15))

    for a in ax_array:
        addition = 0
        for cluster in range(k):
            arr_id = np.random.choice(np.where(model_clusters == cluster)[0])
            photo_id = ImageID(pose_df, arr_id)
            ax = fig.add_subplot(n_examples, k*2, a + addition)
            ax.imshow(importImage(path + photo_id))
            remove_ax_ticks(ax, s=2)
            addition += 1
            
            ax = fig.add_subplot(n_examples, k*2, a+addition)
            plot2D(ax, pose_to_matrix(pose_arr[arr_id][:-1]))
            remove_ax_ticks(ax, s=2)
            remove_labels(ax, s=2)
            
            if a == 1:
                ax.set_title('Cluster ' + str(cluster), position=(-0.1, 1), size=16)
            addition += 1

    plt.tight_layout()
    plt.savefig('viz/' + save + '.png')

    if show:
        plt.show()

def plotXSMap(xs_map, num_clusters, cluster_names, show=False):
    
    pitch = VerticalPitch(half=True, goal_type='box', pad_bottom=-30, 
                          pad_left=-10, pad_right=-10, line_color='black')
    
    fig, ax = pitch.draw(figsize=(10, 5), nrows=1, ncols=num_clusters, tight_layout=True)
    
    for i in range(num_clusters):
        im = ax[i].imshow(xs_map[i], cmap=plt.cm.Greens, interpolation='none', 
                       vmin=xs_map.min(), vmax=xs_map.max(), extent=[0,80,120,90])
        ax[i].set_title('Cluster ' + str(i) + ': ' + cluster_names[i])
    
    cax = plt.axes([1, 0.3, 0.05, 0.4])
    plt.colorbar(im, cax=cax)

    if show:
        plt.show()

def plotDoubleXSMap(xs_map, xs_map_up, cluster_names, num_clusters=4, show=False):

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

    if show:
        plt.show()

def plotBestTechniqueUp(xs_map, xs_map_up, cluster_name, show=False):
    '''
        Best technique to use
    '''

    pitch = VerticalPitch(half=True, goal_type='box', pad_bottom=-37, pad_left=-15, pad_right=-15, line_color='black', orientation='horizontal')
    fig, ax = pitch.draw(figsize=(10,5), nrows=1, ncols=2)
    
    cmap = plt.cm.tab20
    
    im = ax[0].imshow(np.argmax(xs_map, axis=0), cmap=cmap, interpolation='none', extent=[0,80,120,90])
    
    im = ax[1].imshow(np.argmax(xs_map_up, axis=0), cmap=cmap, interpolation='none', extent=[0,80,120,90])
    
    ax[0].set_title('Striker Not Under Pressure')
    ax[1].set_title('Striker Under Pressure')
    
    custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                    Line2D([0], [0], color=cmap(0.33), lw=4),
                    Line2D([0], [0], color=cmap(0.66), lw=4),
                    Line2D([0], [0], color=cmap(1.), lw=4)]
    
    ax[1].legend(custom_lines, cluster_name, loc=1, bbox_to_anchor=(1, 0.38))
    plt.tight_layout()
    if show:
        plt.show()

def plotTSNE(pose_tsne, kmeans_preds, cluster_name, number=4, show=False):
    '''
        Plots the TSNE result
    '''

    plt.figure(figsize=(11, 6))
    for i in range(number):
        current_pose_type = pose_tsne[kmeans_preds == i]
        colors_kmeans = cm.nipy_spectral(kmeans_preds[kmeans_preds==i].astype(float) / number)
        plt.scatter(current_pose_type[:,0], current_pose_type[:,1], c=colors_kmeans, label=cluster_name[i])

    plt.xlabel('t-SNE_1')
    plt.ylabel('t-SNE_2')
    plt.legend()

    if show:
        plt.show()

def plot_cluster(sets_3d_cvi_clean, set_3d_cvi_clean_df, closest, cluster_name, path='images/1v1_images/', show=False):

    '''
        Plot the most representative saves for each cluster
    '''
    fig = plt.figure(figsize=(20,7))
    for i in range(len(cluster_name)):
        photo_id = ImageID(set_3d_cvi_clean_df, closest[i])
        ax = fig.add_subplot(2, len(cluster_name), i+1)
        ax.imshow(importImage(path + photo_id))
        remove_ax_ticks(ax, s=2)
        ax.set_title('Cluster ' + str(i) + ': ' + cluster_name[i], size=20, pad=15)

        ax = fig.add_subplot(2, len(cluster_name), len(cluster_name)+1+i, projection='3d')
        plot3D(ax, pose_to_matrix(sets_3d_cvi_clean[closest[i]][:-1]))
        remove_ax_ticks(ax, s=3)
        remove_labels(ax, s=2)

    if show:
        plt.show()

def pose_overlay(ax, image, pose_2d):
    '''
        Plots skeleton over image
    '''
    ax.imshow(image)
    for e in range(len(mpii_edges)):
        ax.plot(pose_2d[mpii_edges[e]][:, 0], pose_2d[mpii_edges[e]][:, 1], c='cyan', lw=3, marker='o')
    remove_ax_ticks(ax, s=2)
    ax.axis('off')

def plot_pose_estimation(joined_pose_3d_df, pose_arr, pose_2d_arr, photo_id=315, show=False):
    '''
        Show image, image with 2D pose overlay, and 3D pose estimate
    '''

    array_id = auxi.getArrayID(joined_pose_3d_df, photo_id)
    image = importImage('images/pen_images/combined_data/' + str(photo_id)+'.png')
    pose_2d = pose_to_matrix(pose_2d_arr[array_id])
    points = pose_to_matrix(pose_arr[array_id])

    fig = plt.figure(figsize=(12, 4))
    
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(image)
    remove_ax_ticks(ax, s=2)
    ax.axis('off')
    ax.set_title('(a) Input Image', y=-0.14)

    ax = fig.add_subplot(1, 3, 2)
    pose_overlay(ax, image, pose_2d)
    ax.set_title('(b) 2D Pose Estimation', y=-0.14)

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    plot3D(ax, points)
    ax.set_title('(c) 3D Pose Estimation', y=-0.15)
    plt.tight_layout()

    if show:
        plt.show()

def plot_penalty_examples(pose_arr, joined_pose_3d_df, pic_ids, path='images/pen_images/combined_data/', show=False):

    def make_ax_image(ax, picture_id):
        ax.imshow(importImage(path + str(picture_id) + '.png'))
        remove_ax_ticks(ax, s=2)
        ax.axis('off')
    def make_ax_3d(ax, picture_id):
        plot3D(ax, pose_to_matrix(pose_arr[auxi.getArrayID(joined_pose_3d_df, picture_id)]))
        remove_ax_ticks(ax, s=3)
    
    fig = plt.figure(figsize=(15, 6))

    ax = fig.add_subplot(2, 6, 1)
    make_ax_image(ax, picture_id=pic_ids[0])
    ax = fig.add_subplot(2, 6, 2, projection='3d')
    make_ax_3d(ax, picture_id=pic_ids[0])

    ax = fig.add_subplot(2, 6, 3)
    make_ax_image(ax, picture_id=pic_ids[1])
    ax = fig.add_subplot(2, 6, 4, projection='3d')
    make_ax_3d(ax, picture_id=pic_ids[1])

    ax = fig.add_subplot(2, 6, 5)
    make_ax_image(ax, picture_id=pic_ids[2])
    ax = fig.add_subplot(2, 6, 6, projection='3d')
    make_ax_3d(ax, picture_id=pic_ids[2])


    ax = fig.add_subplot(2, 6, 7)
    make_ax_image(ax, picture_id=pic_ids[3])
    ax = fig.add_subplot(2, 6, 8, projection='3d')
    make_ax_3d(ax, picture_id=pic_ids[3])

    ax = fig.add_subplot(2, 6, 9)
    make_ax_image(ax, picture_id=pic_ids[4])
    ax = fig.add_subplot(2, 6, 10, projection='3d')
    make_ax_3d(ax, picture_id=pic_ids[4])

    ax = fig.add_subplot(2, 6, 11)
    make_ax_image(ax, picture_id=pic_ids[5])
    ax = fig.add_subplot(2, 6, 12, projection='3d')
    make_ax_3d(ax, picture_id=pic_ids[5])

    plt.tight_layout()
    if show:
        plt.show()

def penalty_clusterExamples(good_poses_3d_arr, good_poses_3d_df, kmeans_pens_preds, ax_array, path='images/pen_images/combined_data/', show=False):
    '''
        GMM - 3D pose, 2D pose viz cluster examples
    '''
    
    fig = plt.figure(figsize=(15, 15))
    for a in ax_array:
        arr_id = np.random.choice(np.where(kmeans_pens_preds == 0)[0])
        photo_id = auxi.getImageID(good_poses_3d_df, arr_id)
        ax = fig.add_subplot(5, 4, a)
        ax.imshow(importImage(path + str(photo_id) + '.png'))
        remove_ax_ticks(ax, s=2)

        ax = fig.add_subplot(5, 4, a+1)
        plot2D(ax, pose_to_matrix(good_poses_3d_arr[arr_id]))
        remove_ax_ticks(ax, s=2)
        remove_labels(ax, s=2)
        if a == 1:
            ax.set_title('Cluster ' + str(0), position=(-0.1, 1), size=16)
            
        arr_id = np.random.choice(np.where(kmeans_pens_preds == 1)[0])
        photo_id = auxi.getImageID(good_poses_3d_df, arr_id)
        
        ax = fig.add_subplot(5, 4, a+2)
        ax.imshow(importImage(path + str(photo_id) + '.png'))
        remove_ax_ticks(ax, s=2)
        
        ax = fig.add_subplot(5, 4, a+3)
        plot2D(ax, pose_to_matrix(good_poses_3d_arr[arr_id]))
        remove_ax_ticks(ax, s=2)
        remove_labels(ax, s=2)
        if a == 1:
            ax.set_title('Cluster ' + str(1), position=(-0.1, 1), size=16)
        
    plt.tight_layout()

    if show:
        plt.show()
    return



# goal plots
def plot_goal_zones(start=1, show=False):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = [start,start, start+largura, start+largura, ]
    y = [start,start+altura,start+altura, start, ]
    
    # Create posts
    traves = Line2D(x, y, linewidth=5)
    ax.add_line(traves)

    for position, s in positions.items():
      if s:
        # print(position)
        r = Rectangle(s[0], s[1], s[2], edgecolor = s[3],
                  facecolor = s[3], fill=True, alpha=0.3, lw=1, label=position)
        ax.add_patch(r)

    plt.xlim([-32, 764])
    plt.ylim([-32, 276])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(bbox_to_anchor=(1.1, 1.05))
    
    if show:
        plt.show()
    return plt



def plot_straight_hull_rec(points, bbox):
  '''
    Plots a straight rectangle that covers all points
    bbox is that rec. Call auxi.straight_bounding_rectangle(points) to retreive it 
  '''

  plt.scatter(points[:,0], points[:,1])
  plt.fill(bbox[:,0], bbox[:,1], alpha=0.2)
  plt.axis('equal')
  plt.show()


if __name__ == '__main__':

    print("main")
    df = pd.read_csv("data/events/prem_pens_all.csv")
    print("Total pen:", len(df))
    df['count'] = 0
    df = df.loc[df.outcome != 'Off T'][['outcome', 'Direction', 'count']]
    df = df.groupby(by=['outcome', 'Direction']).count().reset_index(col_level=1)


    goal = df.loc[df.outcome == "Goal"][['Direction', 'count']]
    saved = df.loc[df.outcome == "Saved"][['Direction', 'count']]

    print("Total Saved: ",  saved['count'].sum())
    print("Total Goal: ", goal['count'].sum()) 

    # print(df[['outcome', 'Direction']].value_counts())
    # print(goal.to_dict('index'))

    goal_d = goal.set_index('Direction').to_dict("index")
    saved_d = saved.set_index('Direction').to_dict("index")

    print(goal_d)
    print(saved_d)