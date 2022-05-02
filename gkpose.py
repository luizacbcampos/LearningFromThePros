#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 17:29:44 2021

@author: mattwear
"""

### Pose Analysis Functions

import re
import json
import math
import numpy as np
import pandas as pd


from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import cv2

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import LocallyLinearEmbedding, TSNE
from sklearn.decomposition import PCA

from plots import *

mpii_edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], 
              [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], 
              [6, 8], [8, 9]]

# --- PLOT ---
   
        
def pose_to_matrix(pose):
    if len(pose) == 48:
        pose_matrix = pose.reshape(16, 3)
    else:
        pose_matrix = pose.reshape(16, 2)
    return pose_matrix

def importImage(img):
    #Import image
    image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    return image

def rotatePose(pose_3d, theta):
    #Rotate body pose by theta degrees around the y axis
    #Input: pose_3d - 16x3 array representing the coordinates of the body pose
    #Returns: 16x3 array of rotated body pose coordinates
    radian = math.radians(theta)

    rotation_matrix = np.array([[np.cos(radian), 0, np.sin(radian)],
                                [0, 1, 0],
                                [-np.sin(radian), 0, np.cos(radian)]])

    rotated_pose = np.zeros((len(pose_3d), 3))
    for i in range(len(pose_3d)):
        rotated_pose[i] = rotation_matrix @ pose_3d[i]
    return rotated_pose

def hipWidth(pose_3d):
    #Input: pose_3d - 16x3 np array representing a single 3D body pose
    #Returns euclidean distance in x-y space of the two hip joints
    #Indices of both hip locations are 2 and 3.
    return np.linalg.norm(pose_3d[3][:2]-pose_3d[2][:2])

def cameraInvariantPose(pose_3d):
    # Function to get the optimal rotated pose
    best_pose = pose_3d
    max_hip_width = hipWidth(pose_3d)
    theta_ranges = list(range(10, 100, 10)) + list(range(270, 360, 10))

    for theta in theta_ranges:
        rotated_pose = rotatePose(pose_3d, theta=theta)
        hip_width = hipWidth(rotated_pose)
        if hip_width > max_hip_width:
            best_pose = rotated_pose
            max_hip_width = hip_width
    return best_pose

def flipBehindPoses(cvi_arr):
    #Rotates the poses that are photographed from behind 180 degrees
    sets_3d_cvi = np.zeros(cvi_arr.shape)
    for i in range(len(sets_3d_cvi)):
        #Rotate the photos from behind by 180 degrees
        pose = pose_to_matrix(cvi_arr[i])
        if pose[10][0] > pose[15][0]: #if RHx > LHx
            #print("This is a photo from behind")
            #Rotate this pose 180 degrees
            new_pose = rotatePose(pose, 180).flatten()
        else:
            #print("This is a photo from in front")
            new_pose = pose.flatten()
        sets_3d_cvi[i] = new_pose
    return sets_3d_cvi

def cameraInvariantDataset(raw_poses):
    #Converts the raw body point dataset to a cleaned camera-invariant one
    cleaned_pose_arr = raw_poses.copy()
    for i in range(len(raw_poses)):
        pose_3d = pose_to_matrix(raw_poses[i])
        best_pose = cameraInvariantPose(pose_3d)
        cleaned_pose_arr[i] = best_pose.flatten()
    return cleaned_pose_arr

def getFreezeFrame(shots, shot_id):
    onevone = shots.copy()
    ### Plot a shooting Situation
    #onevone['shot_freeze_frame'][shot_id][0]['position']['name']
    shooter_x = onevone['location'][shot_id][0]
    shooter_y = onevone['location'][shot_id][1]

    num_players = len(onevone['shot_freeze_frame'][shot_id])
    is_gk = np.zeros(num_players)
    is_teammate = np.zeros(num_players)
    freeze_frame_x = np.zeros(num_players)
    freeze_frame_y = np.zeros(num_players)
    for i in range(num_players):
        freeze_frame_x[i] = onevone['shot_freeze_frame'][shot_id][i]['location'][0]
        freeze_frame_y[i] = onevone['shot_freeze_frame'][shot_id][i]['location'][1]
        is_gk[i] = onevone['shot_freeze_frame'][shot_id][i]['position']['name'] == 'Goalkeeper'
        is_teammate[i] = onevone['shot_freeze_frame'][shot_id][i]['teammate']

    attacking_team_x = freeze_frame_x[is_teammate.astype(bool)]
    attacking_team_y = freeze_frame_y[is_teammate.astype(bool)]
    defending_team_x = freeze_frame_x[~ is_teammate.astype(bool)]
    defending_team_y = freeze_frame_y[~ is_teammate.astype(bool)]
    gk_x = freeze_frame_x[is_gk.astype(bool)]
    gk_y = freeze_frame_y[is_gk.astype(bool)]
    return shooter_x,shooter_y,attacking_team_x,attacking_team_y,defending_team_x,defending_team_y,gk_x,gk_y,is_gk

def distance_to_goal(shooter_x, shooter_y, goal_x = 120, goal_y=40):
    return np.linalg.norm(np.array([shooter_x,shooter_y])-np.array([goal_x,goal_y]))

def goal_angle(shooter_x, shooter_y, goal_x = 120, goal_y=40):
    return math.degrees(math.atan(np.abs(goal_y-shooter_y) / np.abs(goal_x - shooter_x)))

def getPhotoID(df):
    #Extract photo_id
    photo_id = []
    for i in range(len(df)):
        photo_id.append(int(re.findall(r"(\d+).", df['file'][i])[0]))
    df['photo_id'] = photo_id
    return df

def ImageID(df, array_id):
    #Get photo id's of poses
    return df.loc[array_id, 'file']

def torsoAngle(pose_3d):
    #Torso Angle
    torso_angle = math.atan2(pose_3d[7][0], -pose_3d[7][1])*180/math.pi
    return np.abs(torso_angle)

def bodyHeight(pose_3d):
    #Body Height
    height = np.abs(np.max(-pose_3d[:, 1]) - np.min(-pose_3d[:, 1]))
    return height

def forwardStep(pose_3d):
    #Distance of forward step
    forward_step = np.abs(pose_3d[0][2] - pose_3d[5][2])
    return forward_step

def handHeight(pose_3d):
    hand_height = np.abs(np.min(-pose_3d[:, 1]) - np.min(-pose_3d[[10, 15]][:, 1]))
    return hand_height

def bodyAngle(pose_3d):
    midpoint = (pose_3d[0][:2] + pose_3d[5][:2])/2
    midpoint[1] *= -1
    torso = pose_3d[7][:2] * np.array([1, -1])
    body_angle_vec = torso-midpoint
    return np.abs(math.atan2(body_angle_vec[0], body_angle_vec[1])*180/math.pi)

def cleanPredictions(set_3d_cvi_df):
    #List of the array_ids in which to remove because they are bad prediction of true pose
    to_remove_sets = np.array([1,6,7,11,14,25,27,28,31,32,37,40,42,43,44,51,52,53,55,58,
                               63,65,72,81,83,85,87,94,96,108,109,110,113,114,116,117,119,
                               123,131,133,135,136,137,140,141,143,144,147,150,151,154,156,
                               157,159,160,161,163,167,170,176,189,193,195,196,198,200,
                               202,203,206,207,210,211,213,216,217,218,220,227,228,235,
                               237,238,242,243,244,245,250,251,252,255,261,262,267,268,
                               270,271,274,275,276,282,287,291,296,297,298,304,305,311,312,
                               316,320,323,324,326,327,328,333,334,335,341,350,351,352,370,
                               372,374,379,387,388,389,390,395,397,401,406,411,413,414,418,
                               419,423,433,436,439,443,446,451,452,453,456,462,465,470,472,
                               474,475,480,489,490,494,502,507,509,515,517,522,528,532,533,
                               537,553,555,558,566,567,570,572,575,579,580,585])
    #Remove selected poses
    set_3d_cvi_clean_df = set_3d_cvi_df.drop(to_remove_sets).reset_index(drop=True)
    keep_cols = np.array(list(range(48)) + ['gk_engage'])
    sets_3d_cvi_clean = set_3d_cvi_clean_df.loc[:,keep_cols].values
    return sets_3d_cvi_clean, set_3d_cvi_clean_df

def saveClusters(clusters, file_name, path='data/'):
    pd.DataFrame(clusters, columns=['cluster']).to_csv(path + file_name, index=False)

def getTrainTest(df, test_size=0.3):
    on_target = (df['shot_outcome_name'] == 'Goal') | (df['shot_outcome_name'] == 'Saved')
    features = ['photo_id','gk_name','shot_outcome_name','cluster','shot_angle','distance_to_goal',
                'under_pressure']
    ml_df = df.loc[on_target, features].copy()
    ml_df['shot_outcome_name'].replace({'Goal': 0, 'Saved': 1}, inplace=True)
    ml_df = pd.get_dummies(ml_df, columns=['cluster'])
    ml_df = ml_df.reset_index(drop=True)
    test_ind = np.random.choice(range(ml_df.shape[0]), int(ml_df.shape[0] * test_size))
    print(test_ind)
    test_df = ml_df.loc[test_ind, :].copy()
    train_df = ml_df.drop(test_ind).reset_index(drop=True)
    return train_df, test_df

def getxSInput(df, scaler, angle, dist, up=0, cluster=0):
    k = len(df.filter(regex='cluster').columns)
    angle_dist = scaler.transform([[angle,dist]])[0]
    up = np.array([up])
    clust = np.zeros(k)
    clust[cluster] = 1
    #if assist_type == 'Cross':
    #    ass_t = np.array([1, 0, 0])
    #elif assist_type == 'Other':
    #    ass_t = np.array([0, 1, 0])
    #else:
    #    ass_t = np.array([0, 0, 1])
    return np.array([np.concatenate((angle_dist,up,clust))])

def getXSMap(train_df, model, scaler, num_clusters, up=0, ass='Pass'):
    #Sets: Probability Map
    x_range = np.linspace(90, 120.01, 50)
    y_range = np.linspace(0, 80, 50)
    xs_map = np.zeros((num_clusters, len(x_range), len(y_range)))
    for cluster in range(num_clusters):
        for x in range(len(x_range)):
            for y in range(len(y_range)):
                d = distance_to_goal(shooter_x=x_range[x], shooter_y=y_range[y])
                a = goal_angle(shooter_x=x_range[x], shooter_y=y_range[y])
                xs = []
                for n in range(num_clusters):
                    inp = getxSInput(train_df,scaler,angle=a,dist=d,up=up,cluster=n)
                    xs.append(model.predict_proba(inp)[0][1])
                mean_xs = np.mean(xs)
                inp = getxSInput(train_df,scaler,angle=a,dist=d,up=up,cluster=cluster)
                xs_map[cluster][x, y] = model.predict_proba(inp)[0][1] - mean_xs
        print("done cluster", cluster)
    return xs_map

def getGKEM(amateur_1v1s):
    dist_to_goal = []
    striker_to_gk = []
    goal_angle = []
    for i in range(len(amateur_1v1s)):
        dist_to_goal.append(distance_to_goal(amateur_1v1s['striker_x'][i], amateur_1v1s['striker_y'][i]))
        striker_to_gk.append(distance_to_goal(amateur_1v1s['striker_x'][i], amateur_1v1s['striker_y'][i], amateur_1v1s['gk_x'][i], amateur_1v1s['gk_y'][i]))
        goal_angle.append(goal_angle(amateur_1v1s['striker_x'][i], amateur_1v1s['striker_y'][i]))
    amateur_1v1s['gkem'] = np.array(striker_to_gk) / np.array(dist_to_goal)
    amateur_1v1s['distance_to_goal'] = dist_to_goal
    amateur_1v1s['goal_angle'] = goal_angle
    return amateur_1v1s

def getOptimalSaveTechnique(amateur_1v1s, amateur_model_df, svm, scaler, num_clusters=4):
    optimal_cluster = []
    mean_xs = []
    for i in range(len(amateur_1v1s)):
        angle = amateur_1v1s.loc[i, 'goal_angle']
        dist = amateur_1v1s.loc[i, 'distance_to_goal']
        up = amateur_1v1s.loc[i, 'under_pressure']
        xs_list = []
        for cluster in range(num_clusters):
            inp = getxSInput(amateur_model_df, scaler, angle=angle, dist=dist, up=up, cluster=cluster)
            xs_list.append(svm.predict_proba(inp)[0][1])
        optimal_cluster.append(np.argmax(xs_list))
        mean_xs.append(np.mean(xs_list))
    return optimal_cluster, mean_xs


# --- PENALTIES ---

def cleanPenDataFrames(pose_3d_df, pose_3d_2_df):
    #pose_3d_2_df: 17 - 19 data
    #pose_3d_df: 19 - 21 data
    #Change shots that hit post to 'Off T'
    pose_3d_2_df.loc[pose_3d_2_df.shot_outcome_name == 'Post', 'shot_outcome_name'] = 'Off T'
    pose_3d_df.loc[pose_3d_df.off_target == 1, 'outcome'] = 'Off T'
    pose_3d_df.loc[pose_3d_df.outcome == 'Scored', 'outcome'] = 'Goal'
    pose_3d_df.loc[pose_3d_df.outcome == 'Missed', 'outcome'] = 'Saved'
    pose_3d_df.drop(columns=['url','off_target'], inplace=True)
    reorder = ['pen_taker','outcome','goalkeepers'] + list(map(str, list(range(int(pose_3d_df.columns[-1]) + 1))))
    pose_3d_df = pose_3d_df[reorder]
    pose_3d_2_df.rename(columns={"player_name": "pen_taker", 
                                 "shot_outcome_name": "outcome",
                                 "gk_name": "goalkeepers"}, inplace=True)
    joined_pose_3d_df = pose_3d_2_df.append(pose_3d_df, ignore_index=True) #contains all pens
    joined_pose_3d_df.dropna(inplace=True)
    pose_arr = joined_pose_3d_df.loc[:,'0':].values
    return (joined_pose_3d_df, pose_arr)

def getArrayID(pose_df, photo_id):
    return np.where(np.array(pose_df.index) == photo_id)[0][0]

def getImageID(pose_df, array_id):
    #Input: pose_df - dataframe with raw pose information - index matches to photo name
    #Input: array_id - location of pose in array
    #Returns: photo name/id
    return np.array(pose_df.index)[array_id]

def cleanPenPredictions(joined_pose_3d_df):
    to_remove = np.array([0,5,7,9,10,11,15,17,19,23,25,26,27,30,31,36,40,48,52,53,56,57,58,59,62,65,66,
                 68,69,75,77,79,80,84,86,87,89,92,93,96,97,99,100,101,102,103,108,111,113,116,
                 119,121,125,126,135,138,140,141,143,145,146,147,152,155,159,165,166,169,173,
                 177,178,179,186,187,188,190,191,192,194,203,207,211,213,220,221,222,224,227,
                 235,240,241,242,248,249,257,260,261,262,268,270,271,272,275,276,277,278,279,
                 281,282,291,294,299,301,305,312,314,317,322,324,331,333,335,336,341,347,349,351,
                 352,362,368,376,382,387,392,394,395])
    good_poses_3d_df = joined_pose_3d_df.drop(to_remove)
    return good_poses_3d_df 

def PenFeatureSpace(clean_poses):
    #Input: clean_poses - dataset off all of the camera-invariant poses
    #Returns: dataset of poses in feature space
    pose_features = np.zeros((len(clean_poses), 5))
    for i in range(len(clean_poses)):
        pose_3d = pose_to_matrix(clean_poses[i])
        feature_array = np.array([torsoAngle(pose_3d), bodyHeight(pose_3d), 
                                  forwardStep(pose_3d), handHeight(pose_3d),
                                  bodyAngle(pose_3d)])
        pose_features[i] = feature_array
    return pose_features

def getxSInputNoScaler(df, angle, dist, up=0, cluster=0):
    k = len(df.filter(regex='cluster').columns)
    angle_dist = np.array([angle,dist])
    up = np.array([up])
    clust = np.zeros(k)
    clust[cluster] = 1
    return np.array([np.concatenate((angle_dist,up,clust))])

def getProsOptimalCluster(df, svm, num_clusters=4):
    optimal_cluster = []
    for i in range(len(df)):
        angle = df.loc[i, 'shot_angle']
        dist = df.loc[i, 'distance_to_goal']
        up = df.loc[i, 'under_pressure']
        xs_list = []
        for cluster in range(num_clusters):
            inp = getxSInputNoScaler(df, angle=angle, dist=dist, up=up, cluster=cluster)
            xs_list.append(svm.predict_proba(inp)[0][1])
        optimal_cluster.append(np.argmax(xs_list))
    return optimal_cluster