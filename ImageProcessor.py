import numpy as np
import cv2
from numpy.core.fromnumeric import ptp
from numpy.lib.function_base import append
import pandas as pd
import laspy


def pointcloud(depth, fov):
    matrix = np.loadtxt('model.txt', usecols=range(3))
    matrix2=matrix.T
    print(matrix2[1])

    fy =  0.5 / np.tan(fov * 0.5) 
   
    height = depth.shape[0]

    width = depth.shape[1]
    print("Height: ",height)
    print("Width: ", width)

    aspectratio=width/height 

    fx=fy/aspectratio

    print("fy: ", fy)
    print("fx: ", fx)

    mask = np.where(depth > 0)
    

    x = mask[1]

    y = mask[0]

    normalized_x = (x.astype(np.float32) - width * 0.5) / width
    

    normalized_y = (y.astype(np.float32) - height * 0.5) / height
    
    world_x = normalized_x * depth[y, x] / fx
    

    world_y = -normalized_y * depth[y, x] / fy
    
    world_z = depth[y, x]
    ones = np.ones(world_z.shape[0], dtype=np.float32)
    
    return np.vstack((world_x, world_y, world_z)).T


def pointCloudCalculator(depth, fov):
    sensorModel = np.loadtxt('model.txt', usecols=range(3))    

    mask = np.where(depth > 0 )
    width = depth.shape[1]
    height = depth.shape[0]
    
    #f=0.123
    fy =  0.5 / np.tan(fov * 0.5) 
    aspectratio=width/height
    fx=fy/aspectratio
    u_0=width/2
    v_0=height/2
    gamma=0

    print("fy: ", fy)
    print("fx: ", fx)
    print("Height: ",height)
    print("Width: ", width)
    u=mask[1]
    v=mask[0]  
    listof1s = [1] * len(u)
    listof0s = [0] * len(u)

    K=np.array([[fx,gamma,u_0],
        [0,fy,v_0],
        [0,0,1]])    
    K_inv=np.linalg.inv(K)
    
    
    cord=np.array([u,v,listof1s])
    out=depth[v,u]*np.matmul(K_inv,cord)
    
    x=out[0] / width
    y=-out[1] / height  
    z=out[2]
    out2=np.array([[x],[y],[z],[listof1s]])
    print(max(x))
    theta=-35
    camCord=np.vstack((x, y, z))
    pixelOrigin=int(len(camCord[0])/2)
    camCordWorldOrigin=np.array([camCord[0][pixelOrigin],camCord[1][pixelOrigin],camCord[2][pixelOrigin]])
    print(camCordWorldOrigin)
    #rotMatrix=np.array([[1,0,0],[0,np.cos(0.34),-np.sin(0.34)],[0,np.sin(0.34),np.cos(0.34)]])
    extMatrix=np.array([[1,0,0,0],[0,np.cos(theta*np.pi / 180),-np.sin(theta*np.pi / 180),0],[0,np.sin(theta*np.pi / 180),np.cos(theta*np.pi / 180),0],[0,0,0,1]])
    print(extMatrix)
    extMatrix_inv=np.linalg.inv(extMatrix)
    print(extMatrix_inv)
    ##camCord[0][pixelOrigin]
    #camCord[1][pixelOrigin]
    #camCord[2][pixelOrigin]
    
    worldCord=np.matmul(out2.T,extMatrix_inv)
    print(worldCord.T)

    x_world=(worldCord.T[0])#+ (ptp(x)/4))
    y_world=(worldCord.T[1])# + (ptp(y)/2))
    z_world=(worldCord.T[2] - (ptp(z)/2))#camCordWorldOrigin[2])
    #world=worldCord.T[3]

    return np.vstack((x_world,y_world,z_world)).T
    #return np.vstack((x, y, z)).T
    

    

        
            
            


   
    


