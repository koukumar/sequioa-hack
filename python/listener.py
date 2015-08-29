import socket
from socketIO_client import SocketIO, LoggingNamespace
import numpy as np
from sklearn import linear_model
import math
import thread
import time
import pandas as pd
import pickle
from sklearn import preprocessing

import os

def netcat():
    global point
    print "starting"
    headersTxt = "loggingTime,loggingSample,locationHeadingTimestamp_since1970,locationHeadingX,locationHeadingY,locationHeadingZ,locationTrueHeading,locationMagneticHeading,locationHeadingAccuracy,accelerometerTimestamp_sinceReboot,accelerometerAccelerationX,accelerometerAccelerationY,accelerometerAccelerationZ,gyroTimestamp_sinceReboot,gyroRotationX,gyroRotationY,gyroRotationZ,motionTimestamp_sinceReboot,motionYaw,motionRoll,motionPitch,motionRotationRateX,motionRotationRateY,motionRotationRateZ,motionUserAccelerationX,motionUserAccelerationY,motionUserAccelerationZ,motionAttitudeReferenceFrame,motionQuaternionX,motionQuaternionY,motionQuaternionZ,motionQuaternionW,motionGravityX,motionGravityY,motionGravityZ,motionMagneticFieldX,motionMagneticFieldY,motionMagneticFieldZ,motionMagneticFieldCalibrationAccuracy,state"
    headers = headersTxt.split(",")
    hostname = "192.168.43.80"
    if "hostname" in os.environ:
        hostname = os.environ["hostname"]
    print hostname
    port = 58155
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((hostname, port))
    feat = []
    dff = pd.DataFrame()
    sensor_freq = 20
    window_size = 3 * sensor_freq
    while 1:
    #with SocketIO('127.0.0.1', 3000, LoggingNamespace) as socketIO:
        while 1:
	    #print "loping"
            data = s.recv(8024)
            if data == "" or not data.startswith("2015-08-"):
                continue
            line = data.strip()
            vals = line.split(",")
            sensor_dat = {}
            for header, val in zip(headers, vals):
                try:
                    sensor_dat[header] = float(val)
                except (TypeError, ValueError), te:
                    sensor_dat[header] = val
            point = sensor_dat
            feat.append(point)
            socket_msg = {"rx": point["motionRoll"],
                          "ry": point["motionPitch"],
                          "rz": point["motionYaw"]}
            #socketIO.emit("motion", socket_msg)
            if (len(feat) > window_size):
                dff = dff.append(feat)
                num = 0
                for item in feat:
                    feat.remove(item)
                    num+=1
                    if num >= sensor_freq:
                        break
                print predict(extract_features([dff]))

    print "Connection closed."
    s.close()

def extract_features(df_list):
    features_names = ["avg_acc", "max_acc", "min_acc", "avg_gyro", "max_gyro", "min_gyro", "y"]
    f_map = {}
    for fname in features_names:
        f_map[fname] = []
        
    for df in df_list:
        ndf = df[["state"]]
        ndf.loc[:,"acc"] = (df[["accelerometerAccelerationX", "accelerometerAccelerationY", "accelerometerAccelerationZ"]]**2).sum(axis=1)
        ndf.loc[:,"gyro"] = (df[["gyroRotationX", "gyroRotationY", "gyroRotationZ"]]**2).sum(axis=1)    
        agg = ndf.mean()
        if np.isnan(agg["acc"]):
            continue
        f_map["avg_acc"].append(agg["acc"])
        f_map["avg_gyro"].append(agg["gyro"])

        agg = ndf.max()
        f_map["max_acc"].append(agg["acc"])   
        f_map["max_gyro"].append(agg["gyro"])
        f_map["y"].append(agg["state"]+0.1)

        agg = ndf.min()
        f_map["min_acc"].append(agg["acc"])    
        f_map["min_gyro"].append(agg["gyro"])
    return pd.DataFrame(data=f_map)

def predict(df):
    label_names = {"0.1": "Idle", "1.1":"Moving", "2.1":"Fall", "3.1": "Agitated", "4.1": "Normal"}
    X = df[["avg_acc", "max_acc", "min_acc", "avg_gyro", "max_gyro", "min_gyro"]].values
    model = pickle.load(open("model.pickle", 'r'))
    scaler = pickle.load(open("scaler.pickle", 'r'))
    X_scaled = scaler.fit_transform(X)
    y = model.predict(X_scaled)
    print y
    return label_names[str(y[0])]

if __name__ == "__main__":
    netcat()
