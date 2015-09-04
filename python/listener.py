import socket
from socketIO_client import SocketIO, LoggingNamespace
import numpy as np
from sklearn import linear_model
import math
import thread
import time
import pandas as pd
import pickle, sys, traceback
from sklearn import preprocessing

import os

def netcat():
    global point
    print "starting"
    #headersTxt = "loggingTime,loggingSample,locationHeadingTimestamp_since1970,locationHeadingX,locationHeadingY,locationHeadingZ,locationTrueHeading,locationMagneticHeading,locationHeadingAccuracy,accelerometerTimestamp_sinceReboot,accelerometerAccelerationX,accelerometerAccelerationY,accelerometerAccelerationZ,gyroTimestamp_sinceReboot,gyroRotationX,gyroRotationY,gyroRotationZ,motionTimestamp_sinceReboot,motionYaw,motionRoll,motionPitch,motionRotationRateX,motionRotationRateY,motionRotationRateZ,motionUserAccelerationX,motionUserAccelerationY,motionUserAccelerationZ,motionAttitudeReferenceFrame,motionQuaternionX,motionQuaternionY,motionQuaternionZ,motionQuaternionW,motionGravityX,motionGravityY,motionGravityZ,motionMagneticFieldX,motionMagneticFieldY,motionMagneticFieldZ,motionMagneticFieldCalibrationAccuracy,state"
    headersTxt = "loggingTime,loggingSample,identifierForVendor,accelerometerTimestamp_sinceReboot,accelerometerAccelerationX,accelerometerAccelerationY,accelerometerAccelerationZ,gyroTimestamp_sinceReboot,gyroRotationX,gyroRotationY,gyroRotationZ,motionTimestamp_sinceReboot,motionYaw,motionRoll,motionPitch,motionRotationRateX,motionRotationRateY,motionRotationRateZ,motionUserAccelerationX,motionUserAccelerationY,motionUserAccelerationZ,motionAttitudeReferenceFrame,motionQuaternionX,motionQuaternionY,motionQuaternionZ,motionQuaternionW,motionGravityX,motionGravityY,motionGravityZ,motionMagneticFieldX,motionMagneticFieldY,motionMagneticFieldZ,motionMagneticFieldCalibrationAccuracy,IP_en0,IP_pdp_ip0,state"
    headers = headersTxt.split(",")
    hostname = "192.168.43.34"
    #hostname = ""
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
    #while 1:
    with SocketIO('127.0.0.1', 3000, LoggingNamespace) as socketIO:
        while 1:
	    #print "loping"
            data = s.recv(8024)
            if data == "" or not data.startswith("2015-08-"):
                continue
            #print data
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
            try:
                socket_msg = {"rx": point["motionRoll"],
                              "ry": point["motionPitch"],
                              "rz": point["motionYaw"]}
            except:
                continue
            #socketIO.emit("motion", socket_msg)
            #print point["accelerometerAccelerationZ"]
            if (len(feat) > window_size):
                dff = pd.DataFrame(feat)
                num = 0
                for item in feat:
                    feat.remove(item)
                    num+=1
                    if num >= sensor_freq:
                        print item["loggingTime"]
                        break
                try:
                    dff = normalize(dff)
                    features = extract_features([dff])
                    max_acc = features["max_acc"].values[0]
                    predict(features,socketIO, max_acc)
                except:
                    print sys.exc_info()[0]
                    traceback.print_exc(file=sys.stdout)
                    continue

    print "Connection closed."
    s.close()

def extract_features(df_list):
    features_names = ["avg_acc", "max_acc", "min_acc", "avg_gyro", "max_gyro", "min_gyro", "y"]
    f_map = {}
    for fname in features_names:
        f_map[fname] = []
        
    for df in df_list:
        ndf = df[["state"]]
        #df_acc = df[["accelerometerAccelerationX", "accelerometerAccelerationY", "accelerometerAccelerationZ"]]
        df_acc = df[["accx", "accy", "accz"]]
        ndf.loc[:,"acc"] = pd.Series(data=(df_acc**2).sum(axis=1), index=df.index)
        ndf.loc[:,"gyro"] = (df[["gyx", "gyy", "gyz"]]**2).sum(axis=1)    
        agg = ndf.mean()
        if np.isnan(agg["acc"]):
            continue
        f_map["avg_acc"].append(agg["acc"])
        f_map["avg_gyro"].append(agg["gyro"])

        agg = ndf.max()
        f_map["max_acc"].append(agg["acc"])   
        f_map["max_gyro"].append(agg["gyro"])
        f_map["y"].append(agg["state"])

        agg = ndf.min()
        f_map["min_acc"].append(agg["acc"])    
        f_map["min_gyro"].append(agg["gyro"])
    return pd.DataFrame(data=f_map)

model = pickle.load(open("model.pickle", 'r'))
scaler = pickle.load(open("scaler.pickle", 'r'))
admodel = pickle.load(open("ad.pickle", 'r'))
label_list = []
label_list_limit = 15
previous_label = "Idle"

def predict(df, socketIO, max_acc):
    global previous_label
    label_names = {"0.1": "Idle", "1.1":"Moving", "2.1":"Fall", "3.1": "Agitated", "4.1": "Normal"}
    cols = [x for x in df.columns.get_values() if x!="y"]
    X = df[cols].values
    X_scaled = scaler.transform(X)
    y = model.predict(X_scaled)
    probs = model.predict_log_proba(X_scaled)[0]
    ads = admodel.decision_function(X_scaled).ravel()
    fall = probs[2]
    walk = probs[1]
    Idle = probs[0]
    anomaly = ads[0]
    new_label = "NA"

    if (probs[2] > -5 and max_acc > 100):
        label = "Fall"
    #elif ads[0] < -5:
    elif probs[1] < -40:
        label = "Agitated"
    elif probs[1] > -1:
        label = "Moving"
    else:
        label = "Idle"
    print "State: " + label
    print "Prob: " + str(model.predict_log_proba(X_scaled))
    print "Ad: " + str(admodel.decision_function(X_scaled).ravel())

    #label = label_names[str(y[0])]
    label_list.append(label)
    #print len(label_list)
    if (len(label_list) >= label_list_limit):
        for item in label_list:
            label_list.remove(item)
            break
    print label_list, len(label_list), label_list.count("Idle")

    if ((label_list[-1] != "Fall" and label_list[-1] != "Agitated") and 
        (label_list[-2] != "Fall" and label_list[-2] != "Agitated") and
        (label_list[-3] != "Fall" and label_list[-3] != "Agitated") and
        (label_list.count("Fall") + label_list.count("Agitated") < 7) and
        (label_list.count("Fall") + label_list.count("Agitated") > 1)):
        new_label = "Fall"
    elif (label_list.count("Fall") + label_list.count("Agitated") > 9):
        new_label = "Agitated"
    else:
        if (label == "Fall" or label == "Agitated"):
            new_label = "Moving"
        else:
            new_label = label

    socketIO.emit("motion", {"state" : new_label})
    previous_label = new_label
    print new_label, label

    return new_label

def normalize(df):
    df_acc = df[["motionYaw","motionRoll","motionPitch","accelerometerAccelerationX", 
                 "accelerometerAccelerationY", "accelerometerAccelerationZ",
                "gyroRotationX", "gyroRotationY", "gyroRotationZ"]]
    accx = []
    accy = []
    accz = []
    gyx = []
    gyy = []
    gyz = []
    for row in df_acc.values:
        new_point = rotate_xyz(row[:3], row[3:6])
        accx.append(new_point[0])
        accy.append(new_point[1])
        accz.append(new_point[2])
        new_point = rotate_xyz(row[:3], row[6:])
        gyx.append(new_point[0])
        gyy.append(new_point[1])
        gyz.append(new_point[2])
    df.loc[:,"accx"] = pd.Series(data=accx, index=df.index)
    df.loc[:,"accy"] = pd.Series(data=accy, index=df.index)
    df.loc[:,"accz"] = pd.Series(data=accz, index=df.index)
    df.loc[:,"gyx"] = pd.Series(data=gyx, index=df.index)
    df.loc[:,"gyy"] = pd.Series(data=gyy, index=df.index)
    df.loc[:,"gyz"] = pd.Series(data=gyz, index=df.index)
    return df

def rotate_xyz(theta, point):
    point = np.array([point]).T
    point = rotate_x(theta[0], point)
    point = rotate_y(theta[1], point)
    point = rotate_z(theta[2], point)
    point = np.array(point.T)[0]
    return point

def rotate_x(theta, point):
    rx = np.array([[1, 0, 0],[0, math.cos(theta), -math.sin(theta)],[0, math.sin(theta), math.cos(theta)]])
    return np.asmatrix(rx)*np.asmatrix(point)

def rotate_y(theta, point):
    ry = np.array([[math.cos(theta), 0, math.sin(theta)],
                   [0, 1, 0],
                   [-math.sin(theta), 0, math.cos(theta)]])
    return np.asmatrix(ry)*np.asmatrix(point)

def rotate_z(theta, point):
    ry = np.array([[math.cos(theta), -math.sin(theta), 0],
                   [math.sin(theta), math.cos(theta), 0],
                   [0, 0, 1]])
    return np.asmatrix(ry)*np.asmatrix(point)

if __name__ == "__main__":
    netcat()
