import tensorflow as tf
import numpy as np

dataset_train_path = "./dataset/position_left/"
dataset_test_path = "./dataset/position_right/"

train_x_data = np.zeros(0, dtype = np.float32)
train_y_data = np.zeros(0, dtype = np.float32)
test_x_data = np.zeros(0, dtype = np.float32)
test_y_data = np.zeros(0, dtype = np.float32)

# readfile
def makeData(dataset_path, out_x_data, out_y_data):
    min_value = -10;
    max_value = 10;
    i = 0
    nframe = 1000
    while i < nframe:
        i += 1
        file_name = 'Frame' + str(i) + '.p'
        #print(file_name+'----------------------')
        f = open(dataset_path+file_name, 'r') 
        frame_array = np.zeros(0, dtype = np.float32)
        while True: ################################### FrameN.p
            line = f.readline()
            if not line: break
            else:################### line(node N xyz)
                pos = np.fromstring(line, dtype = np.float32, sep= ' ')
                pos = (pos - min_value) / (max_value - min_value)
                frame_array = np.append(frame_array, pos)
        f.close()
        #print('frame %d'%i)
        #print(frame_array)

        if i < nframe:
            #print('train x , frame %d'%i)
            out_x_data = np.append(out_x_data, frame_array)
            if i > 1:
                #print('train y, frame %d' % i)
               out_y_data = np.append(out_y_data, frame_array)
        else:
            #print('train y, frame %d' % i)
            out_y_data = np.append(out_y_data, frame_array)

        out_x_data = np.reshape(out_x_data, (-1, 60))
        out_y_data = np.reshape(out_y_data, (-1, 60))
    
    return out_x_data, out_y_data

train_x_data, train_y_data = makeData(dataset_train_path, train_x_data, train_y_data)
test_x_data, test_y_data = makeData(dataset_test_path, test_x_data, test_y_data)
print(train_x_data)
print(train_y_data)
