import numpy as np
from skimage import io
import os as os
import tensorflow as tf
graph = tf.get_default_graph()

def read_index(path):
    '''
    to read the index
    data have lable
    '''
    path=path+'.txt'
    file = open(path) #
    try:
        file_context = file.read() 
        #  file_context = open(file).read().splitlines() 
    
    finally:
        file.close()
    file_context=file_context.replace('\t', '\n').split('\n')
    if len(file_context)%2==1:
        len_of_context=len(file_context)-1
    else:
        len_of_context=len(file_context) 
             
    
    x_index=list([])
    y=list([])
    for i in range(len_of_context):
        if i%2==0:
            x_index.append(file_context[i])
        else:
            y.append(file_context[i])
    return x_index,y 

def true_name(all_name,local_name):
    a=False
    for name in all_name:
        if name[0:8]==local_name:
            a=True
            return name,a
    return local_name,a

def isblack (data):
    if sum(sum(data))<20:
        return True
    else:
        return False


def get_picth(all_name,x_train_index,patch_size_1,patch_size_2,y):
    x_patch=[]
    x_patch_index=[]
    x_train_use_or_not=[]
    x_patch_to_image=[]
    y_patch=[]
    image_size=[]
    for i in range(len(x_train_index)):
        x_index_local,have_index=true_name(all_name,x_train_index[i])
        if have_index:
            x_index_local='Dataset_A/data/'+x_index_local
            pic_raw=io.imread(x_index_local)
            
            d1=int(pic_raw.shape[0]/patch_size_1)
            d2=int(pic_raw.shape[1]/patch_size_2)
            
            
            
            for patch_i in range(d1):
                for patch_j in range(d2):
                    picture_patch=pic_raw[patch_i*100:patch_i*100+100,patch_j*100:patch_j*100+100]
                    x_patch.append(picture_patch)
                    picure_index=[patch_i,patch_j]
                    x_patch_index.append(picure_index)
                    x_patch_to_image.append(i)
                    y_patch.append(y[i])
                    image_size.append(pic_raw.shape)
                    if(isblack(picture_patch)):
                        x_train_use_or_not.append(False)
                    else:
                        x_train_use_or_not.append(True)
        
    return x_patch,x_patch_index,x_train_use_or_not,x_patch_to_image,y_patch,image_size
                
def get_data_in_patch(all_name,x_train_index,patch_size_1,patch_size_2,y):
    # delete black one
    x_patch,x_patch_index,x_train_use,image_index,y2,image_size=get_picth(all_name,x_train_index,patch_size_1,patch_size_2,y)
    final_x_patch=[]
    final_x_patch_index=[]
    final_image_index=[]
    final_y=[]
    final_image_size=[]
    for i in range(len(x_train_use)):
        if(x_train_use[i]):
            final_x_patch.append(x_patch[i])
            final_x_patch_index.append(x_patch_index[i])
            final_image_index.append(image_index[i])
            final_y.append(y2[i])
            final_image_size.append(image_size[i])
    return final_x_patch,final_x_patch_index,final_image_index,final_y,final_image_size






def change_to_matrix(x_patch_index,image_index,y_patch_predict,image_size,image_index_interested,patch_size_1,patch_size_2):
    
    
    for i in range(len(image_index)):
        if (image_index_interested==image_index[i]):
            shape_a=int(image_size[i][0]/patch_size_1)
            shape_b=int(image_size[i][1]/patch_size_2)
            break
    h_max_matrix=np.zeros([shape_a,shape_b])
    class_matrix=np.zeros([shape_a,shape_b])
    for i in range(len(image_index)):
        if(image_index_interested==image_index[i]):
            a=x_patch_index[i][0]
            b=x_patch_index[i][1]
            class_matrix[a,b]=np.argmax(y_patch_predict[i])
            h_max_matrix[a,b]=max(y_patch_predict[i])
    return h_max_matrix,class_matrix

def get_thredhold(matrix):
    p1=0.2
    #can change 
    a=matrix
    x=matrix.shape[0]
    y=matrix.shape[1]
    a=a.reshape(x*y)
    a=np.sort(a)
    a_number=0
    for b in a:
        if b!=0:
            a_number=a_number+1
    a_remain=int(p1*a_number)
    thred1=a[a_number-a_remain]
    thred=min(thred1,0.7)
    return thred


def get_data_for_svm(x_patch_index,image_index,y_patch_predict,image_size,patch_size_1,patch_size_2 ):
    data=[]
    for image_index_interested in range(max(image_index)+1):
        h_max_matrix,class_matrix=change_to_matrix(x_patch_index,image_index,y_patch_predict,image_size,image_index_interested,patch_size_1,patch_size_2)
        class0_number=0
        class1_number=0
        
        
        shape_a=h_max_matrix.shape[0]
        shape_b=h_max_matrix.shape[1]
        thed=get_thredhold(h_max_matrix)
        
        for i in range(shape_a):
            for j in range(shape_b):
                if h_max_matrix[i,j]>thed:
                    if class_matrix[i,j]==0:
                        class0_number+=1
                    if class_matrix[i,j]==1:
                        class1_number+=1
        
        local_number=[class0_number,class1_number]
        data.append(local_number)
    return data





patch_size_1=100
patch_size_2=100
all_name=os.listdir('Dataset_A\data')



#main
x_train_index,y_train=read_index("train")
y_train=np.asarray(y_train,np.int32)
x_train_patch,x_train_patch_index,train_image_index,y_train_patch,train_image_size=get_data_in_patch(all_name,x_train_index,patch_size_1,patch_size_2,y_train)

x_train_patch = np.asarray(x_train_patch)
x_train_patch = x_train_patch.reshape((-1, 100, 100, 1))

config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('./CNN_50.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))
y_pred = graph.get_tensor_by_name("y_pred:0")
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1, 2))
feed_dict_testing = {x:x_train_patch, y_true: y_test_images}
pred_list = sess.run(y_pred, feed_dict=feed_dict_testing).tolist()
pred = sess.run(y_pred, feed_dict=feed_dict_testing).tolist()
label = sess.run(tf.argmax(pred, 1))

y_train_patch_predict=pred_list

x_train_svm=get_data_for_svm(x_train_patch_index,train_image_index,y_train_patch_predict,train_image_size,patch_size_1,patch_size_2 )
x_train_svm=np.array(x_train_svm)
print(x_train_svm.shape)
#y_train_patch_predict from cnn 
y_train_svm=y_train



from sklearn.svm import SVC
m=SVC();

m.fit(x_train_svm,y_train_svm)






#val 
x_val_index,y_val=read_index("val")
y_val=np.asarray(y_train,np.int32)
x_val_patch,x_val_patch_index,val_image_index,y_val_patch,val_image_size=get_data_in_patch(all_name,x_val_index,patch_size_1,patch_size_2,y_val)

x_val_patch = np.asarray(x_val_patch)
x_val_patch = x_val_patch.reshape((-1, 100, 100, 1))

config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('./CNN_50.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))
y_pred = graph.get_tensor_by_name("y_pred:0")
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1, 2))
feed_dict_testing = {x:x_val_patch, y_true: y_test_images}
pred_list = sess.run(y_pred, feed_dict=feed_dict_testing).tolist()
pred = sess.run(y_pred, feed_dict=feed_dict_testing).tolist()
label = sess.run(tf.argmax(pred, 1))

y_val_patch_predict=pred_list



x_val_svm=get_data_for_svm(x_val_patch_index,val_image_index,y_val_patch_predict,val_image_size,patch_size_1,patch_size_2 )
x_val_svm=np.array(x_val_svm)
y_val_predict=m.predict(x_val_svm)









#train 
def read_test_index(path):
    '''
    to read the index
    data have lable
    '''
    path=path+'.txt'
    file = open(path) #
    try:
        file_context = file.read() 
        #  file_context = open(file).read().splitlines() 
    
    finally:
        file.close()
    file_context=file_context.replace('\t', '\n').split('\n')
    
    if file_context[len(file_context)-1]=='':
        file_context=file_context[0:len(file_context)-1]
    
    return file_context

x_test_index=read_test_index('test')
y_test=np.zeros(len(x_test_index))-1
x_test_patch,x_test_patch_index,test_image_index,y_test_patch,test_image_size=get_data_in_patch(all_name,x_test_index,patch_size_1,patch_size_2,y_test)

x_test_patch = np.asarray(x_test_patch)
x_test_patch = x_test_patch.reshape((-1, 100, 100, 1))



config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('./CNN_50.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))
y_pred = graph.get_tensor_by_name("y_pred:0")
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1, 2))
feed_dict_testing = {x:x_val_patch, y_true: y_test_images}
pred_list = sess.run(y_pred, feed_dict=feed_dict_testing).tolist()
pred = sess.run(y_pred, feed_dict=feed_dict_testing).tolist()
label = sess.run(tf.argmax(pred, 1))

y_test_patch_predict=pred_list


x_test_svm=get_data_for_svm(x_test_patch_index,test_image_index,y_test_patch_predict,test_image_size,patch_size_1,patch_size_2 )
x_test_svm=np.array(x_test_svm)
y_test_predict=m.predict(x_test_svm)








        