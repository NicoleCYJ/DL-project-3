import numpy as np
from skimage import io
import os as os

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



patch_size_1=100
patch_size_2=100


all_name=os.listdir('Dataset_A\data')
    
x_train_index,y_train=read_index("train")
y_train=np.asarray(y_train,np.int32)


def change_shape(x,patch_size_1,patch_size_2):
    result=[]
    for i in range(len(x)):
        result.append(x[i].resape)
        
    return result


'''
x_train_patch,x_train_patch_index,train_image_index,y_train_patch,train_image_size=get_data_in_patch(all_name,x_train_index[4:10],patch_size_1,patch_size_2,y_train)

total=np.zeros([3300,2500,2])

for i in range(6):
    x_index_local,x=true_name(all_name,x_train_index[i])
    x_index_local='Dataset_A/data/'+x_index_local
    pic_raw=io.imread(x_index_local)
    print(pic_raw.shape)
    a=int(pic_raw.shape[0]/100)
    b=int(pic_raw.shape[1]/100)
    pic_raw=pic_raw[0:a*100,0:b*100]
    for j in range(len(y_train_patch)):
        if image_index[j]==i:
            pic_raw[x_train_patch_index[j][0]*100:x_train_patch_index[j][0]*100+100,x_train_patch_index[j][1]*100:x_train_patch_index[j][1]*100+100]-=x_train_patch[j]
    print(i)
    print(sum(sum(pic_raw)))
'''
    

