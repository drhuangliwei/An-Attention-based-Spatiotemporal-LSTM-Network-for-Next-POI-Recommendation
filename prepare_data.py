# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 19:25:13 2017
@author: Liwei Huang
"""

import numpy as np
import random,collections,itertools
import math

def generate_mask(x,max_len):
    new_mask_x = np.zeros([len(x),max_len])
    for i,y in enumerate(x):
        if len(y)<=max_len:
            new_mask_x[i,0:len(y)]=1
        else: 
            new_mask_x[i,:]=1
    return new_mask_x

def padding(x,y,new_x,new_y,max_len):
    for i,(x,y) in enumerate(zip(x,y)):
        if len(x)<=max_len:
            new_x[i,0:len(x)]=x
            new_y[i]=y
        else:
            new_x[i]=(x[0:max_len])
            new_y[i]=y
    new_set =(new_x,new_y)
    del new_x,new_y
    return new_set

def padding_negative_sample(targets,negative_sample,negative_distance_sample,locations,clusters,sequence):
    suqence_num,num_sample=negative_sample.shape
    for i in range(suqence_num):
        negative_sample[i,:]=np.mat(generate_negative_sample(targets[i][-1],locations,num_sample,clusters,sequence)[0:num_sample])
    for i in range(suqence_num):
        target_location=(locations[1,int(targets[i][-1])],locations[2,int(targets[i][-1])])
        for j in range(num_sample):
            c_location=(locations[1,int(negative_sample[i,j])],locations[2,int(negative_sample[i,j])])
            negative_distance_sample[i,j]=haversine(target_location,c_location)
    return negative_sample,negative_distance_sample

def padding_train_time(x,y,new_x,new_y,max_len):
    for i,(x,y) in enumerate(zip(x,y)):
        if len(x)<=max_len:
            new_x[i,0:len(x)]=x
            new_y[i,0]=y
            new_y[i,1]=y
        else:
            new_x[i]=(x[0:max_len])
            new_y[i,0]=y
            new_y[i,1]=y
    new_set =(new_x,new_y)
    del new_x,new_y
    return new_set

def generate_negative_sample(l,locations,num_sample,clusters,top_500):
    cluster_j=int(locations[3,int(l)])
    n_samples=len(clusters[cluster_j])
    if n_samples>=num_sample:
        index= random.sample(range(n_samples),num_sample) 
        lastindex=[clusters[cluster_j][i] for i in index]
        if l in lastindex:
            index= random.sample(range(n_samples),num_sample) 
            lastindex=[clusters[cluster_j][i] for i in index]
    else:
        if n_samples>1:
            lastindex=list(set(clusters[cluster_j])^set([int(l)]))+top_500[0:num_sample-n_samples+1]
        else:
            lastindex=top_500[0:num_sample-n_samples+1]
    return lastindex

def pop_n(sequence,k): 
    Locations_voc = collections.Counter(list(itertools.chain.from_iterable(sequence)))
    sorted_Locations_voc=sorted(Locations_voc.items(), key=lambda d:d[1], reverse = True )
    return [a for i,(a,b) in enumerate(sorted_Locations_voc) if i<k]

def padding_vocabulary_distance(targets,locations):
    vocabulary_distance=np.zeros([len(targets),locations.shape[1]])
    suqence_num,voc_size=vocabulary_distance.shape
    for i in range(suqence_num):
        target_location=(locations[1,int(targets[i][-1])],locations[2,int(targets[i][-1])])
        for j in range(voc_size):
            c_location=(locations[1,j],locations[2,j])
            vocabulary_distance[i,j]=haversine(target_location,c_location)
    return vocabulary_distance

def load_data(train_set,locations,num_sample,clusters,top_500,test_portion=0.1,sort_by_len=True):
    
    (train_set_sequence,sequence_user, train_set_time, train_set_distance)=train_set
    
    max_len=max([len(x) for x in train_set_sequence]) 
    new_sequence=[]
    new_sequence_user=[]
    new_time=[]
    new_distance=[]

    #data augmentation
    for k in range(len(train_set_sequence)):
        for i in range(len(train_set_sequence[k])-2): 
            new_sequence.append(train_set_sequence[k][0:i+3])
            new_sequence_user.append(sequence_user[k])
            new_time.append(train_set_time[k][0:i+3])
            new_distance.append(train_set_distance[k][0:i+3])

    print("generate the train set and test set")
    n_samples= len(new_sequence)
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - test_portion)))
    
    test_set_sequence = [new_sequence[s] for s in sidx[n_train:]]
    test_set_time= [new_time[s] for s in sidx[n_train:]]
    test_set_distance= [new_distance[s] for s in sidx[n_train:]]
    test_set_user= [new_sequence_user[s] for s in sidx[n_train:]]
    
    train_set_sequence = [new_sequence[s] for s in sidx[:n_train]]
    train_set_time= [new_time[s] for s in sidx[:n_train]]
    train_set_distance= [new_distance[s] for s in sidx[:n_train]]
    train_set_user= [new_sequence_user[s] for s in sidx[:n_train]]
    
    def len_argsort(seq): 
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_sequence)
        test_set_sequence =[test_set_sequence[i] for i in sorted_index]
        test_set_time= [test_set_time[i] for i in sorted_index]
        test_set_distance= [test_set_distance[i] for i in sorted_index]
        test_set_user= [test_set_user[i] for i in sorted_index]
        
        sorted_index = len_argsort(train_set_sequence)
        train_set_sequence =[train_set_sequence[i] for i in sorted_index]
        train_set_time= [train_set_time[i] for i in sorted_index]
        train_set_distance= [train_set_distance[i] for i in sorted_index]
        train_set_user= [train_set_user[i] for i in sorted_index]
    
    test_set_sequence_x = [x[0:len(x)-1] for x in test_set_sequence]
    test_set_time_x = [x[0:len(x)-1] for x in test_set_time]
    test_set_distance_x = [x[0:len(x)-1] for x in test_set_distance]
    train_set_sequence_x =[x[0:len(x)-1] for x in train_set_sequence]
    train_set_time_x= [x[0:len(x)-1] for x in train_set_time]
    train_set_distance_x= [x[0:len(x)-1] for x in train_set_distance]
    
    
    test_set_sequence_y = [x[len(x)-1] for x in test_set_sequence]
    test_set_time_y = [x[len(x)-1] for x in test_set_time]
    test_set_distance_y = [x[len(x)-1] for x in test_set_distance]
    train_set_sequence_y =[x[len(x)-1] for x in train_set_sequence]
    train_set_time_y= [x[len(x)-1] for x in train_set_time]
    train_set_distance_y= [x[len(x)-1] for x in train_set_distance]
    
    new_test_set_sequence_x =np.zeros([len(test_set_sequence_x),max_len])
    new_test_set_time_x = np.zeros([len(test_set_time_x),max_len])
    new_test_set_distance_x = np.zeros([len(test_set_distance_x),max_len])
    new_train_set_sequence_x =np.zeros([len(train_set_sequence_x),max_len])
    new_train_set_time_x= np.zeros([len(train_set_time_x),max_len])
    new_train_set_distance_x= np.zeros([len(train_set_distance_x),max_len])

    new_test_set_sequence_y =np.zeros([len(test_set_sequence_y),1])
    new_test_set_time_y = np.zeros([len(test_set_time_y),1])
    new_test_set_distance_y = np.zeros([len(test_set_distance_y),1])
    new_train_set_sequence_y =np.zeros([len(train_set_sequence_y),1]) 
    new_train_set_time_y= np.zeros([len(train_set_time_y),1]) 
    new_train_set_distance_y= np.zeros([len(train_set_distance_y),1])
    
    negative_sample=np.zeros([len(new_train_set_sequence_y),num_sample]) 
    negative_time_sample=np.zeros([len(new_train_set_sequence_y),num_sample])
    negative_distance_sample=np.zeros([len(new_train_set_sequence_y),num_sample])
    
    print("begin the padding process")
    
    new_train_set_sequence=padding(train_set_sequence_x,train_set_sequence_y,
                                                 new_train_set_sequence_x,new_train_set_sequence_y,max_len)
    new_train_set_time=padding(train_set_time_x,train_set_time_y,
                                                 new_train_set_time_x,new_train_set_time_y,max_len)
    new_train_set_distance=padding(train_set_distance_x,train_set_distance_y,
                                                 new_train_set_distance_x,new_train_set_distance_y,max_len)
    mask_train_x=generate_mask(train_set_sequence_x,max_len)
    
      
    new_test_set_sequence=padding(test_set_sequence_x,test_set_sequence_y,
                                                 new_test_set_sequence_x,new_test_set_sequence_y,max_len)
    new_test_set_time=padding(test_set_time_x,test_set_time_y,
                                                 new_test_set_time_x,new_test_set_time_y,max_len)
    new_test_set_distance=padding(test_set_distance_x,test_set_distance_y,
                                                 new_test_set_distance_x,new_test_set_distance_y,max_len)  
    mask_test_x=generate_mask(test_set_sequence_x,max_len)
    
    negative_samples,negative_distance_samples=padding_negative_sample(train_set_sequence_x,negative_sample,negative_distance_sample,locations,clusters,top_500)
    for i in range(num_sample):
        negative_time_sample[:,i]=train_set_time_y
        
    vocabulary_distances=padding_vocabulary_distance(test_set_sequence_x,locations)
    
    test_set_user=np.array(test_set_user)
    train_set_user=np.array(train_set_user)
    
    final_train_set=(new_train_set_sequence,new_train_set_time,new_train_set_distance,mask_train_x,train_set_user)
    final_test_set=(new_test_set_sequence,new_test_set_time,new_test_set_distance,mask_test_x,test_set_user)
    final_negative_samples=(negative_samples,negative_time_sample,negative_distance_samples)
    
    return final_train_set,final_test_set,final_negative_samples,vocabulary_distances
    
def batch_iter(data,vocabulary_distances,batch_size):  
    sequence,time,distance,mask_x,user=data
    sequence_x,sequence_y=sequence
    time_x,time_y=time
    distance_x,distance_y=distance
    data_size=len(sequence_x)
    
    num_batches_per_epoch=int(data_size/batch_size) 
    for batch_index in range(num_batches_per_epoch):
        start_index=batch_index*batch_size
        end_index=min((batch_index+1)*batch_size,data_size)
        return_sequence_x = sequence_x[start_index:end_index,:]
        return_sequence_y = sequence_y[start_index:end_index,:]
    
        return_time_x = time_x[start_index:end_index,:]
        return_time_y = time_y[start_index:end_index,:]
    
        return_distance_x = distance_x[start_index:end_index,:]
        return_distance_y = distance_y[start_index:end_index,:]
    
        return_vocabulary_distances=vocabulary_distances[start_index:end_index,:]
    
        return_mask_x = mask_x[start_index:end_index,:]
    
        return_user = user[start_index:end_index]
       
        yield (return_sequence_x,return_sequence_y,return_time_x,return_time_y,return_distance_x,
               return_distance_y,return_mask_x,return_vocabulary_distances,return_user)        

def batch_iter_sample(data,negative_samples,batch_size): 
   
    sequence,time,distance,mask_x,user=data
    negative_sample,negative_time_sample,negative_distance_sample=negative_samples
    sequence_x,sequence_y=sequence
    time_x,time_y=time
    distance_x,distance_y=distance
    
    data_size=len(sequence_x)
    
    num_batches_per_epoch=int(data_size/batch_size)
    for batch_index in range(num_batches_per_epoch):
        start_index=batch_index*batch_size
        end_index=min((batch_index+1)*batch_size,data_size)
        return_sequence_x = sequence_x[start_index:end_index,:]
        return_sequence_y = sequence_y[start_index:end_index,:]
    
        return_time_x = time_x[start_index:end_index,:]
        return_time_y = time_y[start_index:end_index,:]
    
        return_distance_x = distance_x[start_index:end_index,:]
        return_distance_y = distance_y[start_index:end_index,:]
    
        return_mask_x = mask_x[start_index:end_index,:]
    
        return_negative_sample=negative_sample[start_index:end_index,:]
        return_negative_time_sample=negative_time_sample[start_index:end_index,:]
        return_negative_distance_sample=negative_distance_sample[start_index:end_index,:] 
        return_user = user[start_index:end_index]
    
        yield (return_sequence_x,return_sequence_y,return_time_x,return_time_y,return_distance_x,
               return_distance_y,return_mask_x,
               return_negative_sample,return_negative_time_sample,return_negative_distance_sample,return_user)
               
def new_build_location_voc(sequence,locations):
    
    Locations_voc = collections.Counter(list(itertools.chain.from_iterable(sequence)))
    location_list=list(Locations_voc.keys())
    newsequence=[]
    word_to_id = dict(zip(location_list, range(len(location_list))))
    
    for lst in sequence:
        newsequence.append([word_to_id[x] for x in lst])
    
    citys=locations[3,:].tolist()
    clusters=[]
    city_voc = collections.Counter(citys)
    city_list=list(city_voc.keys())
    city_to_id = dict(zip(city_list, range(len(city_list))))
    citys_id=[city_to_id[word] for word in citys]
    for i in range(len(city_list)):
        clusters.append([n for n in range(len(citys_id)) if citys_id[n] == i])
    
    return newsequence,clusters

def haversine(lonlat1, lonlat2):
    lat1, lon1 = lonlat1
    lat2, lon2 = lonlat2
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371 
    return c * r

def _build_sequence(userlocation): 

    userlocation=np.array(userlocation)
    user_voc=collections.Counter(userlocation[0,:].tolist())
    sequence=[]  
    sequence_user=[]
    sequence_time=[] 
    sequence_distance=[]

    print("build the sequence！！！！")
    k=0
    sum_sequence=0
 
    for user in user_voc.keys():
        
        k=k+1
        if k%1000==0:
            print(k)

        checkin_user_redex=np.argwhere(userlocation[0,:]==user) 
        checkin_user_all=userlocation[:,checkin_user_redex[:,0]]
        
        user_count=0
        sequence_location=[]
        sequence_time_user=[]
        sequence_distance_user=[]
        
        temperal_sequence_location=[]
        temperal_sequence_time_user=[] 
        temperal_sequence_distance_user=[]
        
        sorted_time=np.sort(checkin_user_all[2,:]) 
        sorted_time_index=np.argsort(checkin_user_all[2,:])
        
        for i in range(len(checkin_user_redex)):

            if i==0: 
                sequence_location.append(checkin_user_all[1,sorted_time_index[i]])
                sequence_time_user.append(100) 
                sequence_distance_user.append(1) 
            else:                
                if sorted_time[i]-sorted_time[i-1]>21600:
                    if len(sequence_location)>4:
                        sequence_location=list(map(int, sequence_location))
                        sequence_time_user=list(map(int, sequence_time_user))
                        temperal_sequence_location.append(sequence_location)
                        temperal_sequence_time_user.append(sequence_time_user)
                        temperal_sequence_distance_user.append(sequence_distance_user)
                        user_count=user_count+1
                                      
                    sequence_location=[]
                    sequence_time_user=[] 
                    sequence_distance_user=[]
                    sequence_location.append(checkin_user_all[1,sorted_time_index[i]])
                    sequence_time_user.append(100) 
                    sequence_distance_user.append(1)
                else:
                    sequence_location.append(checkin_user_all[1,sorted_time_index[i]])
                    sequence_time_user.append(sorted_time[i]-sorted_time[i-1]+1e-5)
                    latitude=checkin_user_all[3,sorted_time_index[i]]
                    longitude=checkin_user_all[4,sorted_time_index[i]]
                    distance=haversine((latitude,longitude),(checkin_user_all[3,sorted_time_index[i-1]],checkin_user_all[4,sorted_time_index[i-1]]))
                    sequence_distance_user.append(distance+1e-5)

        sum_sequence=sum_sequence+user_count
        
        if user_count>5:
            sequence=sequence+temperal_sequence_location
            sequence_time=sequence_time+temperal_sequence_time_user
            sequence_distance=sequence_distance+temperal_sequence_distance_user
            sequence_user=sequence_user+[user]*user_count
    
    max_time=max([max(x) for x in sequence_time])
    max_distance=max([max(x) for x in sequence_distance])
    sequence_time=[[y/max_time for y in x] for x in sequence_time]
    sequence_distance=[[y/max_distance for y in x] for x in sequence_distance]
           
    return  sequence,sequence_user,sequence_time, sequence_distance
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    