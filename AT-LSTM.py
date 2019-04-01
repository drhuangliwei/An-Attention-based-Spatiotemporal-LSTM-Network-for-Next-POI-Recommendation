# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 16:25:31 2017
@author: Liwei Huang
"""

import tensorflow as tf
import numpy as np
import prepare_data
import time

def data_type():
    return tf.float32

class AT_LSTM(object):
    
    def __init__(self,config,is_training=True):
        """
        :param is_training: 
        """
        self.num_steps = num_steps = config.num_steps 
        self.regularization=config.regularization
        size = config.hidden_size 
        vocab_size = config.vocab_size 
        user_size=config.user_size
        num_sample=config.num_sample

        self.batch_size = tf.placeholder(tf.int32, [])  
        self._input_data = tf.placeholder(tf.int32, [None, num_steps])    
        self._input_space = tf.placeholder(tf.float32, [None, num_steps])  
        self._input_time = tf.placeholder(tf.float32, [None, num_steps])   
        self._user = tf.placeholder(tf.int32, [None, 1])  
        self._targets = tf.placeholder(tf.int32, [None,1]) 
        self._target_time=tf.placeholder(tf.float32, [None, 1])
        self._target_space=tf.placeholder(tf.float32, [None, 1])
        
        self.mask_x = tf.placeholder(tf.float32,[None, num_steps])
        self._negative_samples = tf.placeholder(tf.int32,[None, num_sample])
        self._negative_samples_time = tf.placeholder(tf.float32,[None, num_sample])
        self._negative_samples_distance = tf.placeholder(tf.float32,[None, num_sample])
        
        self.vocabulary_distance = tf.placeholder(tf.float32,[None, vocab_size])
        
        batch_size=self.batch_size
        
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)
        self._initial_state = cell.zero_state(batch_size, data_type())
    
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=data_type())           
            squences = tf.nn.embedding_lookup(embedding, self._input_data) 
            targets = tf.nn.embedding_lookup(embedding, self._targets)
            negative_samples=tf.nn.embedding_lookup(embedding, self._negative_samples)
            embedding_user=tf.get_variable(
                "embedding_user", [user_size, size], dtype=data_type()) 
            user=tf.nn.embedding_lookup(embedding_user, self._user)
        state = self._initial_state 
        
        W_p = tf.tile(tf.expand_dims(tf.get_variable("W_p", [size, size], dtype=data_type()), 0), [batch_size,1, 1])
        W_t = tf.tile(tf.expand_dims(tf.get_variable("W_t", [1, size], dtype=data_type()), 0), [batch_size, 1, 1])
        W_s = tf.tile(tf.expand_dims(tf.get_variable("W_s", [1, size], dtype=data_type()), 0), [batch_size, 1, 1])

        inputs_x = tf.matmul(squences,W_p)+tf.matmul(tf.expand_dims(self._input_time,2),W_t)+tf.matmul(tf.expand_dims(self._input_space,2),W_s)
        target_input = tf.matmul(targets,W_p)+tf.matmul(tf.expand_dims(self._target_time,2),W_t)+tf.matmul(tf.expand_dims(self._target_space,2),W_s)
        negative_sample_input = tf.matmul(negative_samples, W_p) + tf.matmul(tf.expand_dims(self._negative_samples_time, 2), W_t) + tf.matmul(tf.expand_dims(self._negative_samples_distance, 2), W_s)
        
        if is_training and config.keep_prob < 1:
            inputs_x = tf.nn.dropout(inputs_x, config.keep_prob)
        
        outputs, states = tf.nn.dynamic_rnn(cell, inputs=inputs_x, initial_state=state, time_major=False)
        self._final_state = states

        W_r = tf.tile(tf.expand_dims(tf.get_variable("W_r", [size, size], dtype=data_type()), 0), [batch_size,1, 1])
        W_u = tf.tile(tf.expand_dims(tf.get_variable("W_u", [size, size], dtype=data_type()), 0), [batch_size,1, 1])
        queries = tf.tile(tf.expand_dims(tf.get_variable("W_z", [1, size], dtype=data_type()), 0), [batch_size,1, 1])
        weights = tf.matmul(queries, tf.transpose(outputs, [0, 2, 1]))  # (batch_size,1, num_steps)
        weights = weights / (size ** 0.5)
        weights = tf.nn.softmax(weights)  # (batch_size,1, num_steps)
        weights *= self.mask_x[:,None,:]  # broadcasting. (batch_size, 1, num_steps)
        r = tf.matmul(weights, outputs) # ( batch_size,1, size)

        output_=tf.matmul(r,W_r)+tf.matmul(user,W_u) # ( batch_size,1, size)
        output_y=tf.matmul(output_,tf.transpose(target_input, [0, 2, 1])) # ( batch_size,1, 1)
        output_sample = tf.matmul(output_, tf.transpose(negative_sample_input, [0, 2, 1]))  # ( batch_size,1, num_sample)
   
        self._lr = tf.Variable(0.0, trainable=False)
        self.tvars = tf.trainable_variables() 
               
        """
        Compute the BPR objective which is sum_uij ln sigma(x_uij) 
        
        """ 
        if is_training:
            ranking_loss = tf.reduce_sum(tf.log(tf.clip_by_value((1.0+tf.exp(-tf.to_float(tf.tile(output_y,[1,1,num_sample])-output_sample))),1e-8,1.0)))
            self.cost = tf.div(ranking_loss,tf.to_float(num_sample*batch_size))

        if not is_training:
            with tf.name_scope("prediciton"):
                all_time = tf.tile(tf.expand_dims(self._target_time,2),[1,vocab_size,1])
                all_input = tf.matmul(tf.tile(tf.expand_dims(embedding,0),[batch_size,1,1]), W_p) + tf.matmul(all_time, W_t) + tf.matmul(tf.expand_dims(self.vocabulary_distance,2), W_s)
                logits = tf.reshape(tf.matmul(output_, tf.transpose(all_input, [0, 2, 1])),[batch_size,-1]) # ( batch_size,vocab_size)

                self.prediction_5=tf.nn.top_k(logits,5)[1]
                self.prediction_10=tf.nn.top_k(logits,10)[1]

                expand_targets = tf.tile(self._targets, [1, 5])
                isequal = tf.equal(expand_targets, self.prediction_5)
                correct_prediction_5 = tf.reduce_sum(tf.cast(isequal, tf.float32))
                self.precison_5 = correct_prediction_5 / tf.cast(batch_size*5,tf.float32)
                self.recall_5 = correct_prediction_5 /tf.cast(batch_size,tf.float32)
                self.f1_5 = 2 * self.precison_5 * self.recall_5 / (self.precison_5 + self.recall_5 + 1e-10)

                expand_targets = tf.tile(self._targets, [1, 10])
                isequal = tf.equal(expand_targets, self.prediction_10)
                correct_prediction_10 = tf.reduce_sum(tf.cast(isequal, tf.float32))
                self.precison_10 = correct_prediction_10 / tf.cast(batch_size*10,tf.float32)
                self.recall_10 = correct_prediction_10 / tf.cast(batch_size,tf.float32)
                self.f1_10 = 2 * self.precison_10 * self.recall_10 / (self.precison_10 + self.recall_10 + 1e-10)
    
        if not is_training:  
            return
    
        #optiminzer
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, self.tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.AdagradOptimizer(self._lr)
        self.train_op=optimizer.apply_gradients(zip(grads, self.tvars))
    
        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")    
        self._lr_update = tf.assign(self._lr, self._new_lr)
        
    def assign_new_lr(self,session,lr_value):
        session.run(self._lr_update,feed_dict={self._new_lr:lr_value})
            
    def assign_new_num_steps(self,session,num_steps_value):
        session.run(self._num_step_update,feed_dict={self.new_num_steps:num_steps_value})
    
    def initial_state(self):
        return self._initial_state
    
    def final_state(self):
        return self._final_state

        
class Config(object):
    """config."""
    init_scale = 1
    learning_rate = 0.01
    max_grad_norm = 5
    num_layers = 1
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 10
    keep_prob = 0.7
    lr_decay = 0.5
    batch_size = 2
    vocab_size = 10000
    regularization=0.0025
    num_sample=500
    user_size=1000

def evaluate(model,session,data,vocabulary_distances,batch_size):
    total_num=0
    total_recall_5=0.0
    total_f1_5=0.0
    total_recall_10=0.0
    total_f1_10=0.0
    state = session.run(model._initial_state,feed_dict={model.batch_size:batch_size})
    fetches = [model.recall_5,model.f1_5,model.recall_10,model.f1_10, model._final_state]
    for step, (return_sequence_x,return_sequence_y,return_time_x,return_time_y,return_distance_x,
               return_distance_y,return_mask_x,return_vocabulary_distances,return_user) in enumerate(prepare_data.batch_iter(data,vocabulary_distances,batch_size)):
        total_num=total_num+1

        feed_dict={}
        feed_dict[model.batch_size]=batch_size 
        feed_dict[model._input_data]=return_sequence_x
        feed_dict[model._input_space]=return_distance_x.astype(np.float32)
        feed_dict[model._input_time]=return_time_x.astype(np.float32)
        feed_dict[model._user]=np.reshape(return_user,[return_user.shape[0],1])
        feed_dict[model._targets]=return_sequence_y
        feed_dict[model._target_time]=return_time_y
        feed_dict[model._target_space]=return_distance_y
        feed_dict[model.mask_x]=return_mask_x
        feed_dict[model.vocabulary_distance]=[x.astype(np.float32) for x in return_vocabulary_distances]
        
        for i , (c,h) in enumerate(model._initial_state):
            feed_dict[c]=state[i].c
            feed_dict[h]=state[i].h
        c_recall_5,c_f1_5,c_recall_10,c_f1_10,state=session.run(fetches,feed_dict)
        total_recall_5=total_recall_5+c_recall_5
        total_f1_5=total_f1_5+c_f1_5
        total_recall_10=total_recall_10+c_recall_10
        total_f1_10=total_f1_10+c_f1_10
        
    total_recall_5=total_recall_5/total_num
    total_f1_5=total_f1_5/total_num
    total_recall_10=total_recall_10/total_num
    total_f1_10=total_f1_10/total_num

    return total_recall_5,total_f1_5,total_recall_10,total_f1_10
    
def run_epoch(model,session,data,negative_samples,global_steps,batch_size):
    """Runs the model on the given data."""
    
    state = session.run(model._initial_state,feed_dict={model.batch_size:batch_size})
    fetches = [model.cost,model._final_state,model.train_op]
    
    for step, (return_sequence_x,return_sequence_y,return_time_x,return_time_y,return_distance_x,
               return_distance_y,return_mask_x,
               return_negative_sample,return_negative_time_sample,return_negative_distance_sample,return_user) in enumerate(prepare_data.batch_iter_sample(data,negative_samples,batch_size)):
        
        feed_dict={}
        feed_dict[model.batch_size]=batch_size 
        feed_dict[model._input_data]=return_sequence_x 
        feed_dict[model._input_space]=return_distance_x.astype(np.float32)
        feed_dict[model._input_time]=return_time_x.astype(np.float32)
        feed_dict[model._user]=np.reshape(return_user,[return_user.shape[0],1])
      
        feed_dict[model._targets]=return_sequence_y
        feed_dict[model._target_time]=return_time_y
        feed_dict[model._target_space]=return_distance_y
        feed_dict[model.mask_x]=return_mask_x
        feed_dict[model._negative_samples]=return_negative_sample
        feed_dict[model._negative_samples_time]=return_negative_time_sample.astype(np.float32)
        feed_dict[model._negative_samples_distance]=return_negative_distance_sample.astype(np.float32)
        
        for i , (c,h) in enumerate(model._initial_state):
            feed_dict[c]=state[i].c
            feed_dict[h]=state[i].h
        
        cost,state,_ = session.run(fetches,feed_dict)

        if(global_steps%100==0):
            print("the %i step, train cost is: %f"%(global_steps,cost))
        global_steps+=1
        
    return global_steps,cost

if __name__=='__main__':
    
    print("loading the dataset...")
    config=Config()
    eval_config=Config()
    
    userlocation = np.loadtxt(open('data\\Gowalla\\userlocation.csv','rb'),delimiter=',',skiprows=0)
    locations=np.loadtxt(open('data\\Gowalla\\locations.csv','rb'),delimiter=',',skiprows=0)

    newsequence, sequence_user, sequence_time, sequence_distance = prepare_data._build_sequence(userlocation)
    sequence,clusters=prepare_data.new_build_location_voc(newsequence,locations)
    top_500=prepare_data.pop_n(sequence,config.num_sample)
    
    total_recall_5=0.0
    total_f1_5=0.0
    total_recall_10=0.0
    total_f1_10=0.0
    
    total_user=len(sequence_user)
    config.user_size=total_user
    eval_config.user_size=total_user
    
    train_set=(sequence,sequence_user, sequence_time, sequence_distance)
        
    final_train_set,final_test_set,final_negative_samples,vocabulary_distances=prepare_data.load_data(train_set,locations,config.num_sample,clusters,top_500,0.1,True)
      
    new_train_set_sequence,new_train_set_time,new_train_set_distance,mask_train_x,train_set_user=final_train_set
    negative_samples,negative_time_sample,negative_distance_samples=final_negative_samples
    config.vocab_size=locations.shape[1]
        
    config.num_steps=new_train_set_sequence[0].shape[1]
    if new_train_set_sequence[0].shape[0]<=10:
        config.batch_size=1
    else:
        if new_train_set_sequence[0].shape[0]<=50:
            config.batch_size=2
        else:
            if new_train_set_sequence[0].shape[0]<=100:
                config.batch_size=5
            else:
                if new_train_set_sequence[0].shape[0]<=200:
                    config.batch_size=10
                else:
                    config.batch_size=20
        
    print("begin the training process")
        
        
    eval_config.keep_prob=1.0
    eval_config.num_steps=config.num_steps
    eval_config.vocab_size=locations.shape[1]
    eval_config.batch_size=1
       
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-1*config.init_scale,1*config.init_scale)
        with tf.variable_scope("model",reuse=None,initializer=initializer):
            model = AT_LSTM(config=config,is_training=True)
    
        with tf.variable_scope("model",reuse=True,initializer=initializer):
            test_model = AT_LSTM(config=eval_config,is_training=False)
    
        summary_writer = tf.summary.FileWriter('/tmp/lstm_logs',session.graph)
    
        tf.global_variables_initializer().run()  # 对参数变量初始化
        global_steps=1
        begin_time=int(time.time())
            
        for i in range(config.max_max_epoch):
            print("the %d epoch training..."%(i+1))
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            model.assign_new_lr(session,config.learning_rate*lr_decay)
            global_steps,cost=run_epoch(model,session,final_train_set,final_negative_samples,global_steps,config.batch_size)
            if cost<0.005:
                break
                    
        print("the train is finished")
        end_time=int(time.time())
        print("training takes %d seconds already\n"%(end_time-begin_time))
        recall_5,f1_5,recall_10,f1_10=evaluate(test_model,session,final_test_set,vocabulary_distances,eval_config.batch_size)
        print("the test data total_recall_5 is %f,total_f1_5 is %f,total_recall_10 is %f,total_f1_10 is %f"%(recall_5,f1_5,recall_10,f1_10))
        print("program end!")
    
    
    
    