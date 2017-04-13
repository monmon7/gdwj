# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 14:07:57 2017

@author: lenovo
"""
import csv
import sys
from keras.preprocessing.text import text_to_word_sequence,base_filter,Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM, TimeDistributed, Input, Bidirectional
from keras.models import Model
import numpy as np
import pandas as pd
word_size = 128
batch_size = 128
EMBEDDING_DIM = 509 ###########词向量+结构的embedding！！！！###
#EMBEDDING_DIM = 409 ###########纯结构的embedding！！！！###
VALIDATION_SPLIT= 0.8
MAX_SEQUENCE_LENGTH= 50

for time in range(2):
    f0 = open('D:\\lmqnlp\\all_dataSet_split.txt.vec','r') #词向量文件
    f1 = open('D:\\lmqnlp\\label1_2tag.txt','r') 
    f2 = open('D:\\lmqnlp\\FXGnew.txt','r')
    
    '''
    load data from csv  新的结构词向量 ==》 rows
    3000 sentences
    '''
    rows = []
    temp0 = []
    with open('D:\\lmqnlp\\result3000.csv', 'rb') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            row = [float(x) for x in row]
            if row == [1.0]:
                rows.append(temp0)  
                temp0 = []  
                continue
            temp0.append(row)           
        csvFile.close
    
    
    datadd= []
    for count in range (3000):#fsource num of lines 
        line1 = f2.readline()
        datad = []
        if line1.strip('\n')!= '' :
            line0 = line1.strip('\n').split('\t')
            line0.pop()
            for x in line0 :
                datad.append(x)              
            if datad!=[''] :
                datadd.append(datad)
                
    
    #################################
    #   提取标签序列  #
    #################################
    d2 = []
    for count in range(3000) :
        line = f1.readline()
        d1 = []
        if line.strip('\n') != '' :        
            lali = line.strip('\n').split(',')
            lali.pop()
            for nu in lali :
                d1.append(int(nu))
            d2.append(d1)
    strucData = pd.DataFrame(index = range(len(datadd)))
    strucData['text'] = datadd
    strucData['label'] = d2
    strucData['embedding'] = rows
    
    strucData = strucData[strucData['text'].apply(len) <= MAX_SEQUENCE_LENGTH] #截断字数长的
    print '11111111111111'
    strucData.index = range(len(strucData))
    
    tagd = pd.Series({0:0, 1:1, 2:2, 3:3})
    # 0:句中不在目标片段的词 1：目标片段第一词 2：目标片段其余词 3：padding
    '''
    给每个word编号
    '''
    
    charsd = []
    index = 0
    word = []
    for i in list(strucData['text']):
        wordi = [] #每句话的编号序列
        for j in i:
            wordi.append(index)
            word.append(index)
    #        print wordi
            index += 1
        charsd.append(np.array(wordi+[0]*(MAX_SEQUENCE_LENGTH-len(i))))
#    print index
    
        
    '''
    得到基于所有评论的词典mapping   #
    '''
                    
    em = []
    for count in list(strucData['embedding']) :
        for ccount in count :
#            for vi in range(100): #纯结构时用该2句 去除前100维的词向量嵌入！！！！！！！！！！！！！！
#                del ccount[vi]#纯结构时用该2句 去除前100维的词向量嵌入！！！！！！！！！！！！！！
            em.append(np.asarray(ccount))    
    
    mapping={}#必须有这个，不然会出现NameError
    mapping=dict(zip(word,em))
    
               
    '''
    准备词向量
    '''
    # prepare embedding matrix 嵌入词向量
    embedding_matrix = np.zeros((len(word) + 1, EMBEDDING_DIM))
    for i in mapping.keys():
        embedding_vector = mapping.get(i)
        print i 
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    
    
    
    f0.close()
    f1.close()
    f2.close()
    
    '''
    lmq!!!!!↓
    '''
    
    
    strucData['x'] = charsd
    strucData['y'] = strucData['label'].apply(lambda x: np.array(map(lambda y:np_utils.to_categorical(y,4), tagd[x].reshape((-1,1)))+[np.array([[0,0,0,1]])]*(MAX_SEQUENCE_LENGTH-len(x))))
    ddy = strucData['y']
    
    trainXdd = np.array(list(strucData['x']))
    trainY0dd = np.array(list(strucData['y']))
    trainYdd = trainY0dd.reshape((-1,MAX_SEQUENCE_LENGTH,4))
    
    #加入验证集
    nb_validation_samples = int(VALIDATION_SPLIT * trainXdd.shape[0])
    #x_train_V = trainXdd[:-nb_validation_samples]
    #y_train_V = trainY0dd[:-nb_validation_samples].reshape((-1,MAX_SEQUENCE_LENGTH,4))
    #
    #x_val = trainXdd[-nb_validation_samples:]
    #y_val = trainY0dd[-nb_validation_samples:].reshape((-1,MAX_SEQUENCE_LENGTH,4))
    
    x_val = trainXdd[:-nb_validation_samples]
    y_val = trainY0dd[:-nb_validation_samples].reshape((-1,MAX_SEQUENCE_LENGTH,4))
    
    x_train_V= trainXdd[-nb_validation_samples:]
    y_train_V = trainY0dd[-nb_validation_samples:].reshape((-1,MAX_SEQUENCE_LENGTH,4))
    
    '''
    lmq!!!!!↑
    '''
    #设计模型
    print('Training model.')
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')#cixl
    print('Training model1.')
    '''
    embedding作用 将word的id或下标（一个整数）进行encoding 形成维数固定的向量
    
    '''
    embedded_sequences = Embedding(len(word)+1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=MAX_SEQUENCE_LENGTH,trainable=False,mask_zero=True)(sequence_input)
    #embedded = Embedding(len(charsd)+1, word_size, input_length=maxlend, mask_zero=True)(sequence)
    print('Training model2.')
    blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded_sequences)    
    print('Training model33333.')
    
        
    output = TimeDistributed(Dense(4, activation ='softmax'))(blstm)
    #4tag output = TimeDistributed(Dense(5, activation='softmax'))(blstm)
    print('Training model33333.')
    model = Model(input=sequence_input, output=output)
    print('Training model33333.')
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    
    print('learning.....')
    
    
    model.fit(x_train_V, y_train_V, batch_size=batch_size,nb_epoch=10)#改epoch
    #score = model.evaluate(x_val,y_val, batch_size=batch_size,verbose=1)
    #a =  model.predict_classes(x_val, batch_size=batch_size, verbose=1)
    #print('Test score:', score[0])
    #print('Test accuracy:', score[1])
    a =  model.predict(x_val, verbose=1)
    
    
    '''
    结果精确度评估
    比例方式  算A P R
    A ： 把padding的0也算进去 结果较高 没有太大参考价值
    P：
    R： 
    '''
    yr = [] #真正标签
    ar = [] #预测出的标签
    acu = 0
    tp= 0
    fp = 0
    fn = 0
    tn = 0
    for count in range(0,len(a)):
        aa = [] #记录真正标签
        rr = [] #记录预测出  
        for aline in y_val[count]: #遍历真正的每一行
            if aline[3] == 1 :
                break
            aa.append(list(aline).index(1.0))
        for rline in a[count] :#遍历 预测的每一行
            if len(rr) >= len(aa):
                break
            rline = list(rline)
            rr.append(rline.index(max(rline)))
        '''
        A P R calculation
        '''
        scoreA = 0
    
        for i in range(0,len(rr)):
            if rr[i] == aa[i]:
                scoreA+=1
            if rr[i] == aa[i] and aa[i]!= 0:
                tp+=1
            if rr[i] == aa[i] and aa[i]== 0:
                tn+=1
            if rr[i] != aa[i] and aa[i]== 0:
                fp+=1
            if rr[i] != aa[i] and aa[i]!= 0 :
                if rr[i] == 0: # 1/2 --> 0
                    fn+=1
                else:         # 1-->2 or 2-->1
                    fn+=0.5
        result = float(scoreA)/len(rr)
        yr.append(aa)
        ar.append(rr)
        acu+=result
        
    #accuracy = float(tp+tn)/float(tp+tn+fn+fp)
    accuracy = float(acu)/len(a)
    precision = float(tp)/float(tp+fp) 
    recall = float(tp)/float(tp+fn) 
    fscore = 2 * (precision*recall)/(precision+recall)
    with open('D:\\lmqnlp\\testResult.txt','a') as f:
        f.write('acurracy: '+str(accuracy)+'\n')
        f.write('precision: '+str(precision)+'\n')
        f.write('recall: '+str(recall)+'\n')      
        f.write('f1score: '+str(fscore)+'\n')
        f.write('\n')
#    print 'acurracy:',accuracy
#    print 'precision:',precision
#    print 'recall',recall
#    print 'f1:',fscore
    print time,'aaaaaaa'