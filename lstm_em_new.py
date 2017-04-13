# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 14:59:20 2017

@author: lenovo
"""
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
batch_size = 256
EMBEDDING_DIM = 100
VALIDATION_SPLIT= 0.8
MAX_SEQUENCE_LENGTH= 140

f0 = open('D:\\lmqnlp\\all_dataSet_split.txt.vec','r') #词向量文件
f1 = open('label1_2tag.txt','r') 
f2 = open('D:\\lmqnlp\\FXGnew.txt','r')

#################################
#   得到基于所有评论的词典mapping   #
#################################
#encoding=utf-8
import sys;
reload(sys);
sys.setdefaultencoding("utf8")

source = []
for count in range(231623) :
    source.append(f0.readline().strip('\n'))
word=[]
vec=[]#6296个list,每个list100个str类型的数据
for item in source:
    word.append((item.split(' ')[0]))
    vec.append(np.asarray(item.split(' ')[1:]))
mapping={}#必须有这个，不然会出现NameError
mapping=dict(zip(word,vec))


'''
lmq!!!!!↓ 提取数据
'''

datadd= []
for count in range (24991):#fsource num of lines 
    line1 = f2.readline()
    datad = []
    if line1.strip('\n')!= '' :
        line0 = line1.strip('\n').split('\t')
        line0.pop()
        for x in line0 :
            datad.append(x)              
        if datad!=[''] :
            datadd.append(datad)

f2.close()#必须要重新打开 否则两次redline会连续读
f2 = open('D:\\lmqnlp\\FXGnew.txt','r')
text = []
p=0
for i in range (3001):
    Line = f2.readline()
    linel = Line.strip('\n').split('\t')
    linel.pop()
    if Line.strip('\n') != '' and len(linel)<= MAX_SEQUENCE_LENGTH :
        text.append(Line)
        print Line
    
tokenizer = Tokenizer(nb_words=None)
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
            
#############
#准备词向量
#############

charsd = [] #统计所有字，跟每个字编号
for i in datadd:
    charsd.extend(i)
 
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, MAX_SEQUENCE_LENGTH)


charsd = pd.Series(charsd).value_counts()
charsd[:] = range(1, len(charsd)+1)


# prepare embedding matrix 嵌入词向量
embedding_matrix = np.zeros((len(charsd) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > len(charsd) :
        continue
    embedding_vector = mapping.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector



#################################
#   提取标签序列  #
#################################
d2 = []
for line in f1.readlines():
    d1 = []
    if line.strip('\n') != '' :        
        lali = line.strip('\n').split(',')
        lali.pop()
        for nu in lali :
            d1.append(int(nu))
        d2.append(d1)
dd = pd.DataFrame(index = range(len(datadd)))
dd['data'] = datadd
dd['label'] = d2
dd = dd[dd['data'].apply(len) <= MAX_SEQUENCE_LENGTH] #截断字数长的
print '11111111111111'
dd.index = range(len(dd))

tagd = pd.Series({0:0, 1:1, 2:2, 3:3})
# 0:句中不在目标片段的词 1：目标片段第一词 2：目标片段其余词 3：padding
'''
lmq!!!!!↑
'''

#生成适合模型输入的格式

'''
lmq!!!!!↓
'''


dd['x'] = dd['data'].apply(lambda x: np.array(list(charsd[x])+[0]*(MAX_SEQUENCE_LENGTH-len(x))))
dd['y'] = dd['label'].apply(lambda x: np.array(map(lambda y:np_utils.to_categorical(y,4), tagd[x].reshape((-1,1)))+[np.array([[0,0,0,1]])]*(MAX_SEQUENCE_LENGTH-len(x))))
ddy = dd['y']

trainXdd = np.array(list(dd['x']))
trainY0dd = np.array(list(dd['y']))
trainYdd = trainY0dd.reshape((-1,MAX_SEQUENCE_LENGTH,4))

#加入验证集
nb_validation_samples = int(VALIDATION_SPLIT * trainXdd.shape[0])
x_val = trainXdd[:-nb_validation_samples]
y_val = trainY0dd[:-nb_validation_samples].reshape((-1,MAX_SEQUENCE_LENGTH,4))

x_train_V= trainXdd[-nb_validation_samples:]
y_train_V = trainY0dd[-nb_validation_samples:].reshape((-1,MAX_SEQUENCE_LENGTH,4))
#x_train_V = trainXdd[:-nb_validation_samples]
#y_train_V = trainY0dd[:-nb_validation_samples].reshape((-1,MAX_SEQUENCE_LENGTH,4))
#
#x_val = trainXdd[-nb_validation_samples:]
#y_val = trainY0dd[-nb_validation_samples:].reshape((-1,MAX_SEQUENCE_LENGTH,4))





'''
lmq!!!!!↑
'''
#import keras.backend as K
#
#def mean_pred(y_true, y_pred):
#    return K.mean(y_pred)
#
#def false_rates(y_true, y_pred):
#    false_neg = ...
#    false_pos = ...
#    return {
#        'false_neg': false_neg,
#        'false_pos': false_pos,
#    }
#model.compile(optimizer='rmsprop',
#              loss='binary_crossentropy',
#              metrics=['accuracy', mean_pred, false_rates])
#设计模型
print('Training model.')
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')#cixl
print('Training model1.')

embedded_sequences = Embedding(len(charsd)+1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=MAX_SEQUENCE_LENGTH,trainable=False,mask_zero=True)(sequence_input)
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
##a =  model.predict_classes(x_val, batch_size=batch_size, verbose=1)
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
#    scoreA = 0

    for i in range(0,len(rr)):
#        if rr[i] == aa[i]:
#            scoreA+=1
        if rr[i] == aa[i] and aa[i]!= 0:
            tp+=1
        if rr[i] == aa[i] and aa[i]== 0:
            tn+=1
        if rr[i] != aa[i] and aa[i]== 0:
            fp+=1
        if rr[i] != aa[i] and aa[i]!= 0:
            fn+=1

    yr.append(aa)
    ar.append(rr)
accuracy = float(tp+tn)/float(tp+tn+fn+fp)
precision = float(tp)/float(tp+fp) 
recall = float(tp)/float(tp+fn) 
print 'acurracy:'
print accuracy
print 'precision:'
print precision
print 'recall'
print recall



#'''
#结果精确度评估
#比例方式
#'''
#yr = [] #真正标签
#ar = [] #预测出的标签
#acu = 0
#for count in range(0,len(a)):
#    rr = [] #记录预测出
#    aa = [] #记录真正标签
#    for aline in y_val[count]: #遍历真正的每一行
#        if aline[3] == 1 :
#            break
#        aa.append(list(aline).index(1.0))
#    for rline in a[count] :#遍历 预测的每一行
#        if len(rr) >= len(aa):
#            break
#        rline = list(rline)
#        rr.append(rline.index(max(rline)))
#    score = 0
#    for i in range(0,len(rr)):
#        if rr[i] == aa[i]:
#            score+=1
#    result = float(score)/len(rr)
#    print result  
#    
#    yr.append(aa)
#    ar.append(rr)
#    acu+=result
#print 'acurracy:'
#print float(acu)/len(a)