import tensorflow as tf
from keras.layers import Layer
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
import keras.backend as K
import keras

# 自注意力层
class Self_Attention(Layer):
    def __init__(self, dropout_rate=0.0):
        super(Self_Attention, self).__init__()
        self.dropout_layer = Dropout(dropout_rate)

    def build(self, input_shape):
        self.k = input_shape[0][-1] 
        self.W_layer = Dense(self.k, activation='tanh', use_bias=True) 
        self.U_weight = self.add_weight(name='U', shape=(self.k, 1),   
                                        initializer=keras.initializers.glorot_uniform(),
                                        trainable=True)

    def call(self, inputs, **kwargs):
        input, mask = inputs 
        if K.ndim(input) != 3:
            raise ValueError("The dim of inputs is required 3 but get {}".format(K.ndim(input)))

        # 计算score
        x = self.W_layer(input)              
        score = tf.matmul(x, self.U_weight)  
        score = self.dropout_layer(score)   

        # softmax之前进行mask
        mask = tf.expand_dims(mask, axis=-1)  
        padding = tf.cast(tf.ones_like(mask)*(-2**31+1), tf.float32) 
        score = tf.where(tf.equal(mask, 0), padding, score)
        score = tf.nn.softmax(score, axis=1)  

        
        output = tf.matmul(input, score, transpose_a=True)   
        output /= self.k**0.5                               
        output = tf.squeeze(output, axis=-1)                
        return output


class Image_Text_Attention(Layer):
    def __init__(self, dropout_rate=0.0):
        super(Image_Text_Attention, self).__init__()
        self.dropout_layer = Dropout(dropout_rate)

    def build(self, input_shape):
        self.l = input_shape[1][1]   
        self.k = input_shape[1][-1] 
        self.img_layer = Dense(1, activation='tanh', use_bias=True) 
        self.seq_layer = Dense(1, activation='tanh', use_bias=True)  
        self.V_weight = self.add_weight(name='V', shape=(self.l, self.l),
                                        initializer=keras.initializers.glorot_uniform(),
                                        trainable=True)

    def call(self, inputs, **kwargs):
        image_emb, seq_emb, mask = inputs 

        # 线性映射
        p = self.img_layer(image_emb)  
        q = self.seq_layer(seq_emb)   

        # 内积+映射(计算score)
        emb = tf.matmul(p, q, transpose_b=True)   
        emb = emb + tf.transpose(q, [0, 2, 1])   
        emb = tf.matmul(emb, self.V_weight)      
        score = self.dropout_layer(emb)          

  
        mask = tf.tile(tf.expand_dims(mask, axis=1), [1, score.shape[1], 1])  
        padding = tf.cast(tf.ones_like(mask) * (-2 ** 31 + 1), tf.float32)
        score = tf.where(tf.equal(mask, 0), padding, score)
        score = tf.nn.softmax(score, axis=-1)      

       
        output = tf.matmul(score, seq_emb)   
        output /= self.k**0.5                
        return output
class VggNet(Layer):
    def __init__(self, block_nums, out_dim=1000, dropout_rate=0.0):
  
        super(VggNet, self).__init__()
        self.cnn_block1 = self.get_cnn_block(64, block_nums[0])
        self.cnn_block2 = self.get_cnn_block(128, block_nums[1])
        self.cnn_block3 = self.get_cnn_block(256, block_nums[2])
        self.cnn_block4 = self.get_cnn_block(512, block_nums[3])
        self.cnn_block5 = self.get_cnn_block(512, block_nums[4])
        self.out_block = self.get_out_block([4096, 4096], out_dim, dropout_rate)
        self.flatten = Flatten()

    # 单个卷积模块的搭建(layer_num个连续卷积加一个池化)
    def get_cnn_block(self, out_channel, layer_num):
        layer = []
        for i in range(layer_num):
            layer.append(Conv2D(filters=out_channel,
                                kernel_size=3,
                                padding='same',
                                activation='relu'))
        layer.append(MaxPool2D(pool_size=(2,2), strides=2))
        return keras.models.Sequential(layer) 
    def get_out_block(self, hidden_units, outdim, dropout_rate):
        layer = []
        for i in range(len(hidden_units)-1):
            layer.append(Dense(hidden_units[i], activation='relu'))
            layer.append(Dropout(dropout_rate))
        layer.append(Dense(outdim, activation='softmax'))
        return keras.models.Sequential(layer) #封装成一个模块

    def call(self, inputs, **kwargs):
        # 标准输入：[batchsize, 224, 224, 3]
        if K.ndim(inputs) != 4:
            raise ValueError("The dim of inputs is required 4 but get {}".format(K.ndim(inputs)))

        x = inputs
        cnn_block_list = [self.cnn_block1, self.cnn_block2, self.cnn_block3, self.cnn_block4, self.cnn_block5]

        # 卷积层
        for cnn_block in cnn_block_list:
            x = cnn_block(x)
        x = self.flatten(x)

        # 输出层
        output = self.out_block(x)
        return output


import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, GRU, Bidirectional

class VistaNet(Model):
    def __init__(self, block_nums=[2,2,3,3,3], out_dim=4096, vgg_dropout=0.0, attention_dropout=0.0, gru_units=[64, 128], class_num=3):
        super(VistaNet, self).__init__()
        self.vgg16 = VggNet(block_nums, out_dim, vgg_dropout)      
        self.word_self_attention = Self_Attention(attention_dropout)
        self.img_seq_attention = Image_Text_Attention(attention_dropout)  
        self.doc_self_attention = Self_Attention(attention_dropout) 
       
        self.BiGRU_layer1 = Bidirectional(GRU(units=gru_units[0],
                                             kernel_regularizer=keras.regularizers.l2(1e-5),
                                             recurrent_regularizer=keras.regularizers.l2(1e-5),
                                             return_sequences=True),
                                          merge_mode='concat')
        self.BiGRU_layer2 = Bidirectional(GRU(units=gru_units[1],
                                             kernel_regularizer=keras.regularizers.l2(1e-5),
                                             recurrent_regularizer=keras.regularizers.l2(1e-5),
                                             return_sequences=True),
                                          merge_mode='concat')
        self.output_layer = Dense(class_num, activation='softmax') # 任务层

    def call(self, inputs, training=None, mask=None):
    	
        image_inputs, text_inputs, mask = inputs 

        
        image_emb = self.vgg16(image_inputs)      

       
        word_emb = self.BiGRU_layer1(text_inputs) 


        input = [word_emb, mask]                  
        seq_emb = self.word_self_attention(input)  

        # 经过GRU层提取语义
        input = tf.expand_dims(seq_emb, axis=0)    
        seq_emb = self.BiGRU_layer2(input)        

        # 经过img_seq_attention得到M个文档向量doc_emb
        image_emb = tf.expand_dims(image_emb, axis=0) 
        mask = tf.argmax(mask, axis=1)             
        mask = tf.expand_dims(mask, axis=0)         
        input = [image_emb, seq_emb, mask]
        doc_emb = self.img_seq_attention(input)      

        # 经过self_attention得到最终的文档向量
        mask = tf.ones(shape=[1, doc_emb.shape[1]])  
        input = [doc_emb, mask]
        D_emb = self.doc_self_attention(input)       

		# output layer
        output = self.output_layer(D_emb)           
        return output


