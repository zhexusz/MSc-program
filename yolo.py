import colorsys
import os
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from PIL import ImageDraw, ImageFont
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from nets.yolo4_tiny import yolo_body, yolo_eval
from util.utils import letterbox_image

import cv2

import Obstacle_avoidance as ob




#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
#--------------------------------------------#
class YOLO(object):
    _defaults = {
        "model_path"        : 'model_data/yolov4_tiny_weights_voc.h5',
        # "model_path"        : 'C:/Users/zhexu/Desktop/program/yolov4-tiny-tf2-master/logs/best_epoch_weights.h5',
        
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "classes_path"      : 'model_data/voc_classes.txt',
        "score"             : 0.5,
        "iou"               : 0.3,
        "eager"             : True,
        "max_boxes"         : 100,
        # 显存比较小可以使用416x416
        # 显存比较大可以使用608x608
        "model_image_size"  : (416, 416)
    }
    

    
    
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化yolo
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        if not self.eager:
            tf.compat.v1.disable_eager_execution()
            self.sess = K.get_session()
        self.generate()

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    #---------------------------------------------------#
    #   获得所有的先验框
    #---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        #---------------------------------------------------#
        #   计算先验框的数量和种类的数量
        #---------------------------------------------------#
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        #---------------------------------------------------------#
        #   载入模型
        #---------------------------------------------------------#
        self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes)
        self.yolo_model.load_weights(self.model_path)
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        
        '''
        h: x / len(self.class_names)根据x范围内的当前位置计算色调值（0 到 1 之间）。
        s: 1.表示饱和度值，所有元组都设置为 1。
        v: 1.表示数值（亮度）值，所有元组都设置为1。
        '''
        
        '''
        这一行将函数应用colorsys.hsv_to_rgb()到hsv_tuplesusing中的每个元组map()，
        并将它们转换为 RGB 元组。将每个元组解*x包为hsv_to_rgb()函数的单独参数。
        '''
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        
        '''
        该行使用map()来将每个颜色元组的 RGB 值缩放到 0-255 的范围。
        这些值乘以 255，然后转换为整数
        '''
        
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        #---------------------------------------------------------#
        #   在yolo_eval函数中，我们会对预测结果进行后处理
        #   后处理的内容包括，解码、非极大抑制、门限筛选等
        #---------------------------------------------------------#
        if self.eager:
            self.input_image_shape = Input([2,],batch_size=1)
            '''
            Input([2,])为神经网络模型创建输入层。输入形状指定为[2,]，表示输入张量的形状应为(batch_size, 2)。这里2表示输入张量沿第二维的元素数量。
            batch_size=1将批量大小设置为 1。这意味着模型在推理或训练期间期望一次以单个样本或示例的形式输入数据
            
            '''
            inputs = [*self.yolo_model.output, self.input_image_shape]
            outputs = Lambda(yolo_eval, output_shape=(1,), name='yolo_eval',
                arguments={'anchors': self.anchors, 'num_classes': len(self.class_names), 'image_shape': self.model_image_size, 
                'score_threshold': self.score, 'eager': True, 'max_boxes': self.max_boxes})(inputs)
            
            '''
            Lambda当您需要对 Keras 模型中的输入数据应用自定义操作或转换时，该层非常有用。您可以使用任何有效的 Python 函数或表达式作为层内的自定义操作Lambda来操作输入张量。
            '''

            self.yolo_model = Model([self.yolo_model.input, self.input_image_shape], outputs)
            '''
            最后，我们通过使用类指定输入和输出层来创建模型Model。
            该类Model允许您对模型执行各种操作，例如使用优化器和损失函数编译模型、使模型适合训练数据、评估模型的性能以及进行预测。
            '''
            
        else:
            self.input_image_shape = K.placeholder(shape=(2, ))
            
            self.boxes, self.scores, self.classes = yolo_eval(self.yolo_model.output, self.anchors,
                    num_classes, self.input_image_shape, max_boxes=self.max_boxes,
                    score_threshold=self.score, iou_threshold=self.iou)
 
    @tf.function
    def get_pred(self, image_data, input_image_shape):
        out_boxes, out_scores, out_classes = self.yolo_model([image_data, input_image_shape], training=False)
        return out_boxes, out_scores, out_classes

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, focus):
        # lines = ob.draw_line(image, focus)
        # -------------------------------------#
        # 物体实际距离：actual_distance   5
        # 物体实际高度：actual_high       (高度) car:1.5  people:1.7  bike:1.2
        # 物体像素高度：pixel_high        (高度) car:291  people:283 bike:212
        # 焦距focus =（pixel_high * actual_distance）/ actual_high
        # 焦距focus已知后，就可以根据像素高度测出物体实际距离，公式如下：actual_distance =（actual_high * focus）/pixel_high
        # -------------------------------------#
        # car：focus_car = (291 * 5) / 1.5 = 970
        # people：focus_people = (283 * 5) / 1.7 = 714
        # bike：focus_bike = (212 * 5) / 1.2 = 514
        # -------------------------------------#
        # 实际高度
        actual_high_car = 1.5
        actual_high_people = 1.7
        actual_high_bike = 1.2
        # 像素高度
        pixel_high_car = 0
        pixel_high_people = 0
        pixel_high_bike = 0
        # 焦距

        focus_car = 970
        #focus_people = 714
        focus_people = 100
        focus_bike = 514
        
        #focus_car = capture.get(cv2.CAP_PROP_FOCUS)
        #focus_people = capture.get(cv2.CAP_PROP_FOCUS)
        #focus_bike = capture.get(cv2.CAP_PROP_FOCUS)   
        # print("focus: %.2f" % capture.get(cv2.CAP_PROP_FOCUS))
        
        # 实际距离
        actual_distance = " "

        actual_distance_car = 0
        actual_distance_people = 0
        actual_distance_bike = 0
        
        start = timer()
        cx = 0
        cy = 0
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        new_image_size = (self.model_image_size[1],self.model_image_size[0])
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        #---------------------------------------------------------#
        #   将图像输入网络当中进行预测
        #---------------------------------------------------------#
        if self.eager:
            # 预测结果
            input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
            out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 
        else:
            # 预测结果
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        
        

        #---------------------------------------------------------#
        #   设置字体
        #---------------------------------------------------------#
        font = ImageFont.truetype(font='font/simhei.ttf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = max((image.size[0] + image.size[1]) // 300, 1)

        out_name = [0]*len(out_boxes)

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            out_name[i] = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            cy = (top+bottom)/2
            cx = (left+right)/2
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            
            # label为识别出的类型
            print(label, top, left, bottom, right)

            flag, place = ob.avoidance(top, left, bottom, right, image, focus, actual_distance_people)


            # 计算障碍物的实际距离
            if "person" in str(label):
                pixel_high_people = bottom - top
                actual_distance_people = (actual_high_people * focus_people) / pixel_high_people
                actual_distance = round(actual_distance_people,1)
                
                
                print("distance_people:", actual_distance_people)
                

                
            elif "car" in str(label):
                pixel_high_car = bottom - top
                actual_distance_car = (actual_high_car * focus_car) / pixel_high_car
                actual_distance = round(actual_distance_car,1)
                print("distance_car:", actual_distance_car)
            elif "bicycle" in str(label) or "motorbike" in str(label):
                pixel_high_bike = bottom - top
                actual_distance_bike = (actual_high_bike * focus_bike) / pixel_high_bike
                actual_distance = round(actual_distance_bike,1)
                print("distance_bike:", actual_distance_bike)
            else:
                actual_distance = "null"
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            
            # print(flag, place)
            
            if flag == True:
                if len(place) == 2:
                    print("Obstacles are %s and %s." % (place[0], place[1]))
                if len(place) == 1:
                    print("Obstacles are %s." % place[0])
            
            
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8') +"  "+"distance：" + 
                    str(actual_distance) + "m", fill="green", font=font)
            del draw

        end = timer()
        print(round(end - start),1) # 打印时间
        if len(out_boxes) != 0:
            j = np.argmax(out_scores)
            predicted_classes = out_name[j]
        else:
            predicted_classes = 'none'
        return image, predicted_classes,cx,cy

