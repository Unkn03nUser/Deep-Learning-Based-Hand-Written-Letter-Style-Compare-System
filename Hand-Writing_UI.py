# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_1.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from keras.preprocessing import image
from keras.layers import Input, Dense
from keras.models import Model
from PIL import Image
from os import listdir
from os.path import isfile, join
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import numpy as np
import tensorflow as tf
import random
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

DEFAULT_DIR = "./data/"

class Deep_Learning:

    def __init__(self, epoch_value, xy, batch, repeat_value):
        self.epoch = epoch_value
        self.xy = xy
        self.batch = batch
        self.repeat_value = repeat_value

    def Autoencode(self, dir1, dir2):
        
        encoding_dim = 32
        input_img = Input(shape=(self.xy*self.xy,))
        encoded = Dense(encoding_dim, activation='relu')(input_img)
        decoded = Dense(self.xy*self.xy, activation='sigmoid')(encoded)

        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img, encoded)
        encoded_input = Input(shape=(encoding_dim,))
        decoded_layer = autoencoder.layers[-1]
        decoder = Model(encoded_input, decoded_layer(encoded_input))
        autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        Image_class = Image_Preparation()

        # =MAIN= data preparation
        imageset_tr = Image_class.Preparation(dir1)
        imageset_tr = np.asarray(imageset_tr)
        imageset_tr = imageset_tr.reshape(imageset_tr.shape[0], self.xy*self.xy).astype('float') / 255

        imageset_te = Image_class.Preparation(dir2)
        imageset_te = np.asarray(imageset_te)
        imageset_te = imageset_te.reshape(imageset_te.shape[0], self.xy*self.xy).astype('float') / 255

        # =TRAIN= Data Random Preparation
        val_loss = []
        for i in range(self.repeat_value):
            print("repeat : ", i+1, "/", self.repeat_value)
            imageset_trRandom = Image_class.Preparation_random(dir1)
            imageset_trRandom1 = np.asarray(imageset_trRandom[0])
            imageset_trRandom1 = imageset_trRandom1.reshape(imageset_trRandom1.shape[0], self.xy*self.xy).astype('float') / 255
            print(len(imageset_trRandom[0]))
            imageset_trRandom2 = np.asarray(imageset_trRandom[1])
            imageset_trRandom2 = imageset_trRandom2.reshape(imageset_trRandom2.shape[0], self.xy*self.xy).astype('float') / 255
            print(len(imageset_trRandom2))
            history_1 = autoencoder.fit(imageset_trRandom1, imageset_trRandom1, epochs=self.epoch,
                            batch_size=self.batch, shuffle=True, validation_data=(imageset_trRandom2, imageset_trRandom2))
            val_loss.append(history_1.history['val_loss'])

        history_2 = autoencoder.fit(imageset_tr, imageset_tr, epochs=self.epoch,
                        batch_size=self.batch, shuffle=True, validation_data=(imageset_te, imageset_te))
        print("\n\n\n")
        if self.repeat_value != 0:
            print("\timageset_trRandom Loss Average")
            print("\t" + str(np.average(val_loss)))
            print("\n\n\n")
        print("\tTrain/Test Data Compare Loss Average Result")
        print("\t" + str(np.average(history_2.history['val_loss'])))
        print("\n\n\n")
        encoded_imgs = encoder.predict(imageset_tr)
        decoded_imgs = decoder.predict(encoded_imgs)

class Image_Preparation:

    def Preparation(self, dir):

        file_dir = [f for f in listdir(dir) if isfile(join(dir, f))]
        file_dir.sort()
        image1 = []
        for i in range(0, len(file_dir)):
            inputImage = dir + "/" + file_dir[i]
            im = Image.open(inputImage)
            image1.append(np.array(im).tolist())
        return image1

    def Preparation_random(self, dir):

        file_dir = [f for f in listdir(dir) if isfile(join(dir, f))]
        file_dir.sort()
        image1_random = []
        file_random = []
        image2_random = []
        return_value = []
        random.shuffle(file_dir)
        file_random.append(file_dir[:round(len(file_dir)/2)])
        del file_dir[:round(len(file_dir)/2)]
        file_random.append(file_dir[:round(len(file_dir))])
        for i in range(0, len(file_random[0])):
            inputImage = dir + "/" + file_random[0][i]
            im = Image.open(inputImage)
            image1_random.append([np.array(im).tolist()])
        for i in range(0, len(file_random[1])):
            inputImage = dir + "/" + file_random[1][i]
            im = Image.open(inputImage)
            image2_random.append([np.array(im).tolist()])
        return_value.append(image1_random)
        return_value.append(image2_random)
        return return_value

class Ui_MainWindow(QtWidgets.QFileDialog, object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(449, 278)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        #Image1 Path Select
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.clicked.connect(self.ImagePath1_Clicked)
        self.pushButton.setGeometry(QtCore.QRect(20, 90, 101, 41))
        self.pushButton.setObjectName("pushButton")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(50, 60, 56, 12))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 20, 101, 31))
        font = QtGui.QFont()
        font.setFamily("02UtsukushiMincho")
        font.setPointSize(16)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(200, 20, 101, 31))
        font = QtGui.QFont()
        font.setFamily("02UtsukushiMincho")
        font.setPointSize(16)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        #Image2 Path Select
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.clicked.connect(self.ImagePath2_Clicked)
        self.pushButton_2.setGeometry(QtCore.QRect(200, 90, 101, 41))
        self.pushButton_2.setObjectName("pushButton_2")

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(230, 60, 56, 12))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(360, 20, 56, 12))
        self.label_5.setObjectName("label_5")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit.setGeometry(QtCore.QRect(330, 40, 104, 21))
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(350, 70, 71, 16))
        self.label_6.setObjectName("label_6")
        self.plainTextEdit_2 = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_2.setGeometry(QtCore.QRect(330, 90, 104, 21))
        self.plainTextEdit_2.setObjectName("plainTextEdit_2")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(340, 120, 91, 20))
        self.label_7.setObjectName("label_7")
        self.plainTextEdit_3 = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_3.setGeometry(QtCore.QRect(330, 140, 104, 21))
        self.plainTextEdit_3.setObjectName("plainTextEdit_3")
        #Excute
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.clicked.connect(self.Excute_Clicked)
        self.pushButton_3.setGeometry(QtCore.QRect(80, 170, 161, 41))
        self.pushButton_3.setObjectName("pushButton_3")

        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(350, 170, 91, 20))
        self.label_10.setObjectName("label_10")
        self.plainTextEdit_4 = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_4.setGeometry(QtCore.QRect(330, 190, 104, 21))
        self.plainTextEdit_4.setObjectName("plainTextEdit_4")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(20, 140, 101, 16))
        self.label_13.setObjectName("label_13")

        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(200, 140, 101, 16))
        self.label_14.setObjectName("label_14")

        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(280, 220, 161, 20))
        self.label_15.setObjectName("label_15")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 449, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def ImagePath1_Clicked(self):
        self.file_path1 = str(QFileDialog.getExistingDirectory(self, 'Select Directory', os.path.dirname(os.path.abspath(__file__)) + "/data/Image1",QFileDialog.ShowDirsOnly))
        if self.file_path1 != "":
            self.label_13.setText("Directory selected")
            QMessageBox.about(self,"Directory Selected",self.file_path1)
            self.Model_Handler = False
        else:
            pass

    def ImagePath2_Clicked(self):
        self.file_path2 = str(QFileDialog.getExistingDirectory(self, 'Select Directory', os.path.dirname(os.path.abspath(__file__)) + "/data/Image2",QFileDialog.ShowDirsOnly))
        if self.file_path2 != "":
            self.label_14.setText("Directory selected")
            QMessageBox.about(self,"Directory Selected", self.file_path2)
        else:
            pass

    def Excute_Clicked(self):
        err = self.Exeption_()
        if err == True:
            pass
        if err == False:
            return
        self.epoch_value = int(self.plainTextEdit.toPlainText())
        self.batch_size = int(self.plainTextEdit_2.toPlainText())
        self.repeat_value = int(self.plainTextEdit_3.toPlainText())
        self.xy = int(self.plainTextEdit_4.toPlainText())
        DLearning = Deep_Learning(self.epoch_value, self.xy, self.batch_size, self.repeat_value)
        DLearning.Autoencode(self.file_path1, self.file_path2)
        self.pushButton_3.setText("EXCUTE")
    
    def Exeption_(self):

        if self.plainTextEdit.toPlainText() == "" or self.plainTextEdit.toPlainText() == 0:
            QMessageBox.about(self,"Error", "Check Epoch_Value TextBox")
            return False
        elif self.plainTextEdit_2.toPlainText() == "" or self.plainTextEdit_2.toPlainText() ==  0:
            QMessageBox.about(self,"Error", "Check Batch_Size TextBox")
            return False
        elif self.plainTextEdit_3.toPlainText() == "":
            QMessageBox.about(self,"Error", "Check Repeat_Value TextBox")
            return False
        elif self.plainTextEdit_4.toPlainText() == "" or self.plainTextEdit_4.toPlainText() == 0:
            QMessageBox.about(self,"Error", "Check X, Y Length TextBox")
            return False
        try:
            if self.file_path1 == "" or self.file_path2 == "":
                QMessageBox.about(self,"Error", "Please Check Train/Test Data Path")
                return False
        except AttributeError:
            QMessageBox.about(self,"Error", "Please Check Train/Test Data Path")
            return False
        if self.plainTextEdit.toPlainText().isdigit() == False or int(self.plainTextEdit.toPlainText()) < 1:
            QMessageBox.about(self,"Error", "Epoch_value 에 0 보다 큰 정수를 입력해주세요")
            return False
        elif self.plainTextEdit_2.toPlainText().isdigit() == False or int(self.plainTextEdit_2.toPlainText()) < 1:
            QMessageBox.about(self,"Error", "Batch_size 에 0 보다 큰 정수를 입력해주세요")
            return False
        elif self.plainTextEdit_3.toPlainText().isdigit() == False or int(self.plainTextEdit_3.toPlainText()) < 0:
            QMessageBox.about(self,"Error", "Repeat_Value 에 -1 보다 큰 정수를 입력해주세요")
            return False
        elif self.plainTextEdit_4.toPlainText().isdigit() == False or int(self.plainTextEdit_4.toPlainText()) < 1:
            QMessageBox.about(self,"Error", "X, Y Length 에 0 보다 큰 정수를 입력해주세요")
            return False
        return True
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Hand-Writing Style Compare System"))
        self.pushButton.setText(_translate("MainWindow", "Path"))
        self.label.setText(_translate("MainWindow", "Image1"))
        self.label_2.setText(_translate("MainWindow", "Train data"))
        self.label_3.setText(_translate("MainWindow", "Test Data"))
        self.pushButton_2.setText(_translate("MainWindow", "Path"))
        self.label_4.setText(_translate("MainWindow", "Image2"))
        self.label_5.setText(_translate("MainWindow", "Epochs"))
        self.plainTextEdit.setPlainText(_translate("MainWindow", "1000"))
        self.label_6.setText(_translate("MainWindow", "Batch_Size"))
        self.plainTextEdit_2.setPlainText(_translate("MainWindow", "256"))
        self.label_7.setText(_translate("MainWindow", "Repeat_Value"))
        self.plainTextEdit_3.setPlainText(_translate("MainWindow", "5"))
        self.pushButton_3.setText(_translate("MainWindow", "EXCUTE"))
        self.label_10.setText(_translate("MainWindow", "X, Y Length"))
        self.plainTextEdit_4.setPlainText(_translate("MainWindow", "141"))
        self.label_13.setText(_translate("MainWindow", ""))
        self.label_14.setText(_translate("MainWindow", ""))
        self.label_15.setText(_translate("MainWindow", "Made By Jae-sung Ver. 1.0"))
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())