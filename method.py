import os
import random
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QGraphicsPixmapItem, QGraphicsScene, \
    QMessageBox
from defectClassificaiton import defect_MainWindow
import torch
from PIL import Image
from torchvision import transforms
from promptWindow import PromptWindow


class Method():
    # def __init__(self):
    #     self.findImg = False
    #     self.path = ""
    #     self.saveImg = None

    def find_img(self):
        filename, _ = QFileDialog.getOpenFileName(self.centralwidget, '选择图片', './data/',
                                                  'ALL(*.*);;Images(*.png *.jpg)')
        if filename:
            self.findImg = True
            self.path = filename
        return filename

    def open_img(self):
        filename = self.find_img()
        if filename:
            self.graphicsView.scene_img = QGraphicsScene()
            imgShow = QPixmap()
            imgShow.load(filename)
            imgShowItem = QGraphicsPixmapItem()
            imgShowItem.setPixmap(QPixmap(imgShow))
            self.graphicsView.scene_img.addItem(imgShowItem)
            self.graphicsView.setScene(self.graphicsView.scene_img)

    def show_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img.shape[1]
        y = img.shape[0]
        frame = QImage(img, x, y, x * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()
        scene.addItem(item)
        self.graphicsView_2.setScene(scene)

        # self.graphicsView_2.scene_img = QGraphicsScene()
        # imgShow = QPixmap(img)
        # # imgShow.load(img)
        # imgShowItem = QGraphicsPixmapItem()
        # imgShowItem.setPixmap(imgShow)
        # self.graphicsView_2.scene_img.addItem(imgShowItem)
        # self.graphicsView_2.setScene(self.graphicsView_2.scene_img)

    def save_img(self):
        file_path = QFileDialog.getSaveFileName(self, '选择保存位置', 'C:/Users/Lance Song/Pictures/*.png',
                                                'Image files(*.png)')
        file_path = file_path[0]
        if file_path:
            print('file_path: ', file_path)
            cv2.imwrite(file_path, self.saveImg)

    def exit(self):
        res = QMessageBox.warning(self, "退出程序", "确定退出程序？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if res == QMessageBox.Yes:
            sys.exit()

    def gray(self, img=None):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            if img.ndim > 2:
                b = img[:, :, 0].copy()
                g = img[:, :, 1].copy()
                r = img[:, :, 2].copy()
                img = 0.2126 * r + 0.7152 * g + 0.0722 * b
            self.saveImg = img.astype(np.uint8)
            self.show_img(self.saveImg)
            return img

    def SaturationAdjustment(self, image=None):
        image = cv2.imread(self.path)
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)
        v1 = np.clip(cv2.add(1 * v, 30), 0, 255)
        hsv_img = np.uint8(cv2.merge((h, s, v1)))
        image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.saveImg = image.astype(np.uint8)
        self.show_img(self.saveImg)
        return image

    def BrightnessAdjustment(self, image=None):
        image = cv2.imread(self.path)
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)
        v2 = np.clip(cv2.add(2 * v, 20), 0, 255)
        image = np.uint8(cv2.merge((h, s, v2)))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.saveImg = image.astype(np.uint8)
        self.show_img(self.saveImg)
        return image

    def cv2file(self, img):
        cv2.imwrite('./tmp.png', img)
        tmp = cv2.imread('./tmp.png')
        os.remove('./tmp.png')
        return tmp

    def startprogram(self):
        if not self.path:
            QMessageBox.warning(self, "提示", "请选择要处理的图片！", QMessageBox.Close)
        else:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            self.saveImg = img
            self.show_img(img)

    def removeImg(self):
        res = QMessageBox.warning(self, "移除确认", "是否确认移除当前图片", QMessageBox.Yes | QMessageBox.No,
                                  QMessageBox.No)
        if res == QMessageBox.Yes:
            self.saveImg = None
            scene = QGraphicsScene()
            self.graphicsView.setScene(scene)
            self.graphicsView_2.setScene(scene)

    def ImageEnhancement(self):
        selection = self.comboBox.currentText()
        if selection == "灰度化":
            self.gray()
        elif selection == "亮度抖动":
            self.SaturationAdjustment()
        elif selection == "色彩抖动":
            self.BrightnessAdjustment()



if __name__ == '__main__':
    pass
    # from defectClassificaiton import defect_MainWindow


