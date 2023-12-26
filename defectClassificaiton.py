import sys
from argparse import ArgumentParser

from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog, QGraphicsScene, QGraphicsPixmapItem
from ui.defect_classification import Ui_defect_classification
import os
import csv
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torchvision import transforms, datasets
import seaborn as sns, pandas as pd
import matplotlib.pyplot as plt
from preclassification.data_loader import BinaryClassificationDataset
from torch.utils.data import DataLoader
from multitask.iterate_results_dir import iterateResulsDirs


class defect_MainWindow(QMainWindow, Ui_defect_classification):
    def __init__(self):
        # 从文件中加载UI定义
        super().__init__()
        self.setupUi(self)
        # 设置图标
        icon = QIcon('./Icon.png')
        self.setWindowIcon(icon)

        # 方法绑定
        self.backMainWindow.clicked.connect(lambda: self.close())
        self.selectImage.clicked.connect(self.selectImages)
        self.preclassify.clicked.connect(self.pre_classify)
        self.multi_task.clicked.connect(self.multitaskclassify)
        self.saveResult.clicked.connect(self.saveresult)
        self.create_table.clicked.connect(self.createTable)
        self.exitSystem.clicked.connect(self.exitSystemMethod)
        self.lastImg.clicked.connect(self.selectlastImg)
        self.nextImg.clicked.connect(self.selectnextImg)

        # 变量定义
        self.array_of_img = []  # 读取图像路径
        self.multi_resu = []
        self.find_imgs = False  # 是否成功读取图像
        self.currentImg = None  # 当前显示图像
        self.isPre_classify = False # 是否已经进行预分类
        self.ismul_classify = False # 是否已经进行多任务分类
        self.defectsImages = 0 # 缺陷图像数量
        self.defectsStatics = [] # 缺陷分类总计
        self.waterStatics = [] # 水位分类总计

    # def getRootPath(self):
    #     curPath = os.path.abspath(os.path.dirname(__file__))
    #     print(curPath)
    #     rootPath = curPath[:curPath.find("DefectClassificationSystem")]
    #     print(rootPath)
    #     return rootPath

    def showImg(self):
        if self.currentImg:
            self.graphicsView.scene_img = QGraphicsScene()
            imgShow = QPixmap()
            imgShow.load(self.currentImg)
            imgShowItem = QGraphicsPixmapItem()
            imgShowItem.setPixmap(QPixmap(imgShow))
            self.graphicsView.scene_img.addItem(imgShowItem)
            self.graphicsView.setScene(self.graphicsView.scene_img)

    def selectImages(self):
        filenames, _ = QFileDialog.getOpenFileNames(self.centralwidget, '选择图片', './data/',
                                                    'ALL(*.*);;Images(*.png *.jpg)')
        if len(filenames) > 0:
            self.array_of_img = filenames
            self.find_imgs = True
            self.currentImg = self.array_of_img[0]
            self.showImg()
            self.showResults()

    def showResults(self, filepath=None):
        if self.find_imgs:
            self.resultShow.clear()
            self.find_imgs = False
            self.resultShow.append("选择图片：")
            for i in self.array_of_img:
                self.resultShow.append(i)
        elif self.isPre_classify:
            self.resultShow.clear()
            self.isPre_classify = False
            self.read_csv(filepath)
        elif self.ismul_classify:
            self.resultShow.clear()
            self.ismul_classify = False
            self.resultShow.append('Images' + ',' + '分类结果')
            for i in range(len(self.array_of_img)):
                # self.resultShow.append(str([self.array_of_img[i].split('/')[-1], self.multi_resu[i]]))
                # self.resultShow.append(str(self.array_of_img[i].split('/')[-1]) + ',' + str(self.multi_resu[i]))
                self.resultShow.append(self.multi_resu[i])

    def pre_classify(self):
        QMessageBox.information(self, "提示", "请选择预分类结果保存路径!", QMessageBox.Yes)

        savefilepath = QFileDialog.getExistingDirectory(self, "请选择文件夹路径", "./results")
        if savefilepath:
            device = torch.device("cpu")
            model = torch.load("model/pre-model/best_val_f1.pth", map_location="cpu")
            BATCH_SIZE = 256
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            valid_data = BinaryClassificationDataset(imgPaths=self.array_of_img, transform=transform)
            valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
            self.resultShow.clear()
            self.resultShow.append("预分类中...")
            with torch.no_grad():
                for images, imagePaths in valid_loader:
                    outputs = model(images.to(device))
                    _, predicted = torch.max(outputs, 1)
                    predicted = predicted.cpu()
                    imagePaths = list(imagePaths)
                    predicted = list(predicted.numpy())
                    # df = pd.DataFrame({
                    #     "Filename": imagePaths,
                    #     "ND": predicted
                    # })
                    # if os.path.exists(os.path.join(savefilepath, 'pred_results.csv')):
                    #     df = pd.DataFrame({
                    #         "Filename": imagePaths,
                    #         "ND": predicted
                    #     })
                    #     df.to_csv(os.path.join(savefilepath, 'pred_results.csv'), mode='a', columns=['Filename', 'ND'], header=False, index=False)
                    # else:
                    df = pd.DataFrame({
                        "Filename": imagePaths,
                        "ND": predicted
                    })
                    df.to_csv(os.path.join(savefilepath, 'pred_results.csv'), mode='w', index=False)

            self.isPre_classify = True
            QMessageBox.information(self, "提示", "预分类结束!", QMessageBox.Yes)
            self.resultShow.clear()
            self.showResults(filepath=os.path.join(savefilepath, 'pred_results.csv'))
        else:
            QMessageBox.information(self, "提示", "没有选择正确路径，退出程序!", QMessageBox.Yes)
            sys.exit()

    def read_csv(self, filepath):
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # s = str(row[0].split('/')[-1]) + ',' + str(row[1])
                # self.resultShow.append(s)
                self.resultShow.append(str(row[0].split('/')[-1]) + ',' + str(row[1]))
                # self.resultShow.append(str([row[0].split('/')[-1], row[1]]))

    def multitaskclassify(self):
        self.resultShow.clear()
        QMessageBox.information(self, "提示", "多任务分类开始!", QMessageBox.Yes)

        parser = ArgumentParser()
        parser.add_argument('--conda_env', type=str, default='qh')
        parser.add_argument('--ann_root', type=str, default='./annotations')
        parser.add_argument('--data_root', type=str, default=self.array_of_img)
        parser.add_argument('--batch_size', type=int, default=256, help="Size of the batch per GPU")
        parser.add_argument('--workers', type=int, default=0)
        parser.add_argument("--results_output", type=str, default="./results")
        parser.add_argument("--log_input", type=str, default='./model/multitask')
        parser.add_argument("--split", type=str, default="Valid", choices=["Train", "Valid", "Test"])
        parser.add_argument("--best_weights", action="store_true",
                            help="If true 'model_path' leads to a specific weight file. If False it leads to the output folder of lightning_trainer where the last.ckpt file is used to read the best model weights.")
        parser.add_argument("--inferce", action="store_true", help="If true, two-stage classification.")
        args = vars(parser.parse_args())

        iterateResulsDirs(args)
        QMessageBox.information(self, "提示", "多任务分类结束!", QMessageBox.Yes)
        self.ismul_classify = True
        self.multilabel_sewerml_evaluation()
        self.showResults()

    def multilabel_sewerml_evaluation(self, threshold=0.5):
        DefectLabels = ["RB","OB","PF","DE","FS","IS","RO","IN","AF","BE","FO","GR","PH","PB","OS","OP","OK"]
        WaterLabels = ["0%<5%","5-15%","15-30%","30%<="]
        pre_csv_path = 'results/pred_results.csv'
        defect_csv_path = 'results/resnet50_ResNetBackbone_CTGNN_GCN_Fixed-Effective-MTL-version_1/resnet50_ResNetBackbone_CTGNN_GCN_Fixed-Effective-MTL-version_1_defect_valid_sigmoid.csv'
        water_csv_path = 'results/resnet50_ResNetBackbone_CTGNN_GCN_Fixed-Effective-MTL-version_1/resnet50_ResNetBackbone_CTGNN_GCN_Fixed-Effective-MTL-version_1_water_valid_sigmoid.csv'
        pre_file = pd.read_csv(pre_csv_path)
        defect_file = pd.read_csv(defect_csv_path)
        water_file = pd.read_csv(water_csv_path)
        pre_f = pre_file['Filename'].values.tolist()
        pre_v = pre_file["ND"].values.tolist()
        defects = np.where(defect_file.iloc[:, 1:] > 0.5, 1, 0).tolist()
        water = np.where(water_file.iloc[:, 1:] > 0.5, 1, 0).tolist()
        
        for i in range(len(pre_v)):
            if pre_v[i] == 1:
                self.multi_resu.append(str(pre_f[i].split('/')[-1]) + ',' + 'ND')
            else:
                defect_index = [i for i, x in enumerate(defects[i]) if x  == 1]
                de = []
                for i in defect_index:
                    de.append(DefectLabels[i])
                self.multi_resu.append(str(pre_f[i].split('/')[-1]) + ',' + str({'Defects': de, 'Water': WaterLabels[water[i].index(1)]}))

    def saveresult(self):
        QMessageBox.information(self, "提示", "请选择文件保存路径!", QMessageBox.Yes)
        openfilepath = QFileDialog.getExistingDirectory(self, "请选择文件夹路径", "./results")
        savefilepath = os.path.join(openfilepath, '分类结果.txt')
        # savefilepath = QFileDialog.getOpenFileName(self, "选取文件路径", "./results", "TXT")
        for root, dirs, files in os.walk(openfilepath):
            for file in files:
                if os.path.join(root, file) == savefilepath:
                    os.remove(savefilepath)
        # if savefilepath:
        #     os.remove(savefilepath)
        text_file = open(savefilepath, 'w', encoding='utf-8')
        str_resu = [line + '\n' for line in self.multi_resu]
        text_file.writelines('Images' + ',' + '分类结果' + '\n')
        text_file.writelines(str_resu)
        text_file.close()
        for root, dirs, files in os.walk(openfilepath):
            for file in files:
                if os.path.join(root, file) == savefilepath:
                    QMessageBox.information(self, "提示", "结果保存成功!", QMessageBox.Yes)
            # with open(os.path.join(savefilepath, '分类结果.txt'), 'w') as text_file:
            #     str_resu = [line + '\n' for line in self.multi_resu]
            #     text_file.writeliness(str_resu)
                # text_file.write(str(self.multi_resu))
        # else:
        #     QMessageBox.information(self, "提示", "没有选择正确路径，退出程序!", QMessageBox.Yes)
        #     sys.exit()

    '''
    柱状图显示标签数量
    '''
    def autolabel(self, num):
        for n in num:
            height = n.get_height()
            plt.text(n.get_x() + n.get_width()/2.-0.08, 1.02*height, '%s' % int(height), size=8, family='Times new roman')

    def createTable(self):
        DefectLabels = ["RB","OB","PF","DE","FS","IS","RO","IN","AF","BE","FO","GR","PH","PB","OS","OP","OK", "ND"]
        WaterLabels = ["0%<5%","5-15%","15-30%","30%<="]
        pre_csv_path = 'results/pred_results.csv'
        defect_csv_path = 'results/resnet50_ResNetBackbone_CTGNN_GCN_Fixed-Effective-MTL-version_1/resnet50_ResNetBackbone_CTGNN_GCN_Fixed-Effective-MTL-version_1_defect_valid_sigmoid.csv'
        water_csv_path = 'results/resnet50_ResNetBackbone_CTGNN_GCN_Fixed-Effective-MTL-version_1/resnet50_ResNetBackbone_CTGNN_GCN_Fixed-Effective-MTL-version_1_water_valid_sigmoid.csv'
        pre_file = pd.read_csv(pre_csv_path)
        defect_file = pd.read_csv(defect_csv_path)
        water_file = pd.read_csv(water_csv_path)
        pre_f = pre_file['Filename'].values.tolist()
        pre_v = pre_file["ND"].values.tolist()
        defects = np.where(defect_file.iloc[:, 1:] > 0.5, 1, 0)
        water = np.where(water_file.iloc[:, 1:] > 0.5, 1, 0)
        water_d = []
        self.defectsStatics = np.sum(defects, axis=0).tolist()
        self.defectsStatics.append(np.sum(pre_v))
        for i in range(len(pre_v)):
            if pre_v[i] == 0:
                water_d.append(water[i])
        self.waterStatics = np.sum(water_d, axis=0)
        # self.defectsImages = len(pre_v) - np.sum(pre_v)
        plt.figure('分类结果统计', figsize=(8, 4))
        # for plt_index in range(1, 3):
        #     plt.subplot(2, 1, plt_index)
        plt.subplot(221)
        defectPic = plt.bar(x=DefectLabels, height=self.defectsStatics, color='#a0c69d')
        # 关闭上、右边框
        ax = plt.gca()
        ax.spines.right.set_color('none')
        ax.spines['top'].set_visible(False)
        # 设置柱状图数量显示
        self.autolabel(defectPic)

        plt.title('defectsClassify', fontproperties='Times New Roman', size=10)
        plt.xticks(DefectLabels, rotation=90, fontproperties='Times New Roman', size=9)
        plt.xlabel('defects')
        plt.ylabel('amount')
        plt.tight_layout()

        plt.subplot(222)
        waterPic = plt.bar(x=WaterLabels, height=self.waterStatics, width=0.3, color='#ebe717')
        # 关闭上、右边框
        ax = plt.gca()
        ax.spines.right.set_color('none')
        ax.spines['top'].set_visible(False)
        # 设置柱状图数量显示
        self.autolabel(waterPic)
        plt.xticks(WaterLabels, rotation=90, fontproperties='Times New Roman', size=9)
        plt.title('waterClassify', fontproperties='Times New Roman', size=10)
        plt.xlabel('water')
        plt.ylabel('amount')
        plt.tight_layout()
        plt.savefig('./results/classifyResult.png', dpi=600, bbox_inches='tight')
        plt.show()

    def exitSystemMethod(self):
        res = QMessageBox.warning(self, "退出程序", "确定退出程序？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if res == QMessageBox.Yes:
            sys.exit()

    def selectlastImg(self):
        scene = QGraphicsScene()
        self.graphicsView.setScene(scene)
        for i in range(len(self.array_of_img)):
            if self.array_of_img[i] == self.currentImg and i == 0:
                QMessageBox.information(self, "提示", "已经是第一张图片啦!", QMessageBox.Yes)
                self.currentImg = self.array_of_img[0]
                self.showImg()
                break
            if self.array_of_img[i] == self.currentImg and i != 0:
                self.currentImg = self.array_of_img[i - 1]
                self.showImg()
                break

    def selectnextImg(self):
        scene = QGraphicsScene()
        self.graphicsView.setScene(scene)
        for i in range(len(self.array_of_img)):
            if i == len(self.array_of_img) - 1:
                QMessageBox.information(self, "提示", "已经是最后一张图片啦!", QMessageBox.Yes)
                self.currentImg = self.array_of_img[-1]
                self.showImg()
                break
            if self.array_of_img[i] == self.currentImg and i != len(self.array_of_img) - 1:
                self.currentImg = self.array_of_img[i + 1]
                self.showImg()
                break


if __name__ == '__main__':
    # import csv
    # filepath = './results/pred_results.csv'
    #
    # with open(filepath, 'r') as f:
    #     reader = csv.reader(f)
    #     file = []
    #     for row in reader:
    #         file.append([row[0].split('/')[-1], row[1]])
    #         # print(str([row[0].split('/')[-1], row[1]]))
    #     # print(file)
    #     for i in file:
    #
    #         print(str(i))
    # parser = ArgumentParser()
    # parser.add_argument('--conda_env', type=str, default='qh')
    # parser.add_argument('--ann_root', type=str, default='./annotations')
    # parser.add_argument('--data_root', type=str, default='./data')
    # parser.add_argument('--batch_size', type=int, default=256, help="Size of the batch per GPU")
    # parser.add_argument('--workers', type=int, default=0)
    # parser.add_argument("--results_output", type=str, default="./results")
    # parser.add_argument("--log_input", type=str, default='./model/multitask')
    # parser.add_argument("--split", type=str, default="Valid", choices=["Train", "Valid", "Test"])
    # parser.add_argument("--best_weights", action="store_true",
    #                     help="If true 'model_path' leads to a specific weight file. If False it leads to the output folder of lightning_trainer where the last.ckpt file is used to read the best model weights.")
    # parser.add_argument("--inferce", action="store_true", help="If true, two-stage classification.")
    # args = vars(parser.parse_args())
    # iterateResulsDirs(args)


    from PyQt5 import QtCore

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)  # 自适应分辨率
    app = QApplication(sys.argv)

    window = defect_MainWindow()
    window.show()

    sys.exit(app.exec_())
