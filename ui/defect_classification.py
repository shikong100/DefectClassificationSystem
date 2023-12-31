# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'defect_classification.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_defect_classification(object):
    def setupUi(self, defect_classification):
        defect_classification.setObjectName("defect_classification")
        defect_classification.resize(614, 592)
        defect_classification.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.centralwidget = QtWidgets.QWidget(defect_classification)
        self.centralwidget.setObjectName("centralwidget")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(200, 40, 361, 301))
        self.graphicsView.setObjectName("graphicsView")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 40, 91, 41))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.pre_classification = QtWidgets.QLabel(self.centralwidget)
        self.pre_classification.setGeometry(QtCore.QRect(10, 40, 121, 281))
        self.pre_classification.setStyleSheet("border: 1px solid black;")
        self.pre_classification.setText("")
        self.pre_classification.setObjectName("pre_classification")
        self.preclassify = QtWidgets.QPushButton(self.centralwidget)
        self.preclassify.setGeometry(QtCore.QRect(40, 150, 75, 23))
        self.preclassify.setStyleSheet("QPushButton{background:#C0C0C0;border-radius:5px;}QPushButton:hover{background:#FFD700;}")
        self.preclassify.setObjectName("preclassify")
        self.multi_task = QtWidgets.QPushButton(self.centralwidget)
        self.multi_task.setGeometry(QtCore.QRect(40, 180, 75, 23))
        self.multi_task.setStyleSheet("QPushButton{background:#C0C0C0;border-radius:5px;}QPushButton:hover{background:#FFD700;}")
        self.multi_task.setObjectName("multi_task")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(50, 350, 81, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.saveResult = QtWidgets.QPushButton(self.centralwidget)
        self.saveResult.setGeometry(QtCore.QRect(40, 210, 75, 23))
        self.saveResult.setStyleSheet("QPushButton{background:#C0C0C0;border-radius:5px;}QPushButton:hover{background:#FFD700;}")
        self.saveResult.setObjectName("saveResult")
        self.create_table = QtWidgets.QPushButton(self.centralwidget)
        self.create_table.setGeometry(QtCore.QRect(40, 240, 75, 23))
        self.create_table.setStyleSheet("QPushButton{background:#C0C0C0;border-radius:5px;}QPushButton:hover{background:#FFD700;}")
        self.create_table.setObjectName("create_table")
        self.backMainWindow = QtWidgets.QPushButton(self.centralwidget)
        self.backMainWindow.setGeometry(QtCore.QRect(40, 80, 75, 23))
        self.backMainWindow.setStyleSheet("QPushButton{background:#008080;border-radius:5px;}QPushButton:hover{background:green;}")
        self.backMainWindow.setObjectName("backMainWindow")
        self.exitSystem = QtWidgets.QPushButton(self.centralwidget)
        self.exitSystem.setGeometry(QtCore.QRect(40, 290, 75, 23))
        self.exitSystem.setStyleSheet("QPushButton{background:#F7D674;border-radius:5px;}\n"
"QPushButton:hover{background:red;}")
        self.exitSystem.setObjectName("exitSystem")
        self.selectImage = QtWidgets.QPushButton(self.centralwidget)
        self.selectImage.setGeometry(QtCore.QRect(40, 120, 75, 23))
        self.selectImage.setStyleSheet("QPushButton{background:#C0C0C0;border-radius:5px;}QPushButton:hover{background:#FFD700;}")
        self.selectImage.setObjectName("selectImage")
        self.lastImg = QtWidgets.QPushButton(self.centralwidget)
        self.lastImg.setGeometry(QtCore.QRect(200, 350, 75, 23))
        self.lastImg.setStyleSheet("QPushButton{background:#FFFACD;border-radius:5px;}QPushButton:hover{background:#D2691E;}")
        self.lastImg.setObjectName("lastImg")
        self.nextImg = QtWidgets.QPushButton(self.centralwidget)
        self.nextImg.setGeometry(QtCore.QRect(480, 350, 75, 23))
        self.nextImg.setStyleSheet("QPushButton{background:#FFFACD;border-radius:5px;}QPushButton:hover{background:#D2691E;}")
        self.nextImg.setObjectName("nextImg")
        self.resultShow = QtWidgets.QTextEdit(self.centralwidget)
        self.resultShow.setGeometry(QtCore.QRect(50, 380, 511, 151))
        self.resultShow.setStyleSheet("background-color:rgb(211,211,211);")
        self.resultShow.setObjectName("resultShow")
        defect_classification.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(defect_classification)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 614, 22))
        self.menubar.setObjectName("menubar")
        defect_classification.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(defect_classification)
        self.statusbar.setObjectName("statusbar")
        defect_classification.setStatusBar(self.statusbar)

        self.retranslateUi(defect_classification)
        QtCore.QMetaObject.connectSlotsByName(defect_classification)

    def retranslateUi(self, defect_classification):
        _translate = QtCore.QCoreApplication.translate
        defect_classification.setWindowTitle(_translate("defect_classification", "缺陷分类"))
        self.label.setText(_translate("defect_classification", "功能选择"))
        self.preclassify.setText(_translate("defect_classification", "预分类"))
        self.multi_task.setText(_translate("defect_classification", "多任务分类"))
        self.label_3.setText(_translate("defect_classification", "分类结果"))
        self.saveResult.setText(_translate("defect_classification", "保存结果"))
        self.create_table.setText(_translate("defect_classification", "统计结果"))
        self.backMainWindow.setText(_translate("defect_classification", "返回主页面"))
        self.exitSystem.setText(_translate("defect_classification", "退出系统"))
        self.selectImage.setText(_translate("defect_classification", "选择图片"))
        self.lastImg.setText(_translate("defect_classification", "上一张"))
        self.nextImg.setText(_translate("defect_classification", "下一张"))
