import sys

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QApplication

from ui.main import Ui_MainWindow
from method import Method
from defectClassificaiton import defect_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow, Method):
    def __init__(self):
        # 从文件中加载UI定义
        super().__init__()
        self.setupUi(self)

        # 初始化成员变量
        self.findImg = False
        self.path = ""
        self.saveImg = None

        # 禁用窗口大小变换
        # self.setWindowFlags(QtCore.Qt.WindowCloseButtonHint)
        # self.setFixedSize(self.width(), self.height())
        # 设置图标
        icon = QIcon('./Icon.png')
        self.setWindowIcon(icon)

        # 方法绑定
        # 文件
        self.action_open.triggered.connect(self.open_img)
        self.action_save.triggered.connect(self.save_img)
        self.action_exit.triggered.connect(self.exit)

        self.startProgramButton.clicked.connect(self.startprogram)
        self.exitProgram.clicked.connect(self.exit)
        self.browseButton.clicked.connect(self.open_img)
        self.removeButton.clicked.connect(self.removeImg)
        self.comboBox.currentTextChanged.connect(self.ImageEnhancement)

        self.defectWindow = defect_MainWindow()
        self.defect_classification.clicked.connect(lambda:(self.defectWindow.show()))


if __name__ == '__main__':
    from PyQt5 import QtCore
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)  # 自适应分辨率
    app = QApplication(sys.argv)

    window = MainWindow()

    window.show()

    sys.exit(app.exec_())
