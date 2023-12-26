# self.pushButton.setShortcut(_translate("MainWindow", "enter")) #设置快捷键
from PyQt5.QtWidgets import QMessageBox

from ui.login import Ui_LoginWindow
from mainWindow import MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon


class login_window(QtWidgets.QMainWindow, Ui_LoginWindow):
    def __init__(self):
        super(login_window, self).__init__()
        self.setupUi(self)  # 创建窗体对象
        self.init()
        self.admin = "root"
        self.Password = "123456"
        icon = QIcon('./Icon.png')
        self.setWindowIcon(icon)

    def init(self):
        self.pushButton.clicked.connect(self.login_button)  # 连接槽

    def login_button(self):
        if self.passwd.text() == "":
            QMessageBox.warning(self, '警告', '密码不能为空，请输入！')
            return None

        # if  self.password == self.lineEdit.text():
        if (self.passwd.text() == self.Password) and self.username.text() == self.admin:
            # Ui_Main = Open_Camera()  # 生成主窗口的实例
            # 1打开新窗口
            Ui_Main.show()
            # 2关闭本窗口
            self.close()
        else:
            QMessageBox.critical(self, '错误', '密码错误！')
            self.lineEdit.clear()
            return None


if __name__ == '__main__':
    from PyQt5 import QtCore
    import sys

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)  # 自适应分辨率
    import ui.NingxiaUniversity
    app = QtWidgets.QApplication(sys.argv)
    login_window = login_window()
    Ui_Main = MainWindow()
    login_window.show()

    sys.exit(app.exec_())