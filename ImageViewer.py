# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\- Data Kampus\- SMSTR 6\TA 1\Aplikasi\ImageViewer.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_imageViewer(object):
    def setupUi(self, imageViewer):
        imageViewer.setObjectName("imageViewer")
        imageViewer.resize(862, 502)
        imageViewer.setStyleSheet("background-color: rgb(48, 48, 48);")
        self.centralwidget = QtWidgets.QWidget(imageViewer)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 640, 480))
        self.label.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label.setText("")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.le_path = QtWidgets.QLineEdit(self.centralwidget)
        self.le_path.setGeometry(QtCore.QRect(660, 440, 161, 20))
        self.le_path.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.le_path.setObjectName("le_path")
        self.tb_path = QtWidgets.QToolButton(self.centralwidget)
        self.tb_path.setGeometry(QtCore.QRect(830, 440, 25, 19))
        self.tb_path.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.tb_path.setObjectName("tb_path")
        self.tb_file = QtWidgets.QTextBrowser(self.centralwidget)
        self.tb_file.setGeometry(QtCore.QRect(660, 10, 191, 421))
        self.tb_file.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.tb_file.setObjectName("tb_file")
        self.pb_back = QtWidgets.QPushButton(self.centralwidget)
        self.pb_back.setGeometry(QtCore.QRect(660, 470, 91, 23))
        self.pb_back.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.pb_back.setObjectName("pb_back")
        self.pb_next = QtWidgets.QPushButton(self.centralwidget)
        self.pb_next.setGeometry(QtCore.QRect(760, 470, 91, 23))
        self.pb_next.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.pb_next.setObjectName("pb_next")
        imageViewer.setCentralWidget(self.centralwidget)

        self.retranslateUi(imageViewer)
        QtCore.QMetaObject.connectSlotsByName(imageViewer)

    def retranslateUi(self, imageViewer):
        _translate = QtCore.QCoreApplication.translate
        imageViewer.setWindowTitle(_translate("imageViewer", "Game Rehabilitasi"))
        self.tb_path.setText(_translate("imageViewer", "..."))
        self.pb_back.setText(_translate("imageViewer", "<"))
        self.pb_next.setText(_translate("imageViewer", ">"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    imageViewer = QtWidgets.QMainWindow()
    ui = Ui_imageViewer()
    ui.setupUi(imageViewer)
    imageViewer.show()
    sys.exit(app.exec_())
