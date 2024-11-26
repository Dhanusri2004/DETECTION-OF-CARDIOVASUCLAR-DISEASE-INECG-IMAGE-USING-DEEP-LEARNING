import tensorflow as tf
import numpy as np

from tkinter import *
import os
from tkinter import filedialog
import cv2
import argparse, sys, os
import time
from matplotlib import pyplot as plt
from tkinter import messagebox


def endprogram():
    print("\nProgram terminated!")
    sys.exit()


def file_sucess():
    global file_success_screen
    file_success_screen = Toplevel(training_screen)
    file_success_screen.title("File Upload Success")
    file_success_screen.geometry("150x100")
    file_success_screen.configure(bg='pink')
    Label(file_success_screen, text="File Upload Success").pack()
    Button(file_success_screen, text='''ok''', font=(
        'Verdana', 15), height="2", width="30").pack()


global ttype


def training():
    global training_screen

    global clicked

    training_screen = Toplevel(main_screen)
    training_screen.title("Training")
    # login_screen.geometry("400x300")
    training_screen.geometry("600x450+650+150")
    training_screen.minsize(120, 1)
    training_screen.maxsize(1604, 881)
    training_screen.resizable(1, 1)
    training_screen.configure()
    # login_screen.title("New Toplevel")

    Label(training_screen, text='''Upload Image ''', background="#d9d9d9", disabledforeground="#a3a3a3",
          foreground="#000000", width="300", height="2", font=("Calibri", 16)).pack()
    Label(training_screen, text="").pack()

    options = [
        'ImpureOvarionCyst', 'NormalPelvic', 'OvarianCyst'
    ]

    # datatype of menu text
    clicked = StringVar()

    # initial menu text
    clicked.set("select")

    # Create Dropdown menu
    drop = OptionMenu(training_screen, clicked, *options)
    drop.config(width="30")

    drop.pack()

    ttype = clicked.get()

    Button(training_screen, text='''Upload Image''', font=(
        'Verdana', 15), height="2", width="30", command=imgtraining).pack()


def Densenet():
    import  DenseModel  as eff



def imgtraining():
    name1 = clicked.get()

    print(name1)

    import_file_path = filedialog.askopenfilename()
    import os
    s = import_file_path
    os.path.split(s)
    os.path.split(s)[1]
    splname = os.path.split(s)[1]

    image = cv2.imread(import_file_path)
    # filename = 'Test.jpg'
    filename = 'Dataset/Train/' + name1 + '/' + splname

    cv2.imwrite(filename, image)
    print("After saving image:")

    fnm = os.path.basename(import_file_path)
    print(os.path.basename(import_file_path))


    print("\n*********************\nImage : " + fnm + "\n*********************")
    img = cv2.imread(import_file_path)
    if img is None:
        print('no data')

    img1 = cv2.imread(import_file_path)
    print(img.shape)
    img = cv2.resize(img, ((int)(img.shape[1] / 5), (int)(img.shape[0] / 5)))
    original = img.copy()
    neworiginal = img.copy()
    cv2.imshow('original', img1)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    img1S = cv2.resize(img1, (960, 540))

    cv2.imshow('Original image', img1S)

    dst = cv2.fastNlMeansDenoisingColored(img1, None, 10, 10, 7, 21)
    cv2.imshow("Nosie Removal", dst)


def main_account_screen():
    global main_screen
    main_screen = Tk()
    width = 600
    height = 600
    screen_width = main_screen.winfo_screenwidth()
    screen_height = main_screen.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    main_screen.geometry("%dx%d+%d+%d" % (width, height, x, y))
    main_screen.resizable(0, 0)
    # main_screen.geometry("300x250")
    main_screen.configure()
    main_screen.title("Heart-ECG Disease Prediction")

    Label(text="Heart-ECG Disease Prediction", width="300", height="5", font=("Calibri", 16)).pack()


    Button(text="Densenet", font=(
        'Verdana', 15), height="2", width="30", command=Densenet, highlightcolor="black").pack(side=TOP)

    Label(text="").pack()


    main_screen.mainloop()


main_account_screen()
