import os
import sys
sys.path.append(os.path.abspath('...') + '//Python')
import tkinter as tk
import time
import random
import msvcrt
import numpy as np
import sys, subprocess
from datetime import datetime
import matplotlib.pyplot as plt
from timeit import default_timer as timer
sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')
import keysightSD1 as key

# IMPORT EXPERIMENTS BELOW...
import testerDeleteMe
import RelaxationDemo
import RabiOscDemo
import RamseyDemo
import SpinEchoDemo
import RandomizedBenchmarkingDemo

# Functions



from tkinter import Tk, Label, Button

class window:
    def __init__(self, master):
        self.master = master
        master.title("Quantum Reference Designs")

        self.label = Label(master, text="Select the Experiment you would like to run...")
        self.label.pack()

        self.test_button = Button(master, text="Test", command=self.runTest)
        self.test_button.pack()

        self.relaxation_button = Button(master, text="Run Energy Relaxation Demo", command=self.runRelaxation)
        self.relaxation_button.pack()

        self.rabi_button = Button(master, text="Run Rabi Oscillation Demo", command=self.runRabiOsc)
        self.rabi_button.pack()

        self.ramsey_button = Button(master, text="Run Ramsey Demo", command=self.runRamsey)
        self.ramsey_button.pack()   

        self.spinEcho_button = Button(master, text="Run Spin Echo Demo", command=self.runSpinEcho)
        self.spinEcho_button.pack()

        self.randomizedBench_button = Button(master, text="Run Randomized Benchmarking Demo", command=self.runRandomizedBenchmarking)
        self.randomizedBench_button.pack()                             

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

    def runTest(self):
        os.system("cls")
        print("Running Test")
        self.runThis("testerDeleteMe")

    def runRelaxation(self):
        os.system("cls")
        self.runThis("RelaxationDemo")

    def runRabiOsc(self):
        os.system("cls")
        self.runThis("RabiOscDemo")

    def runRamsey(self):
        os.system("cls")
        self.runThis("RamseyDemo")

    def runSpinEcho(self):
        os.system("cls")
        self.runThis("SpinEchoDemo")

    def runRandomizedBenchmarking(self):
        os.system("cls")
        self.runThis("RandomizedBenchmarkingDemo")

    def runThis(self, thisFunction):
        os.system("py Python/" + thisFunction + ".py")
    	# os.system("cls")

root = Tk()
my_gui = window(root)
root.mainloop()