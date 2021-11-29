from struct import pack
from Hopfield_model import HopfieldModel,pseudoInverseModel
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

class MyWindow:
    def __init__(self, win):
        
        self.model = HopfieldModel([3,6])
        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.axes = self.fig.add_subplot(1, 1, 1)
        self.axes.imshow(self.model.getState())
        
        self.btn = Button(win, text='Sweep-1x', height = 1, width = 10)
        self.btn.bind('<Button-1>', self.sweep)
        self.btn.place(x=10, y=50)

        self.btn = Button(win, text='Sweep-5x',height = 1, width = 10)
        self.btn.bind('<Button-1>', self.multisweep)
        self.btn.place(x=110, y=50)

        self.btn = Button(win, text='Ramdomise',height = 1, width = 10)
        self.btn.bind('<Button-1>', self.random)
        self.btn.place(x=110, y=80)
        
        # Add noise
        self.btn = Button(win, text='Add Noise',height = 1, width = 10)
        self.btn.bind('<Button-1>', self.noise)
        self.btn.place(x=10, y=80)
        
        # Radio 


        # Show the plot 
        self.plots = FigureCanvasTkAgg(self.fig, win)
        self.plots.get_tk_widget().pack(side=RIGHT, fill=BOTH, expand=0)
    
    def sweep(self,event):
        self.model.hopfieldSweep(1)
        self.axes.imshow(self.model.getState())
        self.plots.draw()
    
    def random(self,event):
        self.model.randomise()
        self.axes.imshow(self.model.getState())
        self.plots.draw()
    
    def noise(self,event):
        self.model.addNoise(0.1)
        self.axes.imshow(self.model.getState())
        self.plots.draw()
    
    def multisweep(self,event):
        self.model.hopfieldSweep(5)
        self.axes.imshow(self.model.getState())
        self.plots.draw()



window = Tk()
T = Text(window, height=2, width=30)
T.pack(side=TOP, fill=X)
T.insert(END, "The Hopfield model can store and recall patterns. This example demonstrates this using the \n MNIST digit dataset")
mywin = MyWindow(window)

window.title('Hopfiled Model')
window.geometry("800x600+10+10")
window.mainloop()