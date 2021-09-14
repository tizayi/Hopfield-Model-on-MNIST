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
        self.model = pseudoInverseModel([3,6,5,4])
        self.fig = Figure(figsize=(4.5, 3), dpi=100)
        self.subplot1 = self.figure.add_subplot(211)
        self.axes = self.fig.add_subplot(1, 1, 1)
        self.axes.imshow(self.model.getState())
        
        self.btn = Button(win, text='sweep')
        self.btn.bind('<Button-1>', self.sweep)
        self.btn.place(x=100, y=50 + 80)

        self.btn = Button(win, text='Ramdomise')
        self.btn.bind('<Button-1>', self.random)
        self.btn.place(x=100, y=100 + 80)
        
        # Add noise
        self.btn = Button(win, text='Add noise')
        self.btn.bind('<Button-1>', self.noise)
        self.btn.place(x=100, y=150 + 80)
        
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





window = Tk()
mywin = MyWindow(window)
window.title('Hopfiled Model')
window.geometry("800x600+10+10")
window.mainloop()