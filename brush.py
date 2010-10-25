#!/usr/bin/env python
"""
brush
  Plots multiple scatter plots of a record array with many columns
  and allows the user to brush data in a given subplot window, showing what
  effect it has in other plot windows (by highlighting the same records 
  in those other subplots)
   
  Press "d" to remove that brush region and restore the previous opacity.
    
Written by Josh Bloom (jbloom@astro.berkeley.edu) for the UC Berkeley Python Seminar class
  homework #4 (AY250, Fall 2010)
"""

import matplotlib.pylab as plt
import numpy as np
import os
import datetime

from matplotlib.patches import Rectangle

class Brusher:
    
    def __init__(self,figure,axis_info,data,opac=4.0):
        """
        figure: mpl figure to brush. Should have been rendered already
        axis_info: list of axes along with their record name
        data: the record array
        opac: how much to make non-brushed point more transparent. Larger will make them more dim.
        """
        self.fig = figure
        self.data = data
        self.axis_info = axis_info
        self.start_xy = (None,None)
        self.mouse_is_down = False
        self.list_of_brushes = []
        self.opac = opac
        
        # register all the events
        self.down_id = self.fig.canvas.mpl_connect('button_press_event', self.down_mouse)
        self.up_id = self.fig.canvas.mpl_connect('button_release_event', self.up_mouse)
        self.key_id = self.fig.canvas.mpl_connect('key_press_event', self.key_press)
        self.motion_id = self.fig.canvas.mpl_connect('motion_notify_event', self.motion)
                    
    def _brush(self,event,region,inverse=False):
        """
        This will loop through all the other subplots (without the brush region)
        and change the opacity of the points not associated with that region.
        
        when inverse is True, it will "unbrush" by resetting the opacity of the brushed points
        """
        opacity_fraction = self.opac
        # what variables are in the plot?
        plot_vars = [x[1] for x in self.axis_info if x[0] == event.inaxes][0]
        
        ## figure out the min max of this region
        minx, miny   = region.get_xy()
        maxx  = minx + region.get_width()
        maxy =  miny + region.get_height()
        
        ## now query the data to get all the sources that are inside this range
        if isinstance(self.data[plot_vars[0]][0],datetime.datetime):
            maxx = datetime.datetime.fromordinal(maxx)
            minx= datetime.datetime.fromordinal(minx)
        elif isinstance(self.data[plot_vars[0]][0],datetime.date):
            maxx = datetime.date.fromordinal(maxx)
            minx= datetime.date.fromordinal(minx)
            
        if isinstance(self.data[plot_vars[1]][0],datetime.datetime):
            maxy = datetime.datetime.fromordinal(maxx)
            miny= datetime.datetime.fromordinal(minx)
        elif isinstance(self.data[plot_vars[1]][0],datetime.date):
            maxy = datetime.date.fromordinal(maxy)
            miny= datetime.date.fromordinal(miny)
        
            #inds is the reccord inside the region
        self.inds = (self.data[plot_vars[0]]<= maxx) & (self.data[plot_vars[0]] > minx) & \
               (self.data[plot_vars[1]] <= maxy) & (self.data[plot_vars[1]] > miny)
        invinds = ~ self.inds  # get all indicies of those records not inside the region
        
        for a,pv in self.axis_info:
            # dont self brush!
            if a == event.inaxes:
                continue

            ## get the scatterplot color and alpha channel data
            self.t = a.collections[0]
            fc = self.t.get_facecolor() # this will be a 2d array
            '''Here we change the color and opacity of the points
            fc[index,0] = Red
            fc[index,1] = Green
            fc[index,2] = Blue
            fc[index,3] = Alpha
            
            default is  [ 0.4   ,  0.4   ,  1.    ,  1.0]
            '''
            if not inverse: 
                fc[invinds,2] /= 20. #reduce blue channel greatly
                fc[invinds,3] /= opacity_fraction 
            else:
                fc[invinds,2] *= 20.
                fc[invinds,3] *= opacity_fraction
            self.t.set_facecolor(fc)
            
        plt.draw()
            
    def key_press(self,event):
        """
        in charge of dealing with the keyboard entry. 
        """
        if event.inaxes is None:
            return

        ## maybe delete a brush
        if event.key in ("d","D"):
            ## loop through the brush regions and delete the first region we hit, if we're in the right axis
            for b in self.list_of_brushes:
                if b[0] == event.inaxes:
                    #print "here3", (event.xdata,event.ydata)
                    #print b[1].contains(event)
                    self.r = b[1]
                    if b[1].contains(event)[0]:
                        #print "here4"
                        self.list_of_brushes.remove(b)
                        ## now we need to unbrush
                        self._brush(event,b[1],inverse=True)
                        event.inaxes.patches.remove(b[1])
                        plt.draw()                
                        break
                        
    def down_mouse(self,event):
        """
        in charge of dealing with the mouse down...this starts the creation 
        of a new rectangle
        """
        ## check to make sure we're in an axes (subplot) if not ignore
        if event.inaxes is None:
            self.mouse_is_down = False
            return
            
        # where are we?
        self.cur_mouse_axis = event.inaxes
        self.start_xy       = (event.xdata,event.ydata)
        self.mouse_is_down  = True
        
        ## let's start a rectangle - add some transparency (alpha) so we 
        ## can see through the rectangle
        rect = Rectangle( self.start_xy, 0,0,facecolor="#442222",alpha=0.21)
        event.inaxes.add_patch(rect)
        self.list_of_brushes.append( (event.inaxes,rect))
        
    def motion(self,event):
        """
        in charge of dealing with the mouse movement...this redraws the rectangle
        """
        if not self.mouse_is_down or event.inaxes is None or (self.cur_mouse_axis !=  event.inaxes):
            self.mouse_is_down = False
            self.start_xy = (None,None)
            return
       
        self.list_of_brushes[-1][1].set_width(abs(self.start_xy[0] - event.xdata))
        self.list_of_brushes[-1][1].set_height(abs(self.start_xy[1] - event.ydata))
        self.list_of_brushes[-1][1].set_x(min(self.start_xy[0],event.xdata))
        self.list_of_brushes[-1][1].set_y(min(self.start_xy[1],event.ydata))
        
        plt.draw()
        
    def up_mouse(self,event):
        """
        in charge of dealing with the mouse button up...this starts the brushing
        """
        if event.inaxes is None or self.mouse_is_down is False or (self.cur_mouse_axis !=  event.inaxes):
            self.mouse_is_down = False
            self.start_xy = (None,None)
            return
        
        # where are we?
        self.end_xy = (event.xdata,event.ydata)
        self.mouse_is_down = False
        
        if self.end_xy == self.start_xy:
            #kill the current rectangle since the user just clicked and released
            event.inaxes.patches.pop(-1)
            self.list_of_brushes.pop(-1)
            self.start_xy = (None,None)
            return
        else:
            self.start_xy = (None,None)
        
        ## now brush the data in the other axes
        self._brush(event,self.list_of_brushes[-1][1],inverse=False)
        
class DataReader:
    """
    class to read in a dataset, draw some scatter plots with labels and then call the Brusher
    """
    def __init__(self,fname=None,plots=[("x","y"),("z","y"),("z","x"),("y","x")],delimiter=",",opac=4.0):
        
        # read in the data - if we don't provide data, just use the test_data
        if fname is None:
            dat = self.test_data()
        else:
            dat = self.load_data(fname,delimiter=delimiter)
    
        ## now plot the data
        if dat is not None:
            self.plot_data(dat,plots)
    
            ## now turn on brushing
            self.b = Brusher(self.fig,self.axes,dat,opac=opac)
     
    def get_inds(self):
        return np.where(self.b.inds)

    def load_data(self,fname,delimiter=","):
        """expecting a data file like (to be read by csv2rec):
            # comments
            x  y  z  a
            1.0 2.0 3.0 4.0
            2.0 3.4  3.4 21.0
            ... 
            """
        if not os.path.exists(fname):
            print "Sorry. File %s could not be found."
            return None
            
        tmp= plt.mlab.csv2rec(fname,delimiter=delimiter)
        print tmp
        return tmp  
        
    def plot_data(self,data,plots):
        """
        plot the parameters requested. In principle, this list of tuple pairs 
        can be very large and you can have multiple subplot views of your data.
        """

        self.fig = plt.figure()
        self.fig.subplots_adjust(left=0.05,bottom=0.05,right=0.99,top=0.99
                            ,wspace=0.11,hspace=0.11)
        ## figure out the number plots to make and how to order them
        if plots is None:
            print len(data[0])
            parameters = ["p%i" % i for i in range(len(data[0]))]
            plots = []
            for i,p in enumerate(parameters[:-1]):
                for pp in parameters[i+1:]:
                    plots.append( (p,pp))
                    
        nrow  = int(np.floor(np.sqrt(len(plots))))
        ncols = int(np.ceil(len(plots)/nrow))
    
        ## loop over all the plots
        self.axes = []
        for i,p in enumerate(plots):
            
            # add the axes and also save what's being plotted here
            self.axes.append( (self.fig.add_subplot(nrow,ncols,i+1),p))

            # Set the color explicitly for each point
            cols = np.ones( (len(data),4))
            cols[:,0:2] = 0.4
            
            plt.scatter(data[p[0]],data[p[1]],c=cols,edgecolors='none')
            plt.xlabel( p[0] )
            plt.ylabel( p[1] )
            
        plt.draw()

            
    def test_data(self):
        """in case there is no datafile. Some silly data just to play with."""
        dat =  """x               y             z
           1.           0.49395274   0.98299284
           2.           0.72180746   0.59904682
           3.           0.25871506   0.45881593
           4.           0.34231552   0.80565325
           5.           0.72094288   0.70655061
           6.           0.6601817    0.08576389
           7.           0.05178359   0.7585382
           8.           0.82424464   0.90159083
           9.           0.926813     0.44040055
          10.           0.11904565   0.93605534
        """
        import StringIO
        tmp= plt.mlab.csv2rec(StringIO.StringIO(dat),delimiter=" ")
        return tmp
        
def test():
    """brush up some Spam-stock"""
    b = DataReader("hormel.csv",plots=[("date","open"),("open","close"),("date","volume"),("low","high")],opac=20)
    
def test1():
    """brush the play dataset"""
    
    d = DataReader()    

def test2():
    """look at object color and distance measurements from the Sloan Digital Sky Survey"""
    
    b = DataReader("sdss.csv",plots=[("ra","dec"),("gr","ug"),("ri","iz"),("dered_g","gr"),("gr","z"),("specclass","z")],opac=20)

if __name__ == "__main__":
    test()
