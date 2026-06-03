'''
Plotting something...

About fig and ax, you can use `fig = plt.figure(figsize=(7,5))` and `ax1 = fig.add_subplot(111)` as an example.

There are `LineChart` and `ImagePlot`, defined therein.
'''
# %% Total Importations
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Functions

def LineChart(xydata, fig=None, ax=None, *args):
    '''
    #### Plot a 2D data. The default `fig` and `ax` are `fig = plt.figure(figsize=(7,5))` and `ax1 = fig.add_subplot(111)`.
    
    - ### Data Style:
    
      `datadict["color"]`           = None (default), "firebrick", "darkorange", "gold", "forestgreen", "darkturquoise", "deepskyblue", "mediumblue", "blueviolet", "Magneta", "black", "white"
           
      `datadict["linewidth"]`       = 1 (default), or a float (you can also set it to be 0)
           
      `datadict["linestyle"]`       = "solid" (default), "dotted", "dashed", "dashdot"
           
      `datadict["marker"]`          = None (default), ".", "^", "+", "x"
           
      `datadict["markersize"]`      = 6 (default), or a float
       
      `datadict["markerfacecolor"]` = None (default), "firebrick", "darkorange", "gold", "forestgreen", "darkturquoise", "deepskyblue", "mediumblue", "blueviolet", "Magneta", "black", "white"
      
      `datadict["markeredgecolor"]` = None (default), "firebrick", "darkorange", "gold", "forestgreen", "darkturquoise", "deepskyblue", "mediumblue", "blueviolet", "Magneta", "black", "white"
      
      `datadict["markeredgewidth"]` = None (default), or a float (you can also set it to be 0)
      
      `datadict["label"]`           = None (default), or a string
           
      `datadict["alpha"]`           = 1 (default), or float in the range of [0,1]
      
    - ### Title:
    
      `titledict["label"]`               = None (default), or a string
       
      `titledict["fontdict"]["family"]`  = 'DejaVu Sans' (default)
       
      `titledict["fontdict"]["weight"]`  = 'normal' (default), "bold", "heavy", "light"
       
      `titledict["fontdict"]["size"]`    = 20 (default), or a float
      
    - ### x-axis:
    
      `otherdict["xlim"]`                = None (default), (-1, 1), or a tuple
                  
      `otherdict["xscale"]`              = "linear" (default), "log"
                
      `xlabeldict["xlabel"]`             = None (default), or a string
      
      `xlabeldict["fontdict"]["family"]` = 'DejaVu Sans' (default)
      
      `xlabeldict["fontdict"]["weight"]` = 'normal' (default), "bold", "heavy", "light"
      
      `xlabeldict["fontdict"]["size"]`   = 14 (default), or a float
      
    - ### y-axis:
    
      `otherdict["ylim"]`                = None (default), (-1, 1), or tuple
                  
      `otherdict["yscale"]`              = "linear" (default), "log"
              
      `ylabeldict["ylabel"]`             = None (default), or a string
      
      `ylabeldict["fontdict"]["family"]` = 'DejaVu Sans' (default)
      
      `ylabeldict["fontdict"]["weight"]` = 'normal' (default), "bold", "heavy", "light"
      
      `ylabeldict["fontdict"]["size"]`   = 14 (default), or a float
      
    - ### color-scatter plot:
    
      `scatterdict["c"]`        = None (default), or an array
          
      `scatterdict["cmap"]`     = None (default), or "Greys", "RdBu", "jet", "plasma", "hot"
        
      `caxdict["position"]`     = "right" (default), "top", "bottom", "left"
          
      `caxdict["size"]`         = "5%" (default), or a float
          
      `caxdict["pad"]`          = 0.05 (default), or "X%"
      
      `cbardict["orientation"]` = "vertical" (default), "horizontal"
      
      `otherdict["cscale"]`     = "linear" (default), "log"
      
      `otherdict["removeCbar"]` = False (default), True
      
      `clabeldict["ylabel"]`    = None (default), or a string
      
      `clabeldict["rotation"]`  = "vertical" (default), or string with the unit of deg
      
      `clabeldict["labelpad"]`  = 20 (default), or a float
      
      `clabeldict["fontsize"]`  = 14 (default), or a float

    - ### errorbar plot:

      `errorbardict["xerr"]`       = None (default), a float (all data has the same error), an (1, N) array (different error but the upper and lower error are the same), or a (2, N) array (different upper and lower error)
    
      `errorbardict["yerr"]`       = None (default), a float (all data has the same error), an (1, N) array (different error but the upper and lower error are the same), or a (2, N) array (different upper and lower error)
          
      `errorbardict["ecolor"]`     = None (default), "firebrick", "darkorange", "gold", "forestgreen", "darkturquoise", "deepskyblue", "mediumblue", "blueviolet", "Magneta", "black", "white"
      
      `errorbardict["elinewidth"]` = 1 (default), or a float
      
      `errorbardict["capsize"]`    = 6 (default), or a float
      
      `errorbardict["capthick"]`   = 1 (default), or a float
      
      `errorbardict["errorevery"]` = True (default), an int (error bar for data[::N]), or a (int, int) (error bar for data[start::N])

    - ### Ticks:
    
      `ticksdict["xticks"]`      = None (default), [0, 1, 2, 3, 5, 6], np.linspace(-1, 1, 5), or 1D list/array
         
      `ticksdict["yticks"]`      = None (default), [0, 1, 2, 3, 5, 6], np.linspace(-1, 1, 5), or 1D list/array
           
      `tickpmsdict["axis"]`      = "both" (default), "x", "y"
         
      `tickpmsdict["length"]`    = 3.5 (default), or a float. To hide the ticks, let it be 0.
    
      `tickpmsdict["direction"]` = "in" (default), "out"
      
      `tickpmsdict["labelsize"]` = 12 (default), or a float
      
      `ticklabform["style"]`     = "sci" (default), or "plain"
      
      `ticklabform["scilimits"]` = (-1, 2) (default), or a tuple. Outside this range will use sci.
    
      `ticklabform["axis"]`      = "both" (default), or "x", "y"
      
    - ### Grid:
    
      `griddict["visible"]`      = False (default), or True
      
      `griddict["which"]`        = "major" (default), "minor", "both"
             
      `griddict["axis"]`         = "both" (default), "x", "y"
      
    - ### Legend:
    
      `legenddict["loc"]`        = "best" (default), "upper left", "center", "lower right"
      
    '''
    
    datadict={
        "color"           : None, 
        "linewidth"       : 1, 
        "markeredgewidth" : None,
        "linestyle"       : "-", 
        "marker"          : None,
        "markersize"      : 6,
        "markerfacecolor" : None,
        "markeredgecolor" : None,
        "label"           : None,
        "alpha"           : 1
    }  
      
    scatterdict={  
    "marker"          : None,
    "s"               : None,
    "label"           : None,
    "c"               : None,
    "cmap"            : None
    }  

    errorbardict={
      "xerr"       : None,
      "yerr"       : None,
      "ecolor"     : None,
      "elinewidth" : 1,
      "capsize"    : 6,
      "capthick"   : 1,
      "errorevery" : True
    }
  
    caxdict={  
        "position"    : "right",
        "size"        : '5%',
        "pad"         : 0.05
    }

    cbardict={
        "orientation" : 'vertical'
    }

    titledict={
        "label"       : None,
        "fontdict"    : {'family': 'DejaVu Sans', 
                         'weight': 'normal', 
                         'size': 20
                        }
    }

    xlabeldict={
        "xlabel"      : None,
        "fontdict"    : {'family': 'DejaVu Sans', 
                         'weight': 'normal', 
                         'size': 14
                        }
    }

    ylabeldict={
        "ylabel"      : None,
        "fontdict"    : {'family': 'DejaVu Sans', 
                         'weight': 'normal', 
                         'size': 14
                        }
    }
    
    clabeldict={
        "ylabel"      : None,
        "rotation"    : "vertical",
        "labelpad"    : 20,
        "fontsize"    : 14
    }
    
    griddict={
        "visible"     : False, 
        "which"       : "major", 
        "axis"        : "both",  
    }
    
    legenddict={
        "loc"         : "best"
    }
    
    otherdict={
      "xlim"          : None,
      "ylim"          : None,
      "xscale"        : "linear",
      "yscale"        : "linear",
      "cscale"        : "linear",
      "removeCbar"    : False
    }
    
    ticksdict={
      "xticks"        : None, 
      "yticks"        : None
    }
    
    tickpmsdict={
      "axis"          : 'both',
      "length"        : 3.5,
      "direction"     : "in", 
      "labelsize"     : 12
    }
    
    ticklabform={
      "style"         : 'sci',
      "scilimits"     : (-1, 2),
      "axis"          : 'both'
    }

    if fig == None:
        fig = plt.figure(figsize=(7, 5))
        
    if ax == None:
        ax = fig.add_subplot(111)

    for command in args:
        exec(command)
        
    x, y = xydata
    if scatterdict["c"] != None:
        scatterdict["marker"] = datadict["marker"]
        scatterdict["s"]      = datadict["markersize"]
        scatterdict["label"]  = datadict["label"]

        im = ax.scatter(x, y, **scatterdict)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(**caxdict)
        cbar = fig.colorbar(im, cax=cax, **cbardict)
        cbar.ax.set_ylabel(**clabeldict)
        cbar.ax.set_yscale(otherdict["yscale"])
        cbar.ax.tick_params(**tickpmsdict)
        
    elif errorbardict["xerr"] != None or errorbardict["yerr"] != None:
        ax.errorbar(x, y, **datadict, **errorbardict)

    else:
        ax.plot(x, y, **datadict)
    
    ax.set_title(**titledict)
    ax.set_xlabel(**xlabeldict)
    ax.set_ylabel(**ylabeldict)
    
    ax.set_xscale(otherdict["xscale"])
    ax.set_yscale(otherdict["yscale"])
    ax.set_xlim(otherdict["xlim"])
    ax.set_ylim(otherdict["ylim"])
    ax.grid(**griddict)
    ax.tick_params(**tickpmsdict)
    ax.ticklabel_format(**ticklabform)
    
    if otherdict["xlim"] == None:
        ax.autoscale(axis='x')
    if otherdict["ylim"] == None:
        ax.autoscale(axis='y')
        
    if ticksdict["xticks"] != None:
        ax.xaxis.set_ticks(ticksdict["xticks"])  
    else:
        pass

    if ticksdict["yticks"] != None:
        ax.yaxis.set_ticks(ticksdict["yticks"])  
    else:
        pass
      
    if datadict["label"] != None:
        ax.legend(**legenddict)
    else:
        pass
    
def ImagePlot(cdata, xydata=None, fig=None, ax=None, *args):
    '''
    #### Plot a 3D data. The default `fig` and `ax` are `fig = plt.figure(figsize=(7,5))` and `ax1 = fig.add_subplot(111)`, and the default `xydata` is the index of cdata.
    
    - ### Image Style:
    
      `imdict["vmin"]` = None (default), or a float
      
      `imdict["vmax"]` = None (default), or a float
      
      `imdict["cmap"]` = "Greys" (default), "RdBu", "jet", "plasma", "hot", "coolwarm"
      
      `imdict["aspect"]` = "auto" (default), "equal", or a float
      
      `imdict["alpha"]` = 1 (default), or a float in the range of [0,1]
    
    - ### Colorbar:
    
      `caxdict["position"]` = "right" (default), "top", "bottom", "left"
      
      `caxdict["size"]` = "5%" (default), or a float
      
      `caxdict["pad"]` = 0.05 (default), or "X%"
      
      `cbardict["orientation"]` = "vertical" (default), "horizontal"
      
      `otherdict["cscale"]` = "linear" (default), "log"
      
      `otherdict["removeCbar"]` = False (default), True
      
      `clabeldict["ylabel"]` = None (default), or a string
      
      `clabeldict["rotation"]` = "vertical" (default), or a string with the unit of deg
      
      `clabeldict["labelpad"]` = 20 (default), or a float 
      
      `clabeldict["fontsize"]` = 14 (default), or a float 
    
    - ### Title:
    
      `titledict["label"]` = None (default), or a string
      
      `titledict["fontdict"]["family"]` = 'DejaVu Sans' (default)
      
      `titledict["fontdict"]["weight"]` = 'normal' (default), "bold", "heavy", "light"
      
      `titledict["fontdict"]["size"]` = 20 (default), or a float
      
    - ### x-axis:
    
      `otherdict["xlim"]` = None (default), (-1, 1), or a tuple
      
      `otherdict["xscale"]` = "linear" (default), "log"
    
      `xlabeldict["xlabel"]` = None (default), or a string
      
      `xlabeldict["fontdict"]["family"]` = 'DejaVu Sans' (default)
      
      `xlabeldict["fontdict"]["weight"]` = 'normal' (default), "bold", "heavy", "light"
      
      `xlabeldict["fontdict"]["size"]` = 14 (default), or a float
      
    - ### y-axis:
    
      `otherdict["ylim"]` = None (default), (-1, 1), or a tuple
      
      `otherdict["yscale"]` = "linear" (default), "log"
    
      `ylabeldict["ylabel"]` = None (default), or a string
       
      `ylabeldict["fontdict"]["family"]` = 'DejaVu Sans' (default)
      
      `ylabeldict["fontdict"]["weight"]` = 'normal' (default), "bold", "heavy", "light"
      
      `ylabeldict["fontdict"]["size"]` = 14 (default), or a float
      
    - ### Ticks:
    
      `ticksdict["xticks"]` = None (default), [0, 1, 2, 3, 5, 6], np.linspace(-1, 1, 5), or 1D list/array
    
      `ticksdict["yticks"]` = None (default), [0, 1, 2, 3, 5, 6], np.linspace(-1, 1, 5), or 1D list/array
      
      `ticksdict["cticks"]` = None (default), [0, 1, 2, 3, 5, 6], np.linspace(-1, 1, 5), or 1D list/array
      
      `tickpmsdict["axis"]` = "both" (default), "x", "y"
      
      `tickpmsdict["length"]` = 3.5 (default), or a float. To hide the ticks, let it be 0.
    
      `tickpmsdict["direction"]` = "in" (default), "out"
      
      `tickpmsdict["labelsize"]` = 12 (default), or a float
      
      `ticklabform["style"]`     = "sci" (default), or "plain"
      
      `ticklabform["scilimits"]` = (-1, 2) (default), or a tuple. Outside this range will use sci.
    
      `ticklabform["axis"]`      = "both" (default), or "x", "y"
      
    '''
    
    imdict={
        "vmin"        : None,
        "vmax"        : None,
        "cmap"        : "Greys",
        "aspect"      : "auto",
        "alpha"       : 1
    }

    titledict={
        "label"       : None,
        "fontdict"    : {'family': 'DejaVu Sans', 
                         'weight': 'normal', 
                         'size': 20
                        }
    }

    xlabeldict={
        "xlabel"      : None,
        "fontdict"    : {'family': 'DejaVu Sans', 
                         'weight': 'normal', 
                         'size': 14
                        }
    }

    ylabeldict={
        "ylabel"      : None,
        "fontdict"    : {'family': 'DejaVu Sans', 
                         'weight': 'normal', 
                         'size': 14
                        }
    }

    caxdict={
        "position"    : "right",
        "size"        : '5%',
        "pad"         : 0.05
    }

    cbardict={
        "orientation" : 'vertical'
    }

    clabeldict={
        "ylabel"      : None,
        "rotation"    : "vertical",
        "labelpad"    : 20,
        "fontsize"    : 14
    }
    
    otherdict={
      "xlim"          : None,
      "ylim"          : None,
      "xscale"        : "linear",
      "yscale"        : "linear",
      "cscale"        : "linear",
      "removeCbar"    : False
    }
    
    ticksdict={
      "xticks"        : None, 
      "yticks"        : None,
      "cticks"        : None
    }
    
    tickpmsdict={
      "axis"          : 'both',
      "length"        : 3.5,
      "direction"     : "in", 
      "labelsize"     : 12
    }
    
    ticklabform={
      "style"         : 'sci',
      "scilimits"     : (-1, 2),
      "axis"          : 'both'
    }
    
    if xydata == None:
        xydata = ([i for i in range(len(cdata[0]))], [i for i in range(len(cdata))])
    
    if fig == None:
        fig = plt.figure(figsize=(7,5))
        
    if ax == None:
        ax = fig.add_subplot(111)
    
    for command in args:
        exec(command)

    x, y = xydata
    im = ax.imshow(cdata, extent=[x[0], x[-1], y[0], y[-1]], **imdict)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(**caxdict)
    cbar = fig.colorbar(im, cax=cax, **cbardict)
    cbar.ax.set_ylabel(**clabeldict)
    cbar.ax.set_yscale(otherdict["yscale"])
    cbar.ax.tick_params(**tickpmsdict)
    cbar.ax.ticklabel_format(**ticklabform)
    
    if ticksdict["cticks"] != None:
        cbar.ax.yaxis.set_ticks(ticksdict["cticks"])  
    else:
        pass
      
    if otherdict["removeCbar"] == True:
        cbar.remove()
    
    ax.set_title(**titledict)
    ax.set_xlabel(**xlabeldict)
    ax.set_ylabel(**ylabeldict)
      
    ax.set_xscale(otherdict["xscale"])
    ax.set_yscale(otherdict["yscale"])
    ax.set_xlim(otherdict["xlim"])
    ax.set_ylim(otherdict["ylim"])
    ax.tick_params(**tickpmsdict)
    ax.ticklabel_format(**ticklabform)
    
    if otherdict["xlim"] == None:
        ax.autoscale(axis='x')
    if otherdict["ylim"] == None:
        ax.autoscale(axis='y')
        
    if ticksdict["xticks"] != None:
        ax.xaxis.set_ticks(ticksdict["xticks"])  
    else:
        pass

    if ticksdict["yticks"] != None:
        ax.yaxis.set_ticks(ticksdict["yticks"])  
    else:
        pass

def moving_window(window_size, time, time_delay, min_grid, max_grid, moving_factor):
    moving_center = sc.c*(time - time_delay)*moving_factor

    if moving_center <= min_grid+window_size/2:
        return (min_grid, min_grid+window_size)
        
    elif moving_center > min_grid+window_size/2 and moving_center < max_grid-window_size/2:
        return (moving_center-window_size/2, moving_center+window_size/2)
        
    elif moving_center >= max_grid-window_size/2:
        return (max_grid-window_size, max_grid)
      
if __name__ == "__main__":
  
    # 2d plot 

    x = [10000,20000,30000]; y = [4,5,6]; z = [-4,-5,-6]

    fig = plt.figure(figsize=(7,5))
    ax1 = fig.add_subplot(111)
    LineChart((x,y), fig, ax1,
            'datadict["color"]           = "firebrick"',
            'datadict["marker"]          = "o"', 
            'datadict["markersize"]      = 10',
            'datadict["markerfacecolor"] = "none"',
            'datadict["markeredgecolor"] = "black"',
            'datadict["markeredgewidth"] = 2',
            'datadict["alpha"]           = 0.7',
            'titledict["label"]          = "Title"',
            # 'otherdict["xlim"]           = (-1, 2)'
            'errorbardict["xerr"]     = 3000',
            'errorbardict["yerr"]     = 1'
            )
    ax1.ticklabel_format(style='scientific')
    
    # 2d color plot
    
    fig = plt.figure(figsize=(7,5))
    ax1 = fig.add_subplot(111)
    LineChart((x,y), fig, ax1,
            'datadict["color"]      = "firebrick"',
            'datadict["marker"]     = "o"', 
            'datadict["markersize"] = 50',
            'titledict["label"]     = "Title"',
            # 'otherdict["xlim"]      = (-1, 2)',
           f'scatterdict["c"]       = {z}',
            'scatterdict["cmap"]    = "Reds"',
            'xlabeldict["xlabel"]   = "z (m)"',
            )
    
    # 3d plot

    xx, yy = np.meshgrid(x, y); zz = xx^2+yy^2

    fig = plt.figure(figsize=(7,5))
    ax1 = fig.add_subplot(111)
    ImagePlot(zz, (x, y), fig, ax1,
                  'imdict["cmap"] = "plasma"',
                  'ticksdict["xticks"] = [1.5, 1.875]',
                  'caxdict["pad"] = 0.05',
                  # 'otherdict["removeCbar"] = True',
                  # 'otherdict["xscale"] = "log"',
                  'ticksdict["xticks"] = [0, 1, 2, 3, 5, 6]'
                  )


# %%
