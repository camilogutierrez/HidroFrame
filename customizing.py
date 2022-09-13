import matplotlib.pyplot as plt
import matplotlib

print('customizing')
plt.rcdefaults()
def hola():
    print(var)
    
    #Graficacion #Esto debe de ir en un archivo .py
#-----------
# matplotlib.rcParams['font.family'] = 'Avenir LT Std'
# matplotlib.rmatplotlibs['font.size'] = 15
SMALL_SIZE = 10
MEDIUM_SIZE = 10
BIGGER_SIZE = 10
# SMALL_SIZE = 12
# MEDIUM_SIZE = 12
# BIGGER_SIZE = 14

# https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
####  Muy Importante   ######
plt.rcParams.keys() #Me dice los parametros como estan en MAtplotlib #Muy IMportante

# plt.rcParams["font.family"] = "Cambria"
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('figure', figsize=[5, 3])    # Cambiar el tamaño por defecto Widht,Height
#plt.rcParams["font.family"] = "Times New Roman"
#plt.rcParams["font.family"] = "sans-serif"
# plt.rc('font',**{'family':'serif','serif':['Times']}) #Latex
#plt.rc('text', usetex=True) #Latex
# matplotlib.rcParams['mathtext.fontset']='custom'
# matplotlib.rcParams['mathtext.default']='rm' # Lates sin cursiva
# matplotlib.rcParams['mathtext.rm']='cambria'

matplotlib.rcParams['axes.edgecolor'] = '0.6'
matplotlib.rcParams['axes.labelcolor'] = '0.1'
matplotlib.rcParams['axes.linewidth'] = '0.65'
matplotlib.rcParams['text.color'] = '.13'
matplotlib.rcParams['xtick.color'] = '.15'
matplotlib.rcParams['xtick.direction'] = 'out' # ticks in or out
matplotlib.rcParams['ytick.color'] = '.15'
matplotlib.rcParams['xtick.major.size'] = "2"     ## major tick size in points
matplotlib.rcParams['ytick.major.size'] = "2"     ## major tick size in points
matplotlib.rcParams['xtick.major.width'] = "0.8"     ## major tick width in points
matplotlib.rcParams['xtick.major.pad'] = "2.5"  ## distance to major tick label in points
matplotlib.rcParams['ytick.major.pad'] = "2.5"  ## distance to major tick label in points
#xtick.minor.size    : 2       ## minor tick size in points
#xtick.major.width   : 0.8     ## major tick width in points
#xtick.minor.width   : 0.6     ## minor tick width in points
#xtick.major.pad     : 3.5     ## distance to major tick label in points
#xtick.minor.pad     : 3.4     ## distance to the minor tick label in points
#matplotlib.rcParams['xtick.minor.visible'] = False   ## visibility of minor ticks on x-axis
matplotlib.rcParams['xtick.minor.size'] = 0  

matplotlib.rcParams['axes.grid'] = True   ## display grid or not
matplotlib.rcParams["grid.linestyle"] = '--'
matplotlib.rcParams["grid.linewidth"] = 0.4
matplotlib.rcParams["grid.alpha"] = 0.6

matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rcParams['axes.spines.top'] = True
matplotlib.rcParams['axes.spines.right'] = True
# register_matplotlib_converters()
matplotlib.rcParams['savefig.bbox']='tight'    ## {tight, standard}

