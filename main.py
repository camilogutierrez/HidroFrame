'''
import sys
sy.path.append('C:\\Users\\Camilo\\Proyectos\\Hidrologia\\')
from HidroFrame import *
'''
import operator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scs
import matplotlib
from scipy import stats
from pathlib import Path
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except:
    pass
try:
    import geopandas as gpd
    import osr
    import gdal
except:
    pass
import seaborn as sns
sns.set(style="whitegrid")
sns.reset_orig()
path = Path(__file__).parent
df_estaciones = pd.read_excel(path/'CNE_IDEAM.xls')
def cicloanual_diario(df, agg='mean'):

    caudal_groupby = df.groupby(by=[df.index.month, df.index.day]).agg(agg)
    caudal_groupby.index.names = ['Mes', 'Dia']
#     caudal_groupby=caudal_groupby.rename(index={0:'Date'})

    return caudal_groupby


def cicloanual_mensual(df, agg='mean'):

    caudal_groupby = df.groupby(by=[df.index.month]).agg(agg)
    caudal_groupby.index.names = ['Mes']
    assert len(caudal_groupby) == 12, "No se tiene información de los 12 meses"
#     caudal_groupby=caudal_groupby.rename(index={0:'Date'})

    return caudal_groupby


def cicloanual_mensual_plot(df, ylabel='Caudal ($m^3/s$)', ax=None, label=None, outlayer_sym='', box_plot=False, bands=False, **kwargs):
    '''
    outlayer_sym: string i.e '.','x','o', Default is '' that means no outlayer is displayed.
    '''
    medianprops = dict(linestyle='-', linewidth=0.5, color='r')
    boxprops = dict(linestyle='-', linewidth=0.5, color='r')
    whiskerprops = dict(linestyle='-', linewidth=0.5, color='r')
    # if kwargs:
    # medianprops = dict(linestyle='-', linewidth=0.5, color='r')
    # boxprops = kwargs.pop('boxprops')
    # whiskerprops = kwargs.pop('whiskerprops')
    if not ax:
        fig, ax = plt.subplots()
        subplots = False

    if box_plot:
        mes_col(df).plot(ax=ax, kind='box', widths=0.5, patch_artist=False, notch=False, medianprops=medianprops,
                         color='black', sym=outlayer_sym, boxprops=boxprops)
        ax.plot(cicloanual_mensual(df), ls='--', lw=1, marker='.',
                label='Media Mensual', color=kwargs.get('color', 'goldenrod'))

    elif box_plot:
        subplots = False
#         fig,ax = plt.subplots()
        ax = plt.gca()
        mes_col(df).plot(ax=ax, kind='box', widths=0.5, patch_artist=False, notch=False,
                         medianprops=medianprops, color='black', sym=outlayer_sym, boxprops=boxprops)
        ax.plot(cicloanual_mensual(df), label='Media mensual', **kwargs)
    elif ax:
        ax.plot(cicloanual_mensual(df), label=label, **kwargs)
    else:
        subplots = False
        ax = plt.gca()
    ax.set_xticks(np.arange(1, 13))
    calendario = ['Ene', 'Feb', 'Mar', 'Abr', 'May',
                  'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    ax.set_xticklabels(calendario)
    if bands:
        sup = cicloanual_mensual(df, agg='mean') + \
            cicloanual_mensual(df, agg='std')
        inf = cicloanual_mensual(df, agg='mean') - \
            cicloanual_mensual(df, agg='std')
        inf1 = list(inf.index)
        # inf1[0] = 0.75
        # inf1[-1] = 12.25
        ax.fill_between(inf1, inf.values.ravel(), sup.values.ravel(),
                        alpha=0.1, label='Desviación estandar')
    try:
        if subplots == False:
            ax.legend(ncol=2, loc='upper center', bbox_to_anchor=(
                0.5, 1), bbox_transform=fig.transFigure)
        else:
            ax.legend(ncol=2, loc='upper center', bbox_to_anchor=(
                0.5, 1), bbox_transform=fig.transFigure)
    except:
        pass
#     ax.set_title(df.columns[0],fontsize= MEDIUM_SIZE)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Mes')
    ax.grid(axis='y', ls='--', alpha=0.8)
    return ax
#     plt.title(df.columns[0],fontsize=BIGGER_SIZE)


def mes_col(dfserie):
    '''vuelve un dataframe que tenga datos con indices en DateTime en columnas meses y filas años'''
    return pd.pivot_table(dfserie, index=dfserie.index.year, columns=dfserie.index.month)


def curva_duracion(df_group):
    '''Solo para dataframes de una columna

    cicloanual_mensual(df1, agg=[percentile(75),'mean','std'])
    '''

    array = df_group.values.ravel()
    array = np.sort(array)[::-1]
    arraytemp = array[np.isfinite(array)]
    exceedence = np.arange(1., len(arraytemp)+1) / (len(arraytemp) + 1)

    return (exceedence*100, arraytemp)


def curva_duracion_plot(df, ax=None, label='', **kwargs):
    """
    Graficar curva de duracion de caudales
    Parametros
    -------------------
    df: dataframe de 

    --------------------
    Ejemplo: 

    Con ax activado:
        fig, ax = plt.subplots(1,2,sharex=True, sharey=True)
        curva_duracion_plot(estacion1,ax=ax[0])
        curva_duracion_plot(estacion2, ax=ax[1])
        ax[1].set_ylabel('')

    sin ax activado:
        curva_duracion_plot(estacion1) 
    """
    if not ax:
        fig, ax = plt.subplots(1, 1)
    ax.plot(curva_duracion(df)[0], curva_duracion(
        df)[1], label=label, **kwargs)
    y = curva_duracion(df)[1]
#     ax.set_xlim(0,100)
#     ax.set_ylim(y.min(),y.max())
#     ax.set_title(label)
    ax.set_aspect(1/ax.get_data_ratio(), adjustable='box')
    ax.set_xlabel("Porcentaje de excedencia [%]")
    ax.set_ylabel("Caudal  $m^3/s$")
    # plt.tight_layout()


def curva_duracion_mensual_plot(df):
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(
        12, 15), sharex=False, sharey=False)

    meses = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio',
             'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']

    for i, ax in enumerate(axes.flat, start=1):
        df_temp = df.groupby(df.index.month).apply(curva_duracion)[i]
        ax.plot(df_temp[0], df_temp[1], label='label')
        ax.set_title('{}'.capitalize().format(meses[i-1].capitalize()))
        if i > 9:
            ax.set_xlabel('Porcentaje de excedencia [%]')
        if i in [1, 4, 7, 10]:
            #         ax.set_xlabel('X axis')
            ax.set_ylabel('Caudal  $m^3/s$')
    fig.tight_layout()


def get_station(dictionary, nombre):
    '''Concatena un diccionario donde cada key es una estacion
    -----Parametros----
    nombre:  string
    dictionary: dictionario de estaciones con dictionary.keys() de la forma "nombre-estacion [11111111]"
    -----Retorna----
    Dataframe 
    ------------------
    '''
    catalogo = pd.Series(list(dictionary.keys()))
    estacion = catalogo[catalogo.str.contains(nombre, case=False)].values[0]
    return dictionary[estacion].resample('M').mean()


def dict_dhime(ruta, etiqueta):
    '''Concatena un diccionario donde cada key es una estacion


    Return
    ---------
    Dictionary
    '''
    series = pd.read_csv(ruta, index_col='Fecha', parse_dates=True)
    dict_dhime = {}
    # by=lista, donde lista son los lab
    for lab, group in series.groupby(by=['Etiqueta', 'NombreEstacion']):
        if lab[0] == etiqueta:
            # lab 1 corresponde al nombre de la estacion
            dict_dhime[lab[1]] = group[['Valor']]
            dict_dhime[lab[1]].columns = [lab[1]]
    return dict_dhime


def df_concatenate(dictionary, plot=False, freq='d', column_names=True):
    '''Concatena un diccionario donde cada key es una estacion
    -----Parametros----
    dictionary.keys() es de la forma "nombre-estacion [11111111]"
    -----Retorna----
    Grafica con subplots=True
    Dataframe 
    ------------------
    '''
    if column_names:
        df_concatenado = pd.concat([dictionary[key] for key in dictionary.keys(
        )], axis=1, join='outer')
        df_concatenado.columns = dictionary.keys()
    else:
        df_concatenado = pd.concat(
            dictionary, axis=1, join='outer').droplevel(level=1, axis=1)

    cols = df_concatenado.columns
    df_concatenado[cols] = df_concatenado[cols].apply(
        pd.to_numeric, errors='coerce', axis=1)
    # El resample solo fucniona si los valores son numericos
    df_concatenado = df_concatenado.resample(freq).mean()

    # Outer es UNION, axis 1 por columnas
    if plot == True:
        df_concatenado.plot(figsize=(15, 10), subplots=True, sharey=True)
        print('\n')
        df_concatenado.plot(figsize=(15, 5), subplots=False)
        plt.legend(bbox_to_anchor=(1, 1))
    return df_concatenado


def valid_obs_year(x, porcentaje_validos=1, function='mean'):
    '''
    Resamplear con observaciones validas desde meses

    df.resample('m').apply(valid_obs_month,**{'function':'min'} o function='min')
    function puede ser min,max,std,median...
    '''
    min_obs = porcentaje_validos * 12
    valid_obs = x.notnull().sum()
    if valid_obs >= min_obs:
        return x.apply(function)


def valid_obs_month(x, porcentaje_validos=0.8, function='mean'):
    '''
    Resamplear con observaciones validas desde dias si no da hay q resamplear 

    df.resample('m').apply(valid_obs_month,**{'function':'min'} o function='min')
    lluvia alternativa df.resample('m').sum(min_count=10)
    function puede ser min,max,std,median...
    '''
    min_obs = porcentaje_validos * x.index.days_in_month[-1]

    valid_obs = x.notnull().sum()
    if valid_obs >= min_obs:
        return x.apply(function)


def estandarizar_df(df):
    '''
    Estandariza con la media y desviacion estandar mensual
    '''
    return df.groupby(lambda x: x.month).transform(lambda x: (x - x.mean()) / x.std())


def estandarizar_df_diario(df_diario):
    '''
    Estandariza con la media y desviacion estandar diaria
    '''
    return df_diario.groupby(by=[df_diario.index.month, df_diario.index.day]).transform(lambda x: (x - x.mean()) / x.std())


def percentile(n):
    '''
    df1.groupby(df1.index.year).agg([percentile(50),'min','count','mean']) year o month...
    '''
    def percentile_(x):
        # el axis cambia cuando no funciona
        return np.nanpercentile(x, n, axis=0)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


def fecha_inicial(df):
    '''
    df1.groupby(df1.index.year).agg([percentile(50),'min','count','mean']) year o month...
    '''
#     fecha_inicial.__name__ = 'inicio'
    return df.index[0]


def fecha_final(df):
    #     fecha_inicial.__name__ = 'inicio'
    return df.index[-1]


# def fecha_inicial(n):
#     '''
#     df1.groupby(df1.index.year).agg([percentile(50),'min','count','mean']) year o month...
#     '''
#     def percentile_(x):
#         return np.nanpercentile(x, n)
#     percentile_.__name__ = 'percentile_%s' % n
#     return percentile_

def plot_info_anual(df):
    '''
    Informacion disponible por año de una serie diaria
    '''
    DF_dibujar = df.groupby(by=[df.index.year]).count()
    days = pd.to_datetime(DF_dibujar.index, format='%Y')
    DF_dibujar['dias'] = DF_dibujar.set_index(
        days).resample('Y').mean().index.dayofyear
    DF_dibujar = DF_dibujar.eval('porcentaje = vel/dias*100')
    DF_dibujar.porcentaje = DF_dibujar.porcentaje.apply(
        lambda x: np.round(x, 1))
    fig, ax1 = plt.subplots(1, 1)
    ax1.bar(DF_dibujar.index, DF_dibujar.porcentaje.values,
            zorder=3, edgecolor='k')
    ax1.grid(axis='y', alpha=0.8, zorder=0)
    ax1.set_ylabel('Info. disponible (%) ')
    # ax1.set_ylim([0,100])
    ax1.set_xlabel('Año')
    return ax1


class SerieHidrologica:
    def __init__(self, data):
        self.data = data
        self.cicloanual = self.cicloanual_mensual()

    def cicloanual_mensual(self):
        caudal_groupby = self.data.groupby(
            by=[self.data.index.month]).mean()
        caudal_groupby.index.names = ['Mes']
        return caudal_groupby

    def plot_ciclo(self, ax):
        #         fig,ax=plt.subplots()
        return cicloanual_mensual(self.data).plot(ax=ax, legend=False)
#         return ax


def create_3axes(figsize=(10, 6)):
    '''Create figure and 3 axis in 2 rows

    '''
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = fig.add_gridspec(4, 4)
    ax1 = fig.add_subplot(gs[:2, :2])
    ax2 = fig.add_subplot(gs[:2, 2:])
    ax3 = fig.add_subplot(gs[2:4, 1:3])
    return fig, ax1, ax2, ax3


def scatter_df(x, y=None, xlabel=None, ylabel=None, ax=None, share_x_lim=True, s=0, loc='lower', color=None,return_tuple = False, draw_line = True,**kwgs):
    '''Grafica de dispersion de dos dataframes que tengan el mismo indice i.e. fecha
    color = 'index.year'

    s: cambiar los set lim y  las bolitas se ven mejor
    '''
    if isinstance(y, pd.Series):
        y = y.to_frame()
    if isinstance(x, pd.Series):
        x = x.to_frame()
    if y is not None:
        joinxy = x.join(y).dropna()
        if joinxy.empty:
            raise ValueError('No hay solape en los datos')
    else:
        #    joinxy=pd.concat([x,y],axis=1,join='outer').dropna()
        joinxy = x.copy().dropna()
    x = joinxy.iloc[:, 0]
    y = joinxy.iloc[:, 1]
    maxi = np.max(np.r_[np.nanmax(x), np.nanmax(y)])
    mini = np.min(np.r_[np.nanmin(x), np.nanmin(y)])
    scs.pearsonr(x, y)

#     plt.figure(figsize=(4,4))
    if not ax:
        fig, ax = plt.subplots()
    if color:
        # its equivalent to getattr(df_reciente, "index")
        f_color = operator.attrgetter(color)
    kwgs.setdefault('edgecolor', 'black')
    kwgs.setdefault('lw', 0.5)
    scatter = ax.scatter(x, y, s=50,
                         c=f_color(joinxy) if color else None, **kwgs)
    if color:
        legend1 = ax.legend(*scatter.legend_elements(num=ticker.MaxNLocator(5, integer=True)),
                            loc="upper left")
        ax.add_artist(legend1)

    if not (xlabel or ylabel):
        xlabel = joinxy.iloc[:, 0].name
        ylabel = joinxy.iloc[:, 1].name
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if share_x_lim:
        maxi += s  # Me cambia para q se vean todas la bolitas
        mini -= s
        ax.plot([mini, maxi], [mini, maxi], ls='--', color='k', lw=0.7)
        ax.set_ylim([mini, maxi])
        ax.set_xlim([mini, maxi])
        locator = plt.MaxNLocator(5)
        ax.xaxis.set_major_locator(locator)
        ax.yaxis.set_major_locator(locator)
        ax.set_aspect(1./ax.get_data_ratio())
        # ax.set_aspect('equal')
    correlation_coef_r = scs.pearsonr(x, y)
    ax.grid(alpha=0.5)
    lg_fontsize = 12
    if loc == 'upper':
        ax.annotate(r'$r={:.2f}$'.format(correlation_coef_r[0]), fontsize=lg_fontsize, xy=(
            0.05, 0.85), xycoords='axes fraction')
    else:
        ax.annotate(r'$r={:.2f}$'.format(correlation_coef_r[0]), fontsize=lg_fontsize, xy=(
            0.8, 0.05), xycoords='axes fraction', ha='center')
    if draw_line:
        x_model = sm.add_constant(x) 
        model = sm.OLS(y, x_model).fit()
        p = model.params
        x_rango = np.arange(x.min(), x.max())
        ax.plot(x_rango, p.const + p.iloc[-1]* x_rango, color='red', lw=2.5)
    
    if return_tuple:
        return joinxy, ax


def correlograma(serie1, serie2=None, ax=None, rezagos=6, freq='m'):
    '''Los rezagos positivos (la derecha de la grafica) deben estas asociados a los indices que PRECEDEN el caudal.'''
#     serie1.index = serie1.index.to_period()
    correlaciones = {}
    correlaciones_scipy = {}

    if isinstance(serie2, pd.DataFrame):
        print('Ejecutandose autocorrolograma')
        serie2 = serie1.copy()

    # if not serie2:
    #    serie2 = serie1.copy()

    for rezago in range(-rezagos, rezagos+1):
        names = ['Caudal', 'Indice']
        series = pd.concat([serie1, serie2], axis=1, sort=True)
        series.columns = names
        series = series.resample(freq).mean()
        # Febrero de caudal con enero del indice si el rezago es positivo.
        series.Indice = series.Indice.shift(rezago, freq=freq)
        # SHift positivo es crear nan arriba y bajarlo
        series.dropna(inplace=True)
        # print(rezago)
        correlaciones_scipy[rezago] = stats.pearsonr(
            series.iloc[:, 0], series.iloc[:, 1])
        correlaciones[rezago] = series.corr().values[0, 1]

    if not ax:
        ax = plt.gca()

    corr_df = pd.DataFrame(data=correlaciones.values(),
                           index=correlaciones.keys())
    corr_df.plot(kind='bar', legend=None, ax=ax, edgecolor='k', zorder=10)
    ax.set_ylabel('Correlación de Pearson')
    ax.set_xlabel('Rezagos (meses)')
    ax.tick_params(labelrotation=0)
    ax.axhline(c='k')
    ax.grid(axis='both', ls='--', zorder=-50, lw=0.4, alpha=0.65)
    # ax.set_aspect(1./ax.get_data_ratio()) # Square

    return ax, corr_df


def correlacion(df):
    matriz_cor = mes_col(df).corr()
    correl = {}
    df = mes_col(df).copy()
    # enero = df.iloc[1:,0]
    enero = df.iloc[:, 0]
    diciembre = df.iloc[:, 11].shift()
    df = pd.concat([enero, diciembre], axis=1, join='outer').dropna()
    correl[1] = scs.pearsonr(df.iloc[:, 0].values.ravel(),
                             df.iloc[:, 1].values.ravel())[0]
    for i in np.arange(2, 13):
        correl[i] = matriz_cor.iloc[i-2, i-1]
    return correl, matriz_cor


def AR_1(df_analizar, primer_valor=None, fecha_primer_valor=None, mode_sintetica=False, log=True):

    df = df_analizar.copy()

    if log:
        df.iloc[:, 0] = np.log(df.iloc[:, 0])

    # Estadisticos basicos
    corr_mensuales, matriz_cor = correlacion(df)
    estadisticos = cicloanual_mensual(df, agg=['mean', 'std', 'count'])
    estadisticos.columns = estadisticos.columns.droplevel()
    medias = estadisticos.loc[:, 'mean']
    medias = medias.to_dict()
    desv_menusales = estadisticos.loc[:, 'std']
    desv_menusales = desv_menusales.to_dict()

    if mode_sintetica:
        if log:
            primer_valor = np.log(primer_valor)
        series_sinteticas = pd.DataFrame(index=pd.date_range(
            start=fecha_primer_valor, periods=12 * 50, freq='m'))
        series_sinteticas['valor'] = np.nan
        series_sinteticas.index.name = 'fecha'
        series_sinteticas.iloc[0, 0] = primer_valor
        series_sinteticas.index = series_sinteticas.index.to_period(freq='m')

    else:
        series_sinteticas = df.copy()
        series_sinteticas.columns = ['valor']
        if not isinstance(df.index, pd.PeriodIndex):
            series_sinteticas.index = series_sinteticas.index.to_period(
                freq='m')
    series_sinteticas['mes'] = series_sinteticas.index.month
    # np.random.seed(1)
    series_sinteticas['random'] = np.random.randn(len(series_sinteticas))

    for index, row in series_sinteticas.iterrows():
        if np.isnan(row['valor']):
            #         print(index.month)
            valor = 0
            mes_anterior = index.to_timestamp() - pd.offsets.DateOffset(months=1)
            q_anterior = series_sinteticas.loc[mes_anterior, 'valor']
            media_anterior = medias[mes_anterior.month]
            desv_anterior = desv_menusales[mes_anterior.month]
            media = medias[index.month]
            desv = desv_menusales[index.month]
            coef_correl = corr_mensuales[index.month]
            z = series_sinteticas.loc[index, 'random']
            valor = media + (coef_correl * desv / desv_anterior) * (q_anterior -
                                                                    media_anterior) + z * desv * (1 - (coef_correl)**2) ** 0.5
            series_sinteticas.loc[index, 'valor'] = valor
    if log:
        series_sinteticas.loc[:, 'valor'] = np.exp(
            series_sinteticas.iloc[:, 0])

    return series_sinteticas


def export_raster(Array, lat, lon, filename='output', EPSG=4326, Format='GTiff'):
    '''
    Array -shape = LAT LON
    Algunas veces toca procesar el Array = np.flipud(Array)
    '''
    # np.flip(preci[0].T,axis=0)

    # Para encontrar el formato de GDAL
    NP2GDAL_CONVERSION = {
        "uint8": 1,
        "int8": 1,
        "uint16": 2,
        "int16": 3,
        "uint32": 4,
        "int32": 5,
        "float32": 6,
        "float64": 7,
        "complex64": 10,
        "complex128": 11}

    gdaltype = NP2GDAL_CONVERSION[Array.dtype.name]

    # get the unique coordinates
    uniqueLats = np.unique(lat)
    uniqueLons = np.unique(lon)

    # get Dimensions: number of columns and rows from coordinates
    ncols = len(uniqueLons)
    nrows = len(uniqueLats)

    # determine pixelsizes
    ys = uniqueLats[1] - uniqueLats[0]
    xs = uniqueLons[1] - uniqueLons[0]

    # determine origin
    originX = np.min(uniqueLons)-xs/2
    originY = np.max(uniqueLats)+ys/2

    # set the SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    # That's (top left x, w-e pixel resolution, rotation (0 if North is up),
    #         top left y, rotation (0 if North is up), n-s pixel resolution)
    transform = (originX, xs, 0, originY, 0, -ys)

    # set the coordinate system
    target = osr.SpatialReference()
    target.ImportFromEPSG(EPSG)

    # set driver
    driver = gdal.GetDriverByName(str('GTiff'))

    # NOMBRE del TIFF o del RASTER
    outputDataset = driver.Create(filename+".tif", ncols, nrows, 1, gdaltype)

    # add some metadata
    # outputDataset.SetMetadata( {'time': str(timedate), 'someotherInfo': 'lala'} )
    outputDataset.SetGeoTransform(transform)
    outputDataset.SetProjection(target.ExportToWkt())
    outputDataset.GetRasterBand(1).WriteArray(Array)  # ARRAY CAMBIAR
    outputDataset.GetRasterBand(1).SetNoDataValue(-9999)
    outputDataset.FlushCache()
    outputDataset = None

# catalogo_ideam = pd.read_excel(r"C:\Users\Camilo\Proyectos\Hidrologia\CNE_IDEAM.xls")


def style_df(df, copy=False, **kwargs):
    '''Lo vuelve todo en mayuscula y le quita las tildes
        kwargs -> remove_col='string'
    '''
    df.rename(columns=lambda x: x.upper(), inplace=True)
    remove_col = kwargs.get('remove_col')
    cols = df.select_dtypes(include=[np.object]).columns
    cols = np.setdiff1d(cols, remove_col)
    df.loc[:, cols] = df.loc[:, cols].apply(lambda x: x.str.upper())
    df.loc[:, cols] = df.loc[:, cols].apply(
        lambda x: x.str.replace('ñ', '$$$', regex=False), )
    df.loc[:, cols] = df.loc[:, cols].apply(lambda x: x.str.normalize(
        'NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))
    df.loc[:, cols] = df.loc[:, cols].apply(lambda x: x.str.replace(
        '$$$',  'Ñ', regex=False), )  # Para que no quite la ñ

    return df


def leer_dhime(path_str):
    """Lee archivos de diferentes etiquetas y nombre descargados del DHIME
    Args:
        path_str (String): Carpeta donde estan los archivos descrgados. Solo puede tener *.csv asociados al dhime

    Returns:
        [type]: [description]
    """
    csv_paths = list(
        Path(path_str).rglob('*.csv'))
    # Checking if Number of columns in each csv file are equal
    list_columns = np.array([list(pd.read_csv(csv_path).columns)
                             for csv_path in csv_paths])

    if not np.all(list_columns == list_columns[0]):
        raise ValueError('Los csv no tienen igual numero de columnas')

    all_data = pd.concat([pd.read_csv(
        csv_path, index_col='Fecha', parse_dates=True) for csv_path in csv_paths])
    etiquetas_disponibles = all_data.Etiqueta.unique()
    print('Las etiquetas disponibles son:', etiquetas_disponibles)

    dict_dhime = {}
    for etiqueta in etiquetas_disponibles:
        dict_estaciones = {}
        for lab, group in all_data.groupby(by=['Etiqueta', 'NombreEstacion']):
            if lab[0] == etiqueta:
                # lab[1] corresponde al nombre de la estacion
                df_estacion = group[['Valor']].copy()
                # Organizo en orden cronologico
                df_estacion.sort_index(inplace=True)
                # Elimino entradas con indice duplicado
                df_estacion = df_estacion[~df_estacion.index.duplicated(
                    keep='first')]
                dict_estaciones[lab[1]] = df_estacion
                dict_estaciones[lab[1]].columns = [lab[1]]
        dict_dhime[etiqueta] = dict_estaciones

    return dict_dhime


class Leer_Ideam:
    '''
    Datos de PQRS del IDEAM.
    Tener cuidado con la info mensual de IDEAM que un dato 01/12/1970 en realidad es el mes de noviembre de 1970
    PQRS_MENSUAL_PPT = True solamente cuando el pqrs es mensual y tiene el problema mencionado arriba.
    '''

    catalogo_ideam = pd.read_excel(
        r"C:\Users\Camilo\Proyectos\Hidrologia\CNE_IDEAM.xls")
    style_df(catalogo_ideam, remove_col='TECNOLOGIA')

    def __init__(self, folder_PQRS, freq, PQRS_MENSUAL_PPT=True, formato='data'):
        self.folder = folder_PQRS
        self.rutas = self.create_df_rutas()
        self.get_catalogo()
        self.etiquetas = self._get_etiquetas()
        self.freq = freq
        self.PQRS_MENSUAL_PPT = PQRS_MENSUAL_PPT
        self.catalogo = style_df(pd.merge(self._catalogo, self.create_df_rutas(
            formato=formato), left_on='CODIGO', right_on='codigo', how='right'), remove_col='RUTAS')
#         self.clip_catalogo()

    def _get_etiquetas(self):
        return self.rutas.etiqueta.unique()

    def analizar(self, create_catalog=False, **kwargs):
        df = self.find_station(read=False, **kwargs).copy()
        # para ver cuantas estaciones tienen la misma ubicacin
        df.pipe(Leer_Ideam.add_repetido)
        if create_catalog:
            return df.pipe(Leer_Ideam.create_dataframe)
        else:
            return df

    def find_station(self, read=False, **kwargs):
        """[Encontrar las estaciones]

        Ejemplo: find_station(nombre='Mapi', categoria = 'limn')
            si hay espacios en el nombre de la columna: find_station(**{'CON ESPACIOS':'limni'})
        Si hay varias que cumplen, siempre lee la primera
        Args:
            read (bool, optional): [description]. Defaults to False.
        """
        df_booleans = {}
        for key in kwargs.keys():
            df_booleans[key] = self.catalogo[key.upper()].str.contains(
                kwargs.get(key), case=False)
        # Se determina las filas que no continen ningun valor de la KEY
        df_posiciones = pd.DataFrame(df_booleans).all(axis=1, skipna=False)
        df_posiciones[df_posiciones.isna()] = False

        if df_posiciones.sum() > 0:
            catalogo_query = self.catalogo.loc[df_posiciones, :]
            if read:
                return catalogo_query.pipe(Leer_Ideam.read_data, self.freq)
            else:
                return catalogo_query
        else:
            print('No hay ninguna estacion')

    def create_df_rutas(self, formato='data'):
        '''Crear un dataframe con las rutas en que las columnas son ruta, etiqueta y codigo
            Input: rutas PAth o string 
            formato: 'data', 'xlsx'
        '''
        df_rutas = list((self.folder).glob(f'**\*.{formato}'))
        if isinstance(df_rutas[0], Path):
            df_rutas = list(map(str, df_rutas))
        rutas_serie = pd.Series(df_rutas)
        rutas_serie.name = 'rutas'
        rutas_serie = rutas_serie.astype('str')
        df_rutas = pd.concat([rutas_serie, rutas_serie.str.extract(
            fr'.*\\+(\w+)@(\d{{8,11}}).{formato}')], axis=1)
        df_rutas = df_rutas.rename(columns={0: 'etiqueta', 1: 'codigo'})
        df_rutas.codigo = df_rutas.codigo.astype('int64')
        print(df_rutas.groupby('etiqueta').count()[['rutas']])
        df_rutas.rutas = df_rutas.rutas.str.replace('data', 'xlsx')
        df_rutas.rutas = df_rutas.apply(lambda row: str(Path(
            row['rutas']).parent/'Datos_procesados'/row['etiqueta']/Path(row['rutas']).name), axis=1)
        return df_rutas

    def clip_catalogo(self, clip_shape):
        clip_shape = clip_shape.to_crs(epsg=4326).copy()
        self.shape = clip_shape.copy()
        gdf = Leer_Ideam.dftogdf(self.catalogo)
        loc_contains = []
        for i in range(len(gdf)):
            loc_contains.append(clip_shape.contains(
                gdf.geometry.iloc[i]).values[0])
        gdf = gdf.loc[loc_contains]
        self.catalogo = gdf

    def plot_catalogo(self):
        'Grafica las estaciones despues de hacer el clip'
        fig, ax = plt.subplots(figsize=(5, 5))
        self.catalogo.plot(ax=ax)
        self.shape.plot(ax=ax, facecolor="none",
                        edgecolor='black')
        ax.legend(['Estaciones'])

    def get_catalogo(self, catalogo=catalogo_ideam):
        df = catalogo.loc[catalogo.CODIGO.isin(
            self.rutas.codigo.values)].copy()
        df = style_df(df, copy=False)
        self._catalogo = df

    def crear_carpetas(self):
        df_rutas = self.rutas
        df_rutas.rutas = df_rutas.rutas.str.replace('xlsx', 'data')
        folder_save = Path(self.folder/'Datos_procesados')
        folder_save.mkdir(exist_ok=True)

        for lab, row in df_rutas.groupby(by='etiqueta'):  # LAB IS THE GROUP
            label_dict = row.to_dict(orient='records')
            for i in range(len(row)):
                folder_save2 = folder_save/lab
                folder_save2.mkdir(exist_ok=True)

                df = pd.read_csv(label_dict[i].get(
                    'rutas'), sep='|', parse_dates=True, index_col=0)

                if self.PQRS_MENSUAL_PPT:
                    df = df.shift(-1, freq='m')  # Si es mensual

                codigo_estacion = label_dict[i].get('codigo')
                nombre_estacion = Leer_Ideam.catalogo_ideam[Leer_Ideam.catalogo_ideam.CODIGO.astype(
                    'int64') == codigo_estacion].NOMBRE.values

                if nombre_estacion.size == 1:
                    df.columns = nombre_estacion
                else:
                    df.columns = [str(codigo_estacion) + '_valor']

                name_file = lab + '@' + str(codigo_estacion) + '.xlsx'
                df.to_excel(Path(folder_save2/name_file))

            print(f'La carpeta EXCEL {lab} fue creada')

    @staticmethod
    def read_data(df, freq):
        '''Funcion para usar en el pipe, lo q cambie aca cambia en la lectrua de los datos'''
        ruta = df.RUTAS.values[0]
        df_read = pd.read_excel(ruta, index_col='Fecha', parse_dates=True)
        return df_read.resample(freq).mean()

    @staticmethod
    def add_repetido(df_analizar):
        '''
        Añadir una fila llamada Misma Ubicacion si hay un codigo repetido
        '''
        df_analizar['MISMA_UBICACION'] = df_analizar.groupby(
            by=['LONGITUD', 'LATITUD'])['NOMBRE'].transform('count').values

    @staticmethod
    def dftogdf(df):
        '''Dataframe to GeoDataFrame'''
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.LONGITUD, df.LATITUD), crs="EPSG:4326")
        return gdf

    @staticmethod
    def create_dataframe(df):
        '''
        Lee las rutas y crea un dtaaframe donde las columnas son estaciones y las filas fechas con observaciones. Tidy Data
        '''
        return pd.concat([pd.read_excel(ruta, index_col='Fecha', parse_dates=True) for ruta in df.RUTAS],
                         axis=1,
                         join='outer',
                         verify_integrity=True)


def coherencia_hidrologica(df1, df2=None, ax=None, legend=True, lgd_kwd={}):
    """df2 tiene que esta aguas abajo o df1 la segunda columna es mas aguas abajo.

    Args:
        df1 ([type]): df de dos columnas con diferentes estaciones
        df2 ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.

    Returns:
        [tuple]: caudales_diff, ax
    """
    ########### Add a legend for labels ####################
    legend_labels = {"red": "Rendimiento Negativo",
                     "blue": "Rendimiento Positivo"}

    patches = [Patch(color=color, label=label)
               for color, label in legend_labels.items()]
    #######################################################
    if df2 is not None:
        df_caudales = pd.concat([df1, df2, ], axis=1)
    else:
        df_caudales = df1.copy()

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(10, 2.5))
    print(df_caudales.shape)
    if df_caudales.shape[1] != 1:
        caudales_diff = df_caudales.diff(axis=1)
        caudales_diff = caudales_diff.iloc[:, 1:]
        df = caudales_diff.iloc[:, [0]]
    else:
        df = df1.copy()

    df_plot = df[df > 0]
    markerline, stemline, baseline = ax.stem(df_plot.index, df_plot.iloc[:, 0], basefmt='k-',
                                             use_line_collection=True)
    plt.setp(stemline, linewidth=0.5, color='blue')
    plt.setp(markerline, color='k', markersize=0)

    df_plot = df[df < 0]
    markerline, stemline, baseline = ax.stem(df_plot.index, df_plot.iloc[:, 0], basefmt='k-',
                                             use_line_collection=True)
    plt.setp(stemline, linewidth=0.5, color='red')
    plt.setp(markerline, color='k', markersize=0, zorder=50)
    limites = df.dropna(how='all')
    ax.set_xlim(limites.index[0], limites.index[-1])
    ax.set_ylabel('Diferencia caudal $(m^3/s)$')
    ax.grid(alpha=1, lw=0.5)
    if legend:
        ax.legend(handles=patches, loc='lower center', bbox_to_anchor=(
            0.5, 1), ncol=2, bbox_transform=fig.transFigure if not ax else ax.transAxes,)
    ax.set_xlabel('Fecha'),
    return df, ax


def count_threshold(df, threshold=0.5, freq='m'):
    '''
    Cuenta el numero de casos consecutivos que cumplen un thershold
    por ahora solo menor e igual... mejorar. Sirve pa contar dias sin lluvia seguidos

    ejemplo:
    count_threshold(index_oni[['ANOM']],threshold = -0.5)

    https://stackoverflow.com/questions/37934399/identifying-consecutive-occurrences-of-a-value-in-a-column-of-a-pandas-dataframe

    Revisar:
    https://stackoverflow.com/a/45886579/13184790
    '''
    df2 = df.copy()
    try:
        df2.index = df2.index.to_timestamp()
    except:
        pass
    column_name = df2.columns[0]
    #Threshold######
    old_col_names = df2.columns
    df2.columns = ['column']
    df2.query('column>= @threshold', inplace=True)
    df2.columns = old_col_names
    # ttt
    df2.iloc[:, 0] = 1
    df2 = df2.resample(freq).mean()
    df2.columns = ['Count']  # Conteo de dias sin lluvia
    df2['consecutive'] = df2.Count.groupby(
        (df2.Count != df2.Count.shift()).cumsum()).transform('size') * df2.Count
    lim_inf = df2.loc[df2['Count'] != df2['Count'].shift()].dropna()
    lim_sup = df2.loc[df2['Count'] != df2['Count'].shift(-1)].dropna()
    summary = pd.concat([pd.Series(lim_inf.index.values), pd.Series(
        lim_sup.index.values), pd.Series(lim_sup.consecutive.values)], axis=1)
    summary.columns = ['Inicio', 'Fin', 'Recuento']
    summary.Recuento = summary.Recuento.astype('int64')
    summary = summary.sort_values(by='Recuento', ascending=False)
    # summary = summary.sort_values(by ='Inicio')
    return summary


def faltantes_columna(df):
    return pd.DataFrame(df.isna().sum(axis=0)).rename({0: "Num_faltantes"}, axis=1).reset_index().rename({'index': 'Estacion'}, axis=1)


def plot_disponible(df: pd.DataFrame, ax=None, freq='m', **kwgs):
    """Create a heatmap plot of the available information

    Args:
        df (pd.DataFrame): [description]
        ax ([type], optional): [description]. Defaults to None.
    """
    df_plot = df.resample(freq).count().T
    df_plot = df_plot.divide(df_plot.columns.day)*100

    if not ax:
        fig, ax = plt.subplots(figsize=(15, 4))
    ax = sns.heatmap(df_plot, annot=True, ax=ax, fmt=".0f", cbar=False, cmap='coolwarm_r',
                     mask=(df_plot == 100), linewidths=0.5, linecolor='lightgray', **kwgs)
    ax.set_xticklabels(df_plot.columns.strftime('%m/%y'))
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    ax.grid(False)


def boxplot_double(df, label1='Olaya', label2='Puente Pescadero', ax=None):
    '''Plot annual cycle of two series'''
    df.index.name = 'Fecha'
    df_join = df.reset_index().melt(id_vars='Fecha')
    df_join = df_join.set_index('Fecha')
    df_join['mes'] = df_join.index.month

    if not ax:
        fig, ax = plt.subplots(figsize=(10, 6))

    flierprops = dict(marker='o', markersize=2.5,
                      linestyle='none', markeredgecolor='none')
    boxprops = {"zorder": 10}

    sns.boxplot(x="mes", y="value", hue="variable", saturation=0.8, flierprops=flierprops, boxprops=boxprops,
                data=df_join, dodge=True, ax=ax, palette='deep', zorder=10)

    ax.set_ylabel('Caudal $[m^3/s]$')
    ax.grid(zorder=-50, alpha=0.4, lw=0.5, ls='--')

    # h, l = ax.get_legend_handles_labels()
    # leg = ax.legend(h, [label1, label2], ncol=1, loc='upper left', bbox_to_anchor=(0.01, 1), borderaxespad=0.5, borderpad=0,
    #                 facecolor='white', framealpha=1)
    # leg.get_frame().set_linewidth(0)

    ax.set_xticks(np.arange(0, 12))
    calendario = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                  'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    # calendario = ['Enero', 'Febreo', 'Marzo', 'Abril', 'Mayo', 'Junio',
    #               'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre, 'Diciembre']
    ax.set_xticklabels(calendario)
    plt.legend(title='Estación')
    # sns.despine(trim=True)


def leer_epm(path=None):
    """
    Lectura de datos de EPM del proyecto Hidroelectrico Ituango 
    Archivos con un ancho fijo

    Parametros
    --------------
    dist1: Distancia a la linea q contiene el dia uno
    dist2: Distancia del primer dato a la linea que contiene el año 6 o 7

    Return
    --------------
    df con todos los años
    """
    linesss = []
    file1 = open(path, 'r')
    Lines = file1.readlines()
    inicio_datos = []
    for num, line in enumerate(Lines):
        if '1' in line:  # si hay un uno en la linea es por q  ahi empiezan los datos
            # Le quito los espacios iniciales a cada linea y reviso que la primera posicion sea un uno y la segunda
            if line.strip()[0] == '1' and line.strip()[1] == ' ':
                linesss.append(line)
                inicio_datos.append(num)

    years = pd.Series(Lines).str.extractall('Year\s+(\d{4})').iloc[:, 0].values

    rows_list = inicio_datos.copy()
    sup = [15+i*9 for i in np.arange(0, 13)]
    inf = list(np.array(sup)-9)
    result = zip(inf, sup)
    col_specification = list(result)
    # print('---'*20)
    # print('Col especification\n')
    # print(col_specification)
    # print('---'*20)
    dictEPM = {}
    for i, row_initial in enumerate(rows_list):
        # print('Dato inicial en linea ---> '+str(row_initial))
        # Dos formas de encontrar el año
        # year_data = pd.read_fwf(path,colspecs=[(127,131)],header=None,skiprows=row_initial - dist2 , nrows=1).iloc[0,0]
        year_data = years[i]
        # print('YEAR')
        # print(year_data)
        # print('--'*10)
        data = pd.read_fwf(path, colspecs=col_specification,
                           header=None, skiprows=row_initial, nrows=31)
        data.rename(columns={0: 'day'}, inplace=True)
        data = data.melt(id_vars=['day'])
        data.variable = data.variable.astype('int')
        data.rename(columns={'variable': 'month'}, inplace=True)
        data['year'] = year_data
        data.loc[:, 'Fecha'] = data['day'].map(lambda x: str(
            x))+'/'+data['month'].map(lambda x: str(x))+'/'+data['year'].map(lambda x: str(x))
        data = data.dropna()
        data.Fecha = pd.to_datetime(data.Fecha, dayfirst=True, errors='coerce')
        data = data.loc[:, ['Fecha', 'value']]
        data.set_index(['Fecha'])
        dictEPM[year_data] = data
    df_concatenado = pd.concat([dictEPM[key]
                                for key in dictEPM.keys()], axis=0, join='outer')
    df_concatenado.set_index('Fecha', inplace=True)
    df_concatenado['value'] = df_concatenado['value'].astype('str')
    df_concatenado['value'] = df_concatenado['value'].str.extract(
        '(\d*\.\d+|\d+)', expand=False).astype(float)
    df_concatenado['value'] = pd.to_numeric(df_concatenado.value)
    df_concatenado = df_concatenado.resample('d').mean()
    # df_concatenado[:].plot()
    return df_concatenado


def corr_heatmap(df):
    corr = df.corr()
    # plot the heatmap
    fig, ax = plt.subplots(figsize=(20, 20))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    def fmt(x, pos): return '{:.2}'.format(x)
    sns.heatmap(corr, ax=ax, cbar_ax=cax, annot=True,
                xticklabels=corr.columns, cbar_kws={
                    'format': plt.FuncFormatter(fmt)},
                yticklabels=corr.columns, cmap='Blues', square=True,
                annot_kws={"size": 15})
    cbar = ax.collections[0].colorbar
    cbar.set_label("Correlacion", rotation=270, labelpad=45, fontsize=20)
    cbar.ax.tick_params(labelsize=15)
    cmap = sns.diverging_palette(5, 250, as_cmap=True)
    ax.set_yticklabels(['\n'.join(i).replace('32', '[32') for i in map(
        lambda x:x.get_text().split('[', 1), ax.get_yticklabels())], va='center')
    ax.set_xticklabels(['\n'.join(i).replace('32', '[32') for i in map(
        lambda x:x.get_text().split('[', 1), ax.get_xticklabels())], rotation=30)
    ax.tick_params(labelsize=12)
    return corr, fig


def filter_pro(df, **kwargs):
    ''' Filtra una tabla segun palabras sin tener en cuenta mayusculas ni que la palabra este completa'''

    df.columns = df.columns.map(str.lower)
    df_booleans = {}
    for key in kwargs.keys():
        df_booleans[key] = df[key.lower()].str.contains(
            kwargs.get(key), case=False)
    # Se determina las filas que no continen ningun valor de la KEY
    df_posiciones = pd.DataFrame(df_booleans).all(axis=1)
    return df.loc[df_posiciones, :]


print('Funcionando')
