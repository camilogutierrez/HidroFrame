from collections import defaultdict
from math import log
from operator import index

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy import stats
import scipy.stats as scs
import seaborn as sns
from scipy.stats import mannwhitneyu, norm, t
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
# if __name__ == '__main__':
#     from main import estandarizar_df
# else:
#     from .main import estandarizar_df
from .main import cicloanual_mensual_plot, estandarizar_df


def ptocambio(datos, label,ax=None, month=None, year=None, save=False, 
              legend=True, estacion = '', ):
    """Determine where to divide the series, by default is in the middle of it. 

    Args:
        datos ([type]): [description]
        month ([integers], optional): [description]. Defaults to None.
        year ([integers], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    datos.columns = ['value']
    if isinstance(datos.index, pd.DatetimeIndex):
        datos.index = datos.index.to_period()
    if month == None:
        fecha_mitad = datos.index[int(len(datos.index)/2)]
        month = fecha_mitad.month
        year = fecha_mitad.year
    lim_subserie1 = pd.Period(year=year, month=month, freq='m')
    lim_subserie2 = lim_subserie1.to_timestamp() + pd.DateOffset(months=1)
    df1 = datos[:lim_subserie1].copy()
    df2 = datos[lim_subserie2:].copy()
    #     assert (len(df1) + len(df2) == len(serie_mensual))
    fig, [ax, ax2] = plt.subplots(1, 2, figsize=(
        8, 3.5), sharey=True, gridspec_kw={'width_ratios': [2, 0.5]})

    pos_x = len(df1) / len(datos)
    lw2 = 0.8
    lwh = 0.9+0.5

    colorline_df1 = 'midnightblue'
    colorline_df2 = 'k'
    df1.dropna().plot(ax=ax, lw=lw2, alpha=0.75, legend=False,marker='.')
    ax.axhline(df1.mean()[0], xmax=pos_x,
               color=colorline_df1, label='$\mu$ Subserie', lw=lwh)
    ax.axhline(df1.mean()[0]-df1.std()[0], linestyle='--',
               xmax=pos_x, color=colorline_df1, lw=lwh)
    ax.axhline(df1.mean()[0]+df1.std()[0], linestyle='--', xmax=pos_x,
               color=colorline_df1, label='$\mu \pm \sigma$ Subserie', lw=lwh)

    df2.dropna().plot(ax=ax, color='dimgrey', lw=lw2, alpha=0.75, legend=False, marker='.')
    ax.axhline(df2.mean()[0], linestyle='-', xmin=pos_x,
               color=colorline_df2, label='$\mu$ Subserie 2', lw=lwh)
    ax.axhline(df2.mean()[0]-df2.std()[0], linestyle='--',
               xmin=pos_x, color=colorline_df2, lw=lwh)
    ax.axhline(df2.mean()[0]+df2.std()[0], linestyle='--', xmin=pos_x,
               color=colorline_df2, label='$\mu \pm \sigma$ Subserie 2', lw=lwh)
    ax.set_ylabel(label)
    # plt.ylabel('Precipitacion total  (mm/mes)',fontsize=fs)
    ax.set_xlabel('Fecha ')
    ax.set_xlim([df1.index[0], df2.index[-1]])
    if legend:
        h, l = ax.get_legend_handles_labels()
        l[0] = 'Subserie'
        l[3] = 'Subserie 2'
        # for  l_i in h:
            # l_i.set_color('black')
        leg = ax.legend(h[:3], l[:3],  ncol=3, loc='upper center', fontsize=8)

    flierprops = dict(marker='.', markersize=2.5,
                      linestyle='none', markeredgecolor='gray')

    bp = ax2.boxplot([df1.dropna().values.ravel(),
                      df2.dropna().values.ravel()], widths=0.65, flierprops=flierprops)

    for i, box in enumerate(bp['boxes']):
        # change outline color
        #         box.set(color='k', linewidth=0.7)
        if i == 0:
            box.set(color=colorline_df1, lw=0.5)
        else:
            box.set(color=colorline_df2, lw=0.75)

    for i, box in enumerate(bp['medians']):
        # change outline color
        #         box.set(color='k', linewidth=0.7)
        if i == 0:
            box.set(color=colorline_df1, linewidth=1.4)
        else:
            box.set(color=colorline_df2, linewidth=1.4)

    for i, box in enumerate(bp['fliers']):
        # change outline color
        #         box.set(color='k', linewidth=0.7)
        if i == 0:
            box.set(markeredgecolor=colorline_df1,
                    lw=0.2, marker='D', markersize=2.1)
        else:
            box.set(markeredgecolor=colorline_df2,
                    lw=0.4,  marker='D', markersize=2.1)

    for i, box in enumerate(bp['whiskers']):
        if i < 2:
            box.set(color=colorline_df1, linewidth=0.75)
        else:
            box.set(color=colorline_df2, linewidth=0.75)

    for i, box in enumerate(bp['caps']):
        if i < 2:
            box.set(color=colorline_df1, linewidth=0.75)
        else:
            box.set(color=colorline_df2, linewidth=0.75)

    ax2.set_xticklabels(['Sub\nserie1', 'Sub\nserie2'])
    fig.suptitle('{} - PC: {:02d}-{}'.format(estacion,month, year), y=0.92)
    # fig.tight_layout()
    ax.annotate('a) ', xy=(0.01, 0.95),  xycoords='axes fraction',
                horizontalalignment='left', verticalalignment='bottom', size=8)

    ax2.annotate('b) ', xy=(0.05, 0.95),  xycoords='axes fraction',
                 horizontalalignment='left', verticalalignment='bottom', size=8)
    # ax2.tick_params(rotation=30)
    fig.subplots_adjust(wspace=0)
    punto_de_cambio = '{:02d}-{}'.format(month, year)

    fig.subplots_adjust(wspace=0.05)
    fig.savefig(f'{estacion}.png', bbox_inches='tight', dpi=100)
    return df1, df2, fig, punto_de_cambio

# TODO ESTO


def plotautocorr(serie, rezagos, function=scs.pearsonr, alpha=0.05):
    corr = []
    limInf = []
    limSup = []
    A_rezagos = np.arange(1, rezagos, 1)
    for k in A_rezagos:
        serie1 = serie[abs(min(k, 0)): (len(serie) - 1) - max(k, 0)]
        serie2 = serie[max(k, 0): (len(serie) - 1) - abs(min(k, 0))]
        corr.append(function(serie1, serie2)[0])
        N = len(serie1)
        limInf.append((-1-norm.ppf(1-alpha/2)*(N-k-1)**0.5)/(N-k))
        limSup.append((-1+norm.ppf(1-alpha/2)*(N-k-1)**0.5)/(N-k))
    x, y, z, t = np.r_[0, A_rezagos], np.r_[
        1, np.array(corr)], np.r_[-1, limInf], np.r_[1, limSup]
    plt.figure()
    markerline, stemline, baseline, = plt.stem(
        x, y, linefmt='grey', markerfmt='D')

    plt.setp(stemline, linewidth=1.25)
    plt.setp(markerline, markersize=4)

    plt.plot(x, z, ':', color='black', label='Límites')
    plt.plot(x, t, ':', color='black')
    plt.xlabel('Rezagos (meses)',)
    plt.xticks(np.arange(0, 13),)
    # plt.yticks(fontsize=14)
    plt.ylabel('Correlación',)
    # plt.title('Autocorrelograma de anderson')
    plt.legend()
    plt.savefig('Seriescorrel.png', bbox_inches='tight', dpi=100)
    # plt.plot(x,y)


def Ansari(df1, df2):
    print('Ansari-Bradley Test')
    yt1 = df1.values.ravel()
    yt2 = df2.values.ravel()
    stadistico_ansari, pvalue_ansari = scs.ansari(yt1, yt2)
    if pvalue_ansari <= 0.05:
        print("Se rechaza Ho: hipótesis de igualdad de las dos varianzas")
        resultado = 'R'
    else:
        print("Se acepta Ho: hipótesis de igualdad de las dos varianzas-Estadistico={stad:.5f}-Valor P={pval:.5f}".format(
            stad=float(stadistico_ansari), pval=pvalue_ansari))
        resultado = 'A'
    return resultado, pvalue_ansari


def barlett(df1, df2):
    print('Barlett Test')
    yt1 = df1.dropna().values.ravel()
    yt2 = df2.dropna().values.ravel()
    stadistico_b, pvalue_b = scs.bartlett(yt1, yt2)
    if pvalue_b <= 0.05:
        print("Se rechaza Ho: hipótesis de igualdad de las dos varianzas - Estadistico = {stad:.5f}-Valor P={pval:.5f}".format(
            stad=float(stadistico_b), pval=pvalue_b))
        resultado = 'R'
    else:
        print("Se acepta Ho-: hipótesis de igualdad de las dos varianzas-Estadistico={stad:.5f}-Valor P={pval:.5f}".format(
            stad=float(stadistico_b), pval=pvalue_b))
        resultado = 'A'
    return resultado, pvalue_b


def Levene(df1, df2):
    print('Levene Test')
    yt1 = df1.dropna().values.ravel()
    yt2 = df2.dropna().values.ravel()
    stadistico_l, pvalue_l = scs.levene(yt1, yt2)
    resultado = ''
    if pvalue_l <= 0.05:
        print("Se rechaza Ho: hipótesis de igualdad de las dos varianzas ->\tEstadistico={stad:.2f}-Valor P={pval:.3f}".format(
            stad=float(stadistico_l), pval=pvalue_l))
        resultado = 'R'
    else:
        print("Se acepta Ho: hipótesis de igualdad de las dos varianzas-Estadistico={stad:.5f}-Valor P={pval:.5f}".format(
            stad=float(stadistico_l), pval=pvalue_l))
        resultado = 'A'
    return resultado, pvalue_l

    #


def analizar_outlayers(df_analizar, desviaciones=3.5):
    """[summary]

    Args:
        df_analizar ([type]): [description]
        desviaciones (float, optional): [description]. Defaults to 3.5.

    Returns:
        [type]: [description]
    """    
    df = df_analizar.copy()
    df.columns = ['value']
    fig, [ax, ax1] = plt.subplots(2, 1, sharex=True)
    df.plot(ax=ax)
    mayor = estandarizar_df(df).value > desviaciones
    menor = estandarizar_df(df).value < -desviaciones
    
    if mayor.sum() > 0:
        df[mayor].plot(ax=ax, lw=0, marker='*', legend=False, c='red')
    if menor.sum() > 0:
        df[menor].plot(ax=ax, lw=0, marker='*', legend=False, c='red')

    def outlayer(inicio_outlayer, df_analizar=None, fin_outlayer='', df=df, N=3):
        '''
        Analiza un dato o varios datos qsi al removerlos siguen siendo outlayers o no.
        '''
        df = df_analizar.copy()
        if not fin_outlayer:
            fin_outlayer = inicio_outlayer
        outlayer = []
        df_return = df[(df.index < inicio_outlayer) | (
            df.index > fin_outlayer)]  # YY MM DD
        lim_inf = df_return.groupby(lambda x: x.month).mean(
        ) - N * df_return.groupby(lambda x: x.month).std()
        lim_sup = df_return.groupby(lambda x: x.month).mean(
        ) + N * df_return.groupby(lambda x: x.month).std()
        df_mask = df[(df.index >= inicio_outlayer) &
                     (df.index <= fin_outlayer)].copy()
        for lab, row in df_mask.iterrows():
            if (row[0] < lim_inf.iloc[lab.month-1, 0]) | (row[0] > lim_sup.iloc[lab.month-1, 0]):
                df_mask.loc[lab] = np.nan
                print(lab)
                outlayer.append(lab)
        df_return = pd.concat([df_return, df_mask], axis=0)
        df_return = df_return.loc[~df_return.index.duplicated(keep='first')]
        df_return.sort_index(inplace=True)
        return outlayer

    df_outlayers = df.copy()
    total_outlayers = []
    for inicio_outlayer in mayor[mayor].index:
        dato_outlayer = outlayer(
            inicio_outlayer, df_analizar=df_outlayers, N=desviaciones)
        print('***'*10 + '\nDato Outlayer:\n'+str(dato_outlayer)+'\n'+'-'*10)
        if len(dato_outlayer) > 0:
            df_outlayers.loc[dato_outlayer] = np.nan
            total_outlayers.append(dato_outlayer)
    for inicio_outlayer in menor[menor].index:
        dato_outlayer = outlayer(
            inicio_outlayer, df_analizar=df_outlayers, N=desviaciones)
        print('***'*10 + '\nDato Outlayer:\n'+str(dato_outlayer)+'\n'+'-'*10)
        if len(dato_outlayer) > 0:
            df_outlayers.loc[dato_outlayer] = np.nan
            total_outlayers.append(dato_outlayer)

    ax.set_title('Posibles Outlayers')
    df_outlayers.plot(ax=ax1, title='Sin Outlayers')
    fig.tight_layout()
    return total_outlayers


def MannWhitney_test(df1, df2):
    stat, pvalue = mannwhitneyu(df1, df2)
    print('Statistics=%.3f, p=%.3f' % (stat, pvalue))
    alpha = 0.05
    if pvalue > alpha:
        print('Same distribution (fail to reject H0)')
        resultado = 'A'
    else:
        print('Different distribution (reject H0)')
        resultado = 'R'
    return resultado, pvalue

# Analisis de tendencia


def MKendall_original(df):
    tendencia_dict = {'increasing': '(+)',
                      'decreasing': '(-)',
                      'no trend': 'NA'}
    trend, h, pvalue, z, Tau, s, var_s, slope, intercept = mk.original_test(df)
    trend = tendencia_dict[trend]
    return trend, pvalue, slope


def MKendall_hamed(df):
    tendencia_dict = {'increasing': '(+)',
                      'decreasing': '(-)',
                      'no trend': 'NA'}
    trend, h, pvalue, z, Tau, s, var_s, slope, intercept = mk.hamed_rao_modification_test(
        df)
    trend = tendencia_dict[trend]
    return trend, pvalue, slope


def get_homogeneidad(df, homogeneidad_final, puntos_cambio, estaciones,estacion, label, **kwgs):
    # df_m = df.resample('m').mean()
    estaciones.append(df.columns[0])
    df1, df2, _, punto_de_cambio = ptocambio(df,estacion = estacion, label = label,   **kwgs)


    df.dropna(inplace=True)
    df1.dropna(inplace=True)
    df2.dropna(inplace=True)

    puntos_cambio.append(punto_de_cambio)

    resultado, valorp = MannWhitney_test(df1, df2)
    homogeneidad_final['Media']['MannWhitney_test']['Resultado'].append(
        resultado)
    homogeneidad_final['Media']['MannWhitney_test']['Valor P'].append(valorp)

    resultado, valorp = Ansari(df1, df2)
    homogeneidad_final['Varianza']['Ansari']['Resultado'].append(resultado)
    homogeneidad_final['Varianza']['Ansari']['Valor P'].append(valorp)

    resultado, valorp = barlett(df1, df2)
    homogeneidad_final['Varianza']['Barlett']['Resultado'].append(resultado)
    homogeneidad_final['Varianza']['Barlett']['Valor P'].append(valorp)

    resultado, valorp = Levene(df1, df2)
    homogeneidad_final['Varianza']['Levene']['Resultado'].append(resultado)
    homogeneidad_final['Varianza']['Levene']['Valor P'].append(valorp)

    trend, valorp, slope = MKendall_original(df)
    homogeneidad_final['Tendencia']['Mann Kendall']['Tendencia'].append(trend)
    homogeneidad_final['Tendencia']['Mann Kendall']['Valor P'].append(valorp)
    homogeneidad_final['Tendencia']['Mann Kendall']['Magnitud'].append(slope)

    trend, valorp, slope = MKendall_hamed(df)
    homogeneidad_final['Tendencia']['Hamed Rao']['Tendencia'].append(trend)
    homogeneidad_final['Tendencia']['Hamed Rao']['Valor P'].append(valorp)
    homogeneidad_final['Tendencia']['Hamed Rao']['Magnitud'].append(slope)

def get_homogeneidad_final():
    homogeneidad_dict = defaultdict(dict)

    for prueba in ['MannWhitney_test']:
        homogeneidad_dict['Media'][prueba] = {'Resultado': [], 'Valor P': []}

    for prueba in ['Ansari', 'Barlett', 'Levene']:
        homogeneidad_dict['Varianza'][prueba] = {
            'Resultado': [], 'Valor P': []}

    for prueba in ['Mann Kendall', 'Hamed Rao']:
        homogeneidad_dict['Tendencia'][prueba] = {
            'Tendencia': [], 'Valor P': [], 'Magnitud': []}
        
    return dict(homogeneidad_dict)

def homogeneidad_iter(df_analizar):
    """[summary]

    Args:
        df_analizar ([type]): [description]

    Returns:
        [type]: [description]
    """
    puntos_cambio = []
    estaciones = []

    homogeneidad_final = get_homogeneidad_final()

    for lab, row in df_analizar.iterrows():
        df = pd.read_excel(row['RUTAS'], index_col='Fecha', parse_dates=True)
        print('*'*50+'\n\t'+row['NOMBRE'])
        get_homogeneidad(df, homogeneidad_final, puntos_cambio, estaciones)

    return homogeneidad_final, estaciones, puntos_cambio

def analizar_homogeneidad(df_analizar,label = 'Caudal  ($m^3$/s)', **kwgs):
    df = df_analizar.copy()
    puntos_cambio = []
    estaciones = []
    homogeneidad_final = get_homogeneidad_final()

    for col in df.columns:       
        df_serie = df[col].copy()
        min_date, max_date  = pd.Series(df_serie.index[~df_serie.isna()]).agg(['min', 'max'])
        get_homogeneidad(df_serie.loc[min_date:max_date].to_frame(), homogeneidad_final, puntos_cambio, estaciones,col, label)
          
    df_homogeneidad = pd.DataFrame(get_tuple(homogeneidad_final))
    df_homogeneidad.index = pd.MultiIndex.from_arrays((estaciones, puntos_cambio))
    return df_homogeneidad

def plot_curva_masa(df, freq=None, ax = None):
    df = df.resample(freq).mean()
    if not ax:
        fig, ax = plt.subplots()
    df.cumsum().plot(ax=ax, legend=False)
    ax.set_ylabel("Caudal  $m^3/s$", labelpad=-0.5)
    ax.set_aspect(1/ax.get_data_ratio(), adjustable='box')
    return ax


class Periodos:
    nina_years = pd.Series([1955, 1964, 1970, 1971, 1973, 1974, 1975, 1983, 1984, 1988, 1995,
                            1998, 1999, 2000, 2005, 2007, 2008, 2010, 2011, 2016, 2017])
    nino_years = np.array([1952, 1953, 1957, 1958, 1963, 1965, 1968, 1969, 1972, 1976, 1977,
                           1979, 1982, 1986, 1987, 1991, 1994, 1997, 2002, 2004, 2006, 2009,
                           2014, 2015, 2018, 2019])

    periodo_inicio = []
    periodo_fin = []

    for year in nina_years:
        inicio = f'{year}-05-01'
        fin = f'{year+1}-04'
        inicio = pd.Period(inicio).to_timestamp(freq='ms')
        periodo_inicio.append(inicio)
        fin = pd.Period(fin).to_timestamp(freq='m')
        periodo_fin.append(fin)

    df_nina = pd.DataFrame(
        {'periodo_inicio': periodo_inicio, 'periodo_fin': periodo_fin})

    periodo_inicio = []
    periodo_fin = []

    for year in nino_years:
        inicio = f'{year}-05-01'
        fin = f'{year+1}-04'
        inicio = pd.Period(inicio).to_timestamp(freq='ms')
        periodo_inicio.append(inicio)
        fin = pd.Period(fin).to_timestamp(freq='m')
        periodo_fin.append(fin)
    df_nino = pd.DataFrame(
        {'periodo_inicio': periodo_inicio, 'periodo_fin': periodo_fin})

    # @staticmethod
    def get_days(row):
        '''hola'''
        start = row['periodo_inicio']
        end = row['periodo_fin']
        return pd.Series(pd.date_range(start=start, end=end))

    fecha1 = df_nina.apply(get_days, axis=1).melt().iloc[:, 1].sort_values(
    ).dropna().to_frame('Fecha').set_index('Fecha').assign(Enso='Niña')
    fecha2 = df_nino.apply(get_days, axis=1).melt().iloc[:, 1].sort_values(
    ).dropna().to_frame('Fecha').set_index('Fecha').assign(Enso='Niño')
    ENSO = pd.concat([fecha1, fecha2])

def ciclo_anual_enso(df_plot, ax=None):
    df = df_plot.join(Periodos.ENSO.pipe(lambda df:df.to_period(freq='m')))
    df.loc[df.Enso.isnull(),'Enso']='Neutro'
    if not ax:
        fig, ax = plt.subplots()
    cicloanual_mensual_plot(df.groupby(by = df.Enso).get_group('Niña').iloc[:,[0]],ax=ax,ls='--', marker='.',lw=0.5)
    cicloanual_mensual_plot(df.groupby(by = df.Enso).get_group('Niño').iloc[:,[0]],ax=ax,color='red',ls='--', marker='.',lw=0.5)
    # cicloanual_mensual_plot(df.groupby(by = df.Enso).get_group('Neutro').iloc[:,[0]],ax=ax, bands=False,color='k', marker='.',lw=0.5)
    cicloanual_mensual_plot(df_plot,ax=ax, bands=False,color='k', marker='.',lw=0.5)
    ax.legend(['Niña','Niño','Historico'])
    # fig.savefig('ciclos_enso.pdf',bbox_inches='tight',pad_inches=0)
    
def ftest(df1,df2): 
    a = df1.values.ravel()
    b = df2.values.ravel()
    print('Variance Subserie1={0:.2f}, Variance Subserie 2={1:.2f}\n'.format(np.var(a, ddof=1), np.var(b, ddof=1)))
    fstatistics = max(np.var(a, ddof=1), np.var(b, ddof=1))/min(np.var(a, ddof=1), np.var(b, ddof=1)) # because we estimate mean from data
    
    fdistribution = stats.f(len(a)-1,len(b)-1) # build an F-distribution object 
    p_value = 1-fdistribution.cdf(fstatistics)
    
    f_critical1 = fdistribution.ppf(0.025)
    f_critical2 = fdistribution.ppf(0.975)
#     print(fstatistics,f_critical1, f_critical2 )
    if (p_value<0.05):
        print('Reject H0', p_value)
    else:
        print('Cant Reject H0', p_value)
    print('')
    if (f_critical1<=fstatistics<=f_critical2):
        print('{}<{}<{}'.format(f_critical1,fstatistics,f_critical2))
        print('Se acepta la hipotesis de igualdad de varianza - Valor p=', p_value)
    else:
        print('{}>{}>{}'.format(f_critical1,fstatistics,f_critical2))
        print('Se rechaza la hipotesis de igualdad de varianza - Valor p=', p_value)
    
def organizar_periodos(periodos_df_nino):
    """Organizar los periodos consecutivos de años niña y nino

    Args:
        periodos_df_nino ([type]): [description]

    Returns:
        [type]: [description]
    """    
    df = periodos_df_nino.copy()
    df['periodo_fin_siguiente'] = df['periodo_inicio'].shift(-1)
    df_filter = df[['periodo_fin','periodo_fin_siguiente']].diff(axis=1)['periodo_fin_siguiente'].dt.days
    df['periodo_fin'] =df['periodo_fin'].where(df_filter!=1).fillna(method='bfill')
    df['periodo_inicio'] = df.groupby('periodo_fin')['periodo_inicio'].transform(lambda x:x.min())
    df.drop_duplicates(subset='periodo_fin', keep='last', inplace=True)
    dias_dif = df[['periodo_inicio','periodo_fin']].diff(axis=1)['periodo_fin'].dt.days
    df['dias'] = dias_dif
    return df.reset_index(drop=True)[['periodo_inicio','periodo_fin']]

def plot_time_series_ENSO(df_plot, outliers=None):

    fig, ax =  plt.subplots(figsize = (6,3))

    df_plot.plot(ax=ax, color='k', lw = 1, zorder = 0)

    for label,row  in organizar_periodos(Periodos.df_nina).iterrows():
        ax.fill_betweenx(x1 = row.periodo_inicio,x2 = row.periodo_fin, y=[-1,1], facecolor='blue',alpha = 0.1,
                         transform = ax.get_xaxis_transform(), zorder = -5)

    for label,row  in organizar_periodos(Periodos.df_nino).iterrows():
        ax.fill_betweenx(x1 = row.periodo_inicio,x2 = row.periodo_fin, y=[-1,1], facecolor='red',alpha = 0.1, 
                         transform = ax.get_xaxis_transform(), zorder = -5)   
    if outliers:    
        total_outlayers = analizar_outlayers(df_plot, desviaciones=outliers)
        outliers_df =  df_plot[np.isin(df_plot.index,total_outlayers)]   
        outliers_df.plot(ax=ax, lw=0, marker='*',markersize=8, c='red', zorder = 100)
    ax.set_ylabel('Caudal ($m^3/s$)')
    ########### Add a legend for labels ####################
    legend_elements = [Line2D([0], [0], marker='*', color='w', label='Outlier' ,markerfacecolor='red', markersize=15)]

    legend_labels = {"red": "Niño", "blue": "Niña"}
    patches = [Patch(color=color, label=label, alpha =0.25)
               for color, label in legend_labels.items()]
    legend_elements = legend_elements+patches
    
    # ax.legend(handles=legend_elements, loc = 'upper center',ncol=3,bbox_to_anchor=(0.5,1.05),bbox_transform=fig.transFigure,)
    ax.legend(handles=legend_elements, loc = 'upper left')
    ###########
    return fig
    # fig.savefig('outlayer.pdf', pad_inches= 0.025)

def get_tuple(dictionary):
    '''Get tuple from nested dictionary to create a multininex column Dataframe'''
    reform = {(level1_key, level2_key, level3_key): values
                   for level1_key, level2_dict in dictionary.items()
                   for level2_key, level3_dict in level2_dict.items()
                   for level3_key, values      in level3_dict.items()}
    return reform

class IndicesEnso(object):
    print('s')
    def __init__(self):
        self.camilo=5
        self._pdo = self.get_pdo()
        
    @property
    def pdo(self):
        return self._pdo
    
    def get_pdo(self):
        index_pdo = pd.read_csv('https://www.ncdc.noaa.gov/teleconnections/pdo/data.csv', header=1)
        index_pdo.Date = index_pdo.Date.astype('str')
        index_pdo.loc[:,'year'] = index_pdo.Date.str.extract('^(\d{4})').iloc[:,0]
        index_pdo.loc[:,'month'] =index_pdo.Date.str.extract('(\d{2})$').iloc[:,0]
        index_pdo.Date = pd.to_datetime(index_pdo.apply(lambda row: str(row.year)+'-'+str(row.month), axis=1))
        index_pdo = index_pdo.set_index('Date')[['Value']]
        return index_pdo
    
    def get_oni(self):
        index_oni = pd.read_csv('https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt',delim_whitespace=True)
        index_oni.tail(8)
        index_oni['Fecha']=pd.date_range(start='01-1950',periods = len(index_oni),freq='MS')
        index_oni.Fecha = index_oni.Fecha.dt.to_period()
        index_oni.set_index('Fecha',inplace=True)
        index_oni.head()
        self.oni = index_oni['ANOM']
        
    # MEI_V2=pd.read_csv('https://psl.noaa.gov/enso/mei/data/meiv2.data',
    #                delim_whitespace=True,skiprows=1,header=None,error_bad_lines=False)
    # MEI_V2 = MEI_V2.iloc[0:42,:]
    # MEI_V2 = MEI_V2.melt(id_vars=0)
    # MEI_V2.columns = ['year','month','MEI_V2']
    # MEI_V2
    # MEI_V2['Date'] = pd.to_datetime(MEI_V2.apply(lambda row: str(row.year)+'-'+str(row.month), axis=1))
    # MEI_V2.set_index('Date',inplace=True)
    # MEI_V2.sort_index(inplace = True)
    # MEI_V2.index = MEI_V2.index.to_period()
    # MEI_V2 = MEI_V2[['MEI_V2']]
    # MEI_V2.MEI_V2 = MEI_V2.MEI_V2.astype('float')

print('melo')
