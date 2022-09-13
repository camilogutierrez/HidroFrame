import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import unicodedata
#### DATAFRAMES MANIPULATIONS ####




def movecol(df, cols_to_move=[], ref_col='', place='After'):
    '''Reordering Pandas DataFrame Columns
    cols_to_move must be a list
    '''

    cols = df.columns.tolist()
    if place == 'After':
        seg1 = cols[:list(cols).index(ref_col) + 1]
        seg2 = cols_to_move
    if place == 'Before':
        seg1 = cols[:list(cols).index(ref_col)]
        seg2 = cols_to_move + [ref_col]

    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols if i not in seg1 + seg2]

    return(df[seg1 + seg2 + seg3])


def find(df, string):
    '''Find the columns that match the pattern
    return:df filtered

    df.pipe(find,args)

    for more than one columns:
        'nombre1|nombre2|nombre3'
    '''
    df1 = df.copy()
    return df1.loc[:, df.columns.str.contains(string, case=False)]


def mes_col(dfserie):
    '''vuelve un dataframe que tenga datos con indices en DateTime en columnas meses y filas años'''
    return pd.pivot_table(dfserie, index=dfserie.index.year, columns=dfserie.index.month)


def correlation_matrix_plot(df):
    '''plot the correlation matrix'''
    corr = df.corr()
    # Draw the heatmap
    g = sns.heatmap(corr, center=0, linewidths=1,
                    annot=True, fmt=".2f", square=True)
    g.set_yticklabels(g.get_yticklabels(), rotation=0,
                      horizontalalignment='right')
    g.set_xticklabels(g.get_xticklabels(), rotation=90,
                      horizontalalignment='right')
    plt.tight_layout()


# How to get the last value of each month
# monthly = df.resample('BM', how=lambda x: x[-1])


def create_date_col(df, year='año', month='mes', day='dia'):
    # df_enviar[[year,month,day]] = df_enviar[[year,month,day]].astype(int)
    df['fecha'] = df.apply(
        lambda row: "{:.0f}-{:02.0f}-{:02.0f}".format(row[year], row[month], row[day]), axis=1)
    df['fecha'] = pd.to_datetime(df['fecha'], format="%Y-%m-%d", errors='coerce')


# * Saving dataframes
def save_df_one_sheet(df_list, sheets, file_name, spaces):
    '''Put multiple dataframes into one xlsx sheet'''
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    row = 0
    for dataframe in df_list:
        dataframe.to_excel(writer, sheet_name=sheets, startrow=row, startcol=0)
        row = row + len(dataframe.index) + spaces + 1
    writer.save()


def save_df_multiple_sheets(df_list, sheet_list, file_name):
    '''Put multiple dataframes across separate tabs/sheets'''
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    for dataframe, sheet in zip(df_list, sheet_list):
        dataframe.to_excel(writer, sheet_name=sheet, startrow=0, startcol=0)
    writer.save()


def remove_accents(input_str)->str:
    
    """[summary]
        df['column'] = df['column'].apply(remove_accents)
    Args:
        a ([type]): [description]

    Returns:
        [type]: [description]
    """
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    text = only_ascii.decode("utf-8")
    return text
