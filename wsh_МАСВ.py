# -*- coding: utf-8 -*-
"""
Анализ стохастических взаимосвязей
Числовой анализ  

@author: Поляков К. Л.
"""
import os
os.chdir("d:/work.p/")
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Чтение и подготовка данных
pth_b = './data/AUTO21053B.xlsx'
CARSB = pd.read_excel(pth_b)
print(CARSB.head())
print(CARSB.tail())

pth_b_w = './data/AUTO21053B.csv'
CARSB_W = pd.read_table(pth_b_w, sep=';', header=0, decimal=',', encoding='utf-8')

CB = CARSB.copy()
print(CB.dtypes)
CB['music'].replace({1:"есть", 0:"нет"}, inplace=True)
print(CB.dtypes)
CB['music'].head()
# CARSB.head()
CB['music']=CB['music'].astype("category")
CB.loc[0, 'music'] = 'yes'
print(CB.dtypes)
CB['music'].head()


from pandas.api.types import CategoricalDtype
cat_type = CategoricalDtype(categories=["нет", "есть"], ordered=True)
CB['music'] = CB['music'].astype(cat_type)
CB['music'].head()

#*************** Описательная статистика *****************

pth_a = './data/AUTO21053A.xlsx' # Путь относительно рабочего каталога
CARSA = pd.read_excel(pth_a)

CARSA = CARSA.astype({'age':np.float64, 'music':'category', 
                      'signal':'category', 'price':np.float64})

CA = CARSA.select_dtypes(include='float')
CA_STAT = CA.describe()
W = CA.quantile(q=0.75) - CA.quantile(q=0.25) # Получается pandas.Series
# Создаем pandas.DataFrame из новых статистик
CA_irq = pd.DataFrame([W], index=['IQR'])
# Объединяем CA_STAT и W
CA_STAT = pd.concat([CA_STAT, CA_irq])

# Обнаружение выбросов
irq = CA_irq['price']
wisker_u = (CA_STAT.loc['75%', 'price'] + 1.5*irq).values[0]
wisker_l = (CA_STAT.loc['25%', 'price'] - 1.5*irq).values[0]
sel = (CA['price'] > wisker_u) + (CA['price'] <= wisker_l)
OUT = CARSA.loc[sel, :]


#*************** Анализ взаимосвязи *****************

from scipy.stats import pearsonr
from scipy.stats import spearmanr

# Здесь будут значения оценок коэффициента корреляции Пирсона
C_P = pd.DataFrame([], index=CA.columns, columns=CA.columns) 
# Здесь будут значения значимости оценок коэффициента корреляции Пирсона
P_P = pd.DataFrame([], index=CA.columns, columns=CA.columns)
# Здесь будут значения оценок коэффициента корреляции Спирмена
C_S = pd.DataFrame([], index=CA.columns, columns=CA.columns)
# Здесь будут значения значимости оценок коэффициента корреляции Спирмена
P_S = pd.DataFrame([], index=CA.columns, columns=CA.columns)
for x in CA.columns:
    for y in CA.columns:
        C_P.loc[x,y], P_P.loc[x,y] = pearsonr(CA[x], CA[y])
        C_S.loc[x,y], P_S.loc[x,y] = spearmanr(CA[x], CA[y])
        
# Сохраняем текстовый отчет на разные листы Excel файла
with pd.ExcelWriter('./output/CARS_STAT.xlsx', engine="openpyxl") as wrt:
# Общая статистика
    CA_STAT.to_excel(wrt, sheet_name='stat')
# Корреляция Пирсона
    C_P.to_excel(wrt, sheet_name='Pearson')
    dr = C_P.shape[0] + 2
    P_P.to_excel(wrt, startrow=dr, sheet_name='Pearson') # Значимость
# Корреляция Спирмена
    C_S.to_excel(wrt, sheet_name='Spirmen')
    dr = C_S.shape[0] + 2
    P_S.to_excel(wrt, startrow=dr, sheet_name='Spirmen') # Значимость
    

# Анализ корреляции между количественной целевой переменной
# и качественной объясняющей
# Используем библиотеку scipy
# Критерий Крускала-Уоллиса
from scipy.stats import kruskal
# Качественная переменная - 'signal'
# Создаем подвыборки
sel_yes = CARSA['signal']=='есть'
x_1 = CARSA.loc[sel_yes, 'price']
sel_no = CARSA['signal']=='нет'
x_2 = CARSA.loc[sel_no, 'price']
# Используем криетрий Крускала-Уоллиса
Price_sig = kruskal(x_1, x_2)
# Сохраняем текстовый отчет
with open('./output/CARS_STAT.txt', 'w') as fln:
    print('Критерий Крускала-Уоллиса для переменных \'price\' и \'signal\'',
          file=fln)
    print(Price_sig, file=fln)
    
    
    
# Анализ взаимосвязи между двумя качественными показателями
import statsmodels.api as sm
# Читаем и преобразуем данные
pth_a = './data/AUTO21053A.xlsx'
CARSA = pd.read_excel(pth_a)
CARSA = CARSA.astype({'age':np.float64, 'music':'category', 
                      'signal':'category', 'price':np.float64})
CA = CARSA.copy()
# Строим таблицу сопряженности. С маргинальными частотами!!! 
crtx = pd.crosstab(CA['music'], CA['signal'], margins=True)
# Даем имена переменным
crtx.columns.name = 'signal'
crtx.index.name = 'music\signal'
# Из уже готовой таблицы сопряженности
# Создаем объект sm.stats.Table для проведения анализа
# В объекте находятся все необходимые статистики и дополнительные методы
tabx = sm.stats.Table(crtx)
# Альтернативный вариант создания sm.stats.Table
#table = sm.stats.Table.from_data(CA[['music', 'signal']])
# Сохраняем полученные результаты
with pd.ExcelWriter('./output/CARS_STAT.xlsx', engine="openpyxl", 
                    if_sheet_exists='overlay', mode='a') as wrt:
# Таблица сопряженности
    tabx.table_orig.to_excel(wrt, sheet_name='music-signal') 
    dr = tabx.table_orig.shape[0] + 2 # Смещение по строкам
# Ожидаемые частоты при независимости
    tabx.fittedvalues.to_excel(wrt, sheet_name='music-signal', startrow=dr)
# Критерий хи квадрат для номинальных переменных
resx = tabx.test_nominal_association()
# Сохраняем результат в файле 
with open('./output/CARS_STAT.txt', 'a') as fln:
    print('Критерий HI^2 для переменных \'music\' и \'signal\'',
          file=fln)
    print(resx, file=fln)
# # Дополнительно. Только для порядковых переменных
# # Cochran-Armitage trend test
# rslt = tabx.test_ordinal_association() 
# with open('./output/CARS_COAR.txt', 'w') as fln:
#     print(rslt, file=fln)

# Рассчет Cramer V по формуле
nr = tabx.table_orig.shape[0]
nc = tabx.table_orig.shape[1]
N = tabx.table_orig.iloc[nr-1, nc-1]
hisq = resx.statistic
CrV = np.sqrt(hisq/(N*min((nr - 1, nc - 1))))
with open('./output/CARS_STAT.txt', 'a') as fln:
    print('Статистика Cramer V для переменных \'music\' и \'signal\'',
          file=fln)
    print(CrV, file=fln)


# ***************** Графический анализ ******************
# Качественные переменные. Проверяем сбалансированность
# Отбираем качественные признаки
dfn = CARSA.select_dtypes(include=["category"]) 
# Красиво размещаем рисунки на листе "в столбик"
plt.figure(figsize=(15, 9)) # Создаем лист нужного размера в дюймах
# Добавляем пространство между рисунками, чтобы не перекрывались
plt.subplots_adjust(wspace=0.5, hspace=0.5)
nplt = 0 # Номер рисунка
# Количество рисунков равно количеству качественных признаков
nrow = dfn.shape[1] 
# Размещаем рисунки 
for s in dfn.columns:
    nplt += 1
    ax = plt.subplot(nrow, 1, nplt) # Алрес (положение) очередного рисунка
# Подсчет количества представителей каждой категории
    ftb = pd.crosstab(dfn[s], s) 
    ftb.index.name = 'Категории'
# Построение столбчатой диаграммы на выделенном месте
    # ftb.columns.name = s    
    # ftb.plot.bar(ax=ax, grid=True, legend=False, title=s, 
    #              color="green")
    ftb.columns.name = None # Без этого будет ненужная подпись
    ftb.T.plot.bar(ax=ax, grid=True, legend=True, title=s, rot=0, table=False,
              color={"есть":"green", "нет":"yellow"})
plt.savefig('./graphics/cars_nom.pdf', format='pdf') # Сохранение графика
plt.show() # Показ листа с графиками

# Аналогичное построение круговых диаграмм
# Для анализа долей категорий
plt.figure(figsize=(15, 9)) # Дюймы
plt.subplots_adjust(wspace=0.5, hspace=0.5)
nplt = 0
nrow = dfn.shape[1]
tit = {'music': "Музыкальная система", "signal":"Сигнализация"}
for s in dfn.columns:
    nplt += 1
    ax = plt.subplot(nrow, 1, nplt)
    ftb = pd.crosstab(dfn[s], s)
    ftb.index.name = 'Категории'
#    ftb.columns.name = s 
    ftb.plot.pie(subplots=True, table=True, ax=ax, grid=True, legend=False, 
                 colors=['green', 'yellow'])
    ax.set_title(tit[s], fontdict={'fontsize':15, 'color':'blue'}, loc='left')
plt.savefig('./graphics/cars_nom.pdf', format='pdf')
plt.show()

# Качественные переменные. Графический анализ связи
CA = CARSA.copy()
CA['music'] = CA['music'].astype('category')
CA['signal'] = CA['signal'].astype('category')
print(CA.dtypes)
# Возврат к прежнему типу переменной
# CA['music'] = CA['music'].astype('str')
# Строим таблицу сопряженности. С маргинальными частотами!!! 
crtx = pd.crosstab(CA['music'], CA['signal'], margins=True)
# Даем имена переменным
crtx.columns.name = 'signal'
crtx.index.name = 'music'
plt.figure(figsize=(15, 9)) # Создаем лист нужного размера в дюймах
# По горизонтальной оси music
ax=plt.subplot(2, 1, 1)
crtx.iloc[:2, :2].plot.bar(ax=ax) # Для music
# По горизонтальной оси signal
ax=plt.subplot(2, 1, 2)
crtx.iloc[:2, :2].T.plot.bar(ax=ax) # Для signal
# Добавляем пространство между рисунками, чтобы не перекрывались
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

"""
Неравномерная компановка
Визуализация включает в себя:
    1. График анализа взаимосвязи (снизу)
    2. Столбчатые диаграммы для обеих переменных (сверху)
"""
CA = CARSA.copy()
dfn = CA.select_dtypes(include=['O', "category"])
plt.figure(figsize=(15, 9)) # Дюймы
ax=plt.subplot(2, 1, 2)
crtx = pd.crosstab(dfn['music'], dfn['signal'], margins=True)
crtx.iloc[:2, :2].plot.bar(ax=ax, color=['green', 'orange']) # Для xc1
ftb_1 = pd.crosstab(dfn['signal'], 'signal')
ax=plt.subplot(2, 2, 1)
ftb_1.plot.bar(ax=ax, grid=True, legend=False, title='signal',
               color='green')
ftb_2 = pd.crosstab(dfn['music'], 'music')
ax=plt.subplot(2, 2, 2)
ftb_2.plot.bar(ax=ax, grid=True, legend=False, title='music',
               color='orange')
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.savefig('./graphics/cars_combo.pdf', format='pdf')
"""
Общая подпись к графикам
Используем форматированные строки
%s - вставить в строку строку
"""
plt.suptitle('Анализ взаимосвязи переменных %s и %s' % ('music','signal'))
plt.show()

# Графический анализ
# Количественные переменные.
print(CARSA.dtypes)
# Приведение нескольких столюцов pandas.DataFrame к нужному классу
CARSA = CARSA.astype({'age':np.float64, 'music':'category', 
                      'signal':'category', 'price':np.float64})
"""
Гистограммы
Стратегии построения:
    Фридман-Диаконис - 'fd'; 
    Скотт - 'scott';
    Стерджес - 'sturges'.
Строим двумя способами. Первый - 'sturges'.
Выбираем в зависимости от свойств данных.
Для ассиметричных - 'fd'. 
"""
dfn = CARSA.select_dtypes(include='float64')
# Не рекомендуется. Простейший вариант. 
dfn.hist(bins='fd', density=True, grid=True, legend=False, color=None)
# Рекомендуется. Размещение на листе (Pandas и Matplotlib).
# Не более, чем по три графика в столбце 
nrow = dfn.shape[1]
fig, ax_lst = plt.subplots(nrow, 1)
fig.figsize=(15, 9) # Дюймы
nplt = -1
for s in dfn.columns:
    nplt += 1
    # ax_lst[nplt].hist(dfn[[s]], bins='fd', density=True, color=None)
    dfn.hist(column=s, ax=ax_lst[nplt], bins='fd', density=True, grid=True, 
             legend=False, color=None)
#    ax_lst[nplt].hist(dfn[s], bins='fd', density=True, color='darkkhaki')
    # ax_lst[nplt].grid(visible=True)
    ax.set_title(tit[s], fontdict={'fontsize':15, 'color':'blue'}, loc='left')
fig.subplots_adjust(wspace=0.5, hspace=1.0)
"""
Общая подпись к графикам
Используем форматированные 'f'-строки
{} - позиция для подстановки значения
"""
fig.suptitle(f'Гистограммы переменных {list(dfn.columns)}')
plt.savefig('./graphics/cars_qnt.pdf', format='pdf')
plt.show()

# Графический анализ
# Анализ связи между количественной и качественной переменной
# В столбце не более трех графиков
dfn = CARSA.copy()
# Отбор имен качественных переменных
cols = dfn.select_dtypes(include='category').columns
# Количество графиков в столбце
nrow = len(cols) # Количество переменных в cols
fig, ax_lst = plt.subplots(nrow, 1)
fig.figsize=(15, 9) # Дюймы
nplt = -1
for s in cols:
    nplt += 1
# Доверительные интервалы строятся методом бутстрепа    
    dfn.boxplot(column='price', by=s, ax=ax_lst[nplt], grid=True, notch=True, 
                bootstrap=50, showmeans=True, color=None)
fig.subplots_adjust(wspace=0.5, hspace=1.0)
# Общая подпись к графикам
fig.suptitle('Категоризированные диагарммы Бокса-Вискера')
plt.savefig('./graphics/cars_qqdep.pdf', format='pdf')
plt.show()

"""
Графический анализ
Анализ связи между количественной целевой переменной и 
количественными объясняющими переменными
Рекомендуется зависимую переменную размещать первой или последней в 
pandas.DataFrame. Это упрощает автоматизацию расчетов.
"""
# Отбираем количественные переменные.
# Зависимая идет последней
dfn = CARSA.select_dtypes(include='float64')
nrow = dfn.shape[1] - 1 # Учитываем, что одна переменная целевая - ось 'Y'
fig, ax_lst = plt.subplots(nrow, 1)
fig.figsize=(15, 9) # Дюймы
nplt = -1
for s in dfn.columns[:-1]: # Последняя переменная - целевая ('Y')
    nplt += 1
    dfn.plot.scatter(s, 'price', ax=ax_lst[nplt])
    ax_lst[nplt].grid(visible=True)
    ax_lst[nplt].set_title(f'Связь цены с {s}')
fig.subplots_adjust(wspace=0.5, hspace=1.0)
"""
Общая подпись к графикам
Используем форматированные 'f'-строки
{} - позиция для подстановки значения
"""
fig.suptitle(f'Связь цены с {list(dfn.columns[:-1])}')
plt.savefig('./graphics/cars_scat.pdf', format='pdf')
plt.show()