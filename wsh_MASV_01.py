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

# **********************************************************
# ============= МОДЕЛИРОВАНИЕ ==============================
# **********************************************************
import statsmodels.api as sm
# Читаем и преобразуем данные
pth_a = './data/AUTO21053A.xlsx'
CARSA = pd.read_excel(pth_a)
CARSA = CARSA.astype({'age':np.float64, 'music':'category', 
                      'signal':'category', 'price':np.float64})
CA = CARSA.copy()
# Разбиение данных на тренировочное и тестовое множество
# frac- доля данных в тренировочном множестве
# random_state - для повторного отбора тех же элементов
CA_train = CA.sample(frac=0.8, random_state=42) 
# Символ ~ обозначает отрицание (not)
CA_test = CA.loc[~CA.index.isin(CA_train.index)] 

# Будем накапливать данные о качестве постреонных моделей
# Используем  adjR^2 и AIC
mq = pd.DataFrame([], columns=['adjR^2', 'AIC']) # Данные о качестве

"""
Построение базовой модели
Базовая модель - линейная регрессия, которая включает в себя 
все количественные переменные и фиктивные переменные дял качественных 
переменных с учетом коллинеарности. Для каждого качетсвенного показателя
включаются все уровни за исключением одного - базового. 
"""
# Формируем целевую переменную
Y = CA_train['price']
# Формируем фиктивные (dummy) переменные для всех качественных переменных
DUM = pd.get_dummies(CA_train[['music', 'signal']])
# Выбираем переменные для уровней, которые войдут в модель
# Будет исключен один - базовый. ВЛияние включенных уровней на зависимую 
# переменную отсчитывается от него
DUM = DUM[['music_есть', 'signal_есть']]
# Формируем pandas.DataFramee содержащий матрицу X объясняющих переменных 
# Добавляем слева фиктивные переменные
X = pd.concat([DUM, CA_train[['age', 'mlg']]], axis=1)
# Добавляем переменную равную единице для учета константы
X = sm.add_constant(X)
X = X.astype({'const':'uint8'}) # Сокращаем место для хранения константы
# Формируем объект, содержащй все исходные данные и методы для оценивания
linreg00 = sm.OLS(Y,X)
# Оцениваем модель
fitmod00 = linreg00.fit()
# Сохраняем результаты оценки в файл
with open('./output/CARS_STAT.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели ******',
          file=fln)
    print(fitmod00.summary(), file=fln)

# Проверяем степень мультиколлинеарности только базовой модели
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame() # Для хранения 
X_q = X.select_dtypes(include='float64')# Только количественные регрессоры
vif["vars"] = X_q.columns
vif["VIF"] = [variance_inflation_factor(X_q.values, i) 
              for i in range(X_q.shape[1])]
# Сохраняем полученные результаты
with pd.ExcelWriter('./output/CARS_STAT.xlsx', engine="openpyxl", 
                    if_sheet_exists='overlay', mode='a') as wrt:
    vif.to_excel(wrt, sheet_name='vif')

# Проверяем гетероскедастичность базовой модели
# помощью коритерия White(а) и F критерия
from statsmodels.stats.diagnostic import het_white
e = fitmod00.resid
WHT = pd.DataFrame(het_white(e, X), index= ['LM', 'LM_P', 'F', 'F_P'])
# Сохраняем полученные результаты
with pd.ExcelWriter('./output/CARS_STAT.xlsx', engine="openpyxl", 
                    if_sheet_exists='overlay', mode='a') as wrt:
    WHT.to_excel(wrt, sheet_name='het')

# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod00.rsquared_adj, fitmod00.aic], 
                 index=['adjR^2', 'AIC'], columns=['base_00']).T
mq = pd.concat([mq, q])    

"""
Исключаем из базовой модели незначимые переменные
Строго пошагово. После исключения каждой пересчитываем модель
Мультиколлинеарность и гетероскедастичность проверяем для итога
Учитываем необходимость проверки гипотез.

Здесь на 10% уровне незначима 'signal_есть'
"""
X_1 = X.drop('signal_есть', axis=1)
# Формируем объект, содержащий все исходные данные и методы для оценивания
linreg01 = sm.OLS(Y,X_1)
# Оцениваем модель
fitmod01 = linreg01.fit()
# Сохраняем результаты оценки в файл
with open('./output/CARS_STAT.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели ******',
          file=fln)
    print(fitmod01.summary(), file=fln)
    
# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod01.rsquared_adj, fitmod01.aic], 
                 index=['adjR^2', 'AIC'], columns=['base_01']).T
mq = pd.concat([mq, q])    

# Обратите внимание - модель стала хуже. Лучше вернуться к предыдущей.

# ****************** Примеры проверок гипотез ******************

"""
Сила влияния пробега на цену зависит от возраста.
Чем старше машина тем влиянее меньше.
Для более старых машин на пробег меньше обращают внимание
Модель для проверки:
price = a0 + a1*'signal_есть'+ a2*'music_есть' + 
+ (a30 + a31*age)*mlg + a4*age + v 
раскрывая скобки
price = a0 + a1*'signal_есть'+ a2*'music_есть' + 
+ a30*mlg + a31*age*mlg + a4*age + v
*****************
Если гипотеза справедлива, то a30<0, a31>0 и значим 
*****************

Целевая переменная не меняется.

"""
# Вводим переменную взаимодействия
X_1 = X.copy()
X_1['ma'] = X_1['mlg']*X_1['age']
linreg02 = sm.OLS(Y,X_1)
fitmod02 = linreg02.fit()
# Сохраняем результаты оценки в файл
with open('./output/CARS_STAT.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели ******',
          file=fln)
    print(fitmod02.summary(), file=fln)
    
# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod02.rsquared_adj, fitmod02.aic], 
                 index=['adjR^2', 'AIC'], columns=['hyp_01']).T
mq = pd.concat([mq, q])    

# Гипотеза не отвергается. Модель стала заметно лучше.

"""
Сила влияния пробега на цену зависит от наличия музыкальной системы.
При наличии музыкальной системы сила влияния меньше.
Наличие музыкальной системы указывают дл япривлечения внимания к более старым
машинам 
Модель для проверки:
price = a0 + a1*'signal_есть'+ a2*'music_есть' + 
+ (a30 + a31*music_есть)*mlg + a4*age + v 
раскрывая скобки
price = a0 + a1*'signal_есть'+ a2*'music_есть' + 
+ a30*mlg + a31*music_есть*mlg + a4*age + v
*****************
Если гипотеза справедлива, то a30<0, a31>0 и значим 
*****************

Целевая переменная не меняется.

"""
# Вводим переменную взаимодействия
X_2 = X.copy()
X_2['mm'] = X_2['mlg']*X_2['music_есть']
linreg03 = sm.OLS(Y,X_2)
fitmod03 = linreg03.fit()
# Сохраняем результаты оценки в файл
with open('./output/CARS_STAT.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели ******',
          file=fln)
    print(fitmod03.summary(), file=fln)
    
# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod03.rsquared_adj, fitmod03.aic], 
                 index=['adjR^2', 'AIC'], columns=['hyp_02']).T
mq = pd.concat([mq, q])   

# Коэффициент при переменной взаимодействия не значим. Гипотеза отвергается.

"""
Сила влияния пробега на цену зависит от пробега.
Скорость падения цены с ростом пробега до определенного значения больше, 
чем после него. 
Модель для проверки:
Вводим переменную
mlg_thr = 1, если mlg >= thr и 0, если нет
thr - неизвестный порог
!!! Порог надо подбирать экспериментально, увеличивая adjR^2 !!!
price = a0 + a1*'signal_есть'+ a2*'music_есть' + 
+ (a30 + a31*mlg_thr)*mlg + a4*age + v 
раскрывая скобки
price = a0 + a1*'signal_есть'+ a2*'music_есть' + 
+ a30*mlg + a31*mlg_thr*mlg + a4*age + v
*****************
Если гипотеза справедлива, то a30<0, a31>0 и значим 
*****************

Целевая переменная не меняется.

"""
thr = 40 # Порог пробега - вариант
X_3 = X.copy()
# Формируем dummy из качественных переменных
mlg_thr = X_3['mlg'] >= thr
X_3['mth'] = X_3['mlg']*mlg_thr # Взаимодействие
linreg04 = sm.OLS(Y,X_3)
fitmod04 = linreg04.fit()
# Сохраняем результаты оценки в файл
with open('./output/CARS_STAT.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели ******',
          file=fln)
    print(fitmod04.summary(), file=fln)
    
# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod04.rsquared_adj, fitmod04.aic], 
                 index=['adjR^2', 'AIC'], columns=['hyp_03']).T
mq = pd.concat([mq, q])   

# Коэффициент при переменной взаимодействия не значим. Надо подбирать порог

# Предсказательная сила
Y_test = CA_test['price']
DUM = pd.get_dummies(CA_test[['music', 'signal']])
# Выбираем переменные для уровней, которые войдут в модель
# Будет исключен один - базовый. ВЛияние включенных уровней на зависимую 
# переменную отсчитывается от него
DUM = DUM[['music_есть', 'signal_есть']]
# Формируем pandas.DataFramee содержащий матрицу X объясняющих переменных 
# Добавляем слева фиктивные переменные
X_test = pd.concat([DUM, CA_test[['age', 'mlg']]], axis=1)
# Добавляем переменную равную единице для учета константы
X_test = sm.add_constant(X_test)
X_test = X_test.astype({'const':'uint8'})
# Генерация предсказаний на тестовом множестве 
pred_ols = fitmod00.get_prediction(X_test)
# Генерация доверительных интервалов с доверительной вероятностью alpha
frm = pred_ols.summary_frame(alpha=0.05)
iv_l = frm["obs_ci_lower"] # Нижняя граница доверительных интервалов
iv_u = frm["obs_ci_upper"] # Верхняя граница доверительных интервалов
fv = frm['mean'] # Предсказанное значение целевой переменной
# Построение графиков
name = 'mlg' # Имя переменной относительно которой строим прогноз
Z = X_test.loc[:, name]
dfn = pd.DataFrame([Z, Y_test, fv, iv_u, iv_l]).T
dfn = dfn.sort_values(by=name)
fig, ax = plt.subplots(figsize=(8, 6))
for z in dfn.columns[1:]:
    dfn.plot(x=dfn.columns[0], y=z, ax=ax)
ax.legend(loc="best")
plt.show()

# Подсчет среднеквадратической ошибки
dif = np.sqrt((dfn.iloc[:,1] - dfn.iloc[:,2]).pow(2).sum()/dfn.shape[0])

# Доля выходов за границы доверительых интервалов
# Сортируем, чтобы индексы во всех рядах совпадали
mn = dfn.iloc[:,1].sort_index() 
out = ((mn > iv_u) + (mn < iv_l)).sum()/dfn.shape[0]
