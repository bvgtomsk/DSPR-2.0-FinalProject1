import numpy as np
import datetime
import collections
import sklearn
import pandas as pd

# Класс для декомпозиции временного ряда с возможностью прогнозирования компонентов в будущее
class TrendRemover:
    
    # в конструкторе инициализируем списки компонентов
    def __init__(self):
        self.week_component = dict()
        self.day_component = dict()
        self.components = []
        self.vol_components = []
    
    # Функция для первичного разложения ряда, определения компонентов и преобразования поданного ряда
    def fit_transform(self, data, target='count', n_components = 5):
        # удалим из ряда данные за 2020 и 2021 год, т.к. они содержат выраженные аномалии связаные с пандемией COVID-19
        data_with_date = data[(data.index.year < 2020) | (
            data.index.year > 2021)].copy(deep=True)
        # Выделим признак даты из индекса для удобства дальнешего формирования признаков
        data_with_date['date'] = pd.to_datetime(
            data_with_date.index.to_pydatetime())
        # Создадим признак дня недели
        data_with_date['day_of_week'] = data_with_date.date.dt.day_of_week
        # CСоздадим признак дня года - основной признак по которому будем определять среднее значение для компонентов.
        data_with_date['day'] = data_with_date.date.dt.day_of_year
        
        # Список с днями года
        days = data_with_date['day'].unique()

        # цикл для создания компонентов с сужающимся окном скользящего среднего
        # Т.к. ряд имеет меняющуюся волатильность в зависиомсти от 
        # величины среднего значения за период, то будем использовать мультипликативный подход.
        # Каждый последующий компонент будте иметть окно в два раза уже предыдущего.
        for i in [round(365/(2**x)) for x in range(0, n_components)]:
            # Задаем компонент сколзящего среднего
            comp = data_with_date[target].rolling(i).mean()
            comp_df = pd.DataFrame(columns=['target'], index=comp.index)
            comp_df['target'] = comp
            comp_df = comp_df.dropna()
            comp_df['day'] = comp_df.index.day_of_year
            comp_dict = dict()
            for day in days:
                comp_dict[day] = comp_df[comp_df['day']
                                         == day]['target'].mean()
            self.components.append(comp_dict)
            data_with_date[target] = data_with_date.apply(
                lambda x: x[target] / comp_dict[x['day']], axis=1)
            
            # Задаем компонент волатильности за период, 
            # т.к. период волатильности в пределах месяца, то необходимо, 
            # чтобы окно захватывало месячный и недельный компоненты, 
            # поэтому при числе заданных компонентов меньше 6, 
            # будем уменьшать изначальное окно пропорционально разнице числа компонетнов с 6 
            vol_window = round(i if n_components > 5 else i / 2**(6 - n_components))
            max_month = data_with_date[target].rolling(vol_window).max().dropna()
            min_month = data_with_date[target].rolling(vol_window).min().dropna()
            min_max_df = pd.DataFrame(
                columns=['min, max'], index=max_month.index)
            min_max_df['min'] = min_month
            min_max_df['max'] = max_month
            min_max_df['day'] = min_max_df.index.day_of_year
            vol_component = dict()
            for day in days:
                curr = min_max_df[min_max_df['day'] == day]
                min = curr['min'].mean()
                max = curr['max'].mean()
                vol_component[day] = max - min
            self.vol_components.append(vol_component)
            data_with_date[target] = data_with_date.apply(
                lambda x: x[target] / vol_component[x['day']], axis=1)        
        
        
        #for day in days:
        #    self.day_component[day] = data_with_date[data_with_date['day'] == day][target].median()
        #data_with_date[target] = data_with_date.apply(
        #        lambda x: x[target] / self.day_component[x['day']], axis=1)   
        # Для возможнсти построения прогноза без дополнительных моделей
        # сформируем компонент недльной сезонности
        days_of_week = data_with_date.day_of_week.unique()
        for day in days_of_week:
            self.week_component[day] = data_with_date[data_with_date.day_of_week ==
                                                      day][target].mean()
        data_with_date[target] = data_with_date.apply(
                lambda x: x[target] / self.week_component[x['day_of_week']], axis=1)

        self.mean = data_with_date[target].mean()

        return self.transform(data = data, target = target)

    # Функция обратной трансформации признака
    # Умножает ряд на все компоненты в обратном порядке
    def revers_transform(self, data, target='count'):
        inner_data = pd.DataFrame(data.copy(deep=True))
        inner_data['date'] = pd.to_datetime(inner_data.index.to_pydatetime())
        inner_data['day'] = inner_data.date.dt.day_of_year
        inner_data['day_of_week'] = inner_data.date.dt.day_of_week
        
        inner_data[target] = inner_data.apply(
                lambda x: x[target] * self.week_component[x['day_of_week']], axis=1)
        
        for i in range(len(self.vol_components)-1, -1, -1):
            comp = self.vol_components[i]
            inner_data[target] = inner_data.apply(
                lambda x: x[target] * comp[x['day']], axis=1)
            comp = self.components[i]
            inner_data[target] = inner_data.apply(
                lambda x: x[target] * comp[x['day']], axis=1)

        return inner_data[target]

    # Функция трансформации ряда. Преобразует ряд разделяя его на уже cформированные ранее компоненты
    def transform(self, data, target='count'):
        inner_data = pd.DataFrame(data.copy(deep=True))
        inner_data['date'] = pd.to_datetime(inner_data.index.to_pydatetime())
        inner_data['day'] = inner_data.date.dt.day_of_year
        inner_data['day_of_week'] = inner_data.date.dt.day_of_week
        
        for i in range(len(self.components)-1, -1, -1):
            comp = self.components[i]
            inner_data[target] = inner_data.apply(
                lambda x: x[target] / comp[x['day']], axis=1)
            comp = self.vol_components[i]
            inner_data[target] = inner_data.apply(
                lambda x: x[target] / comp[x['day']], axis=1)
        
        inner_data[target] = inner_data.apply(
                lambda x: x[target] / self.week_component[x['day_of_week']], axis=1)

        return inner_data[target]
    
    # Функция возвращающая прогноз по декомпозированному ряду
    def get_day_component(self, data, target):
        inner_data = pd.DataFrame(data.copy(deep=True))
        inner_data['date'] = pd.to_datetime(inner_data.index.to_pydatetime())
        inner_data['day_of_week'] = inner_data.date.dt.day_of_week

        return inner_data.apply(
            lambda x: self.week_component[x['day_of_week']], axis=1)

    # Функция возвращающая прогноз с восттановленными компонентами, на основе компонента недельной сезонности
    def predict(self, data, target='count'):
        data[target] = self.mean

        return self.revers_transform(data, target = target)
