from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import numpy as np
import datetime
import collections
import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os
import sys
# [Fix] Create a wrapper for importing imgres
import trend_remover

# [Fix] Register imgres into system modules
sys.modules['trend_remover'] = trend_remover
from trend_remover import TrendRemover

# Словарь соответствия месяцам и праздничным дням
holydays_dict = {
    1: list(range(1, 10)),
    2: [23],
    3: [8],
    4: [],
    5: [1, 2, 3, 9, 10],
    6: [12],
    7: [],
    8: [],
    9: [1],
    10: [],
    11: [4],
    12: [31]
}
# Словарь соответствия месячам и дням после праздников
day_after_holydays_dict = {
    1: list(range(10, 18)),
    2: list(range(24, 29)),
    3: list(range(9, 12)),
    4: [],
    5: [4, 5] + list(range(11, 16)),
    6: [],
    7: [],
    8: [],
    9: [],
    10: [],
    11: list(range(5, 8)),
    12: []
}
# Словарь соответствия месячам и дням до праздников
days_before_holydays_dict = {
    1: [],
    2: [20, 21, 22],
    3: [6, 7],
    4: [27, 28, 29, 30],
    5: [],
    6: [],
    7: [],
    8: [],
    9: [],
    10: [],
    11: [1, 2, 3],
    12: [26, 27, 28, 29, 30]
}
# Словарь с месяцами максимальных ограничений по COVID-19
covid_months ={
   2020: list(range(4,13)),
   2021: list(range(1,5)) + list(range(9,13))
}

# Функция для добавления признака прздничных дней
def holydays_add(date):
    month = date.month
    day = date.day
    result = 0
    if day in holydays_dict[month]:
        result = 1
    return result

# Функция для добавления признака дней после праздников
def day_after_holydays_add(date):
    month = date.month
    day = date.day
    result = 0
    if day in day_after_holydays_dict[month]:
        result = 1
    return result

# Функция для добавления признака дней до праздников
def day_before_holydays_add(date):
    month = date.month
    day = date.day
    result = 0
    if day in days_before_holydays_dict[month]:
        result = 1
    return result

# Функция для добавления признака пандемии
def get_covid(date):
    result = 0
    if date.year in covid_months.keys():
        result = int(date.month in covid_months[date.year])
    return result

# Инициализируем приложение Flask
app = Flask(__name__)

# Функция для отоображения основной страницы
@app.route('/', methods=['GET'])
def index():
  return '''
<!DOCTYPE html>
<html>
    <head>

    </head>
    <body>
        <span>Начальная дата</span>  
        <input type = "date" id = "start"/>
        <span>Конечная дата</span>
        <input type = "date" id = "end"/>
        <span>Периодичность прогноза</span>
        <select id="period">
          <option disabled>Выберите кратность прогноза</option>
          <option selected value="day">Подневный прогноз</option>
          <option value="week">Понедельный прогноз</option>
          <option value="month">Помесячный прогноз</option>
        </select>
        <br/>
        <button text = "Запросить" onclick="Send()">Запросить</button>
        <div id="result_holder"/>
        <script>
           async function postData(url = "", data = {}) {
               const response = await fetch(url, {
                 method: "POST",
                 mode: "cors", 
                 cache: "no-cache",
                 credentials: "same-origin",
                 headers: {
                   "Content-Type": "application/json",
                 },
                 redirect: "follow", 
                 referrerPolicy: "no-referrer", 
                 body: JSON.stringify(data), 
               });
               return response.json(); 
             }
              
            function Send(){
              start = document.getElementById('start').value
              end = document.getElementById('end').value
              period_tag = document.getElementById('period')
              period = period_tag.value
              console.log(start, end, period)
              document.getElementById('result_holder').innerHTML = "<span>Обработка данных...<br/>Пожалуйста подождите.</span>"
              postData("/predict", {"start": start, "end": end, "period": period }).then((data) => {
                   console.log(data);
                   document.getElementById("result_holder").innerHTML = data["prediction"]
                 });
            }
        </script>
    </body>
</html>'''

# Функция для обработки постзапросов и возвращения прогноза модели в виде таблицы и графика
@app.route('/predict', methods=['POST'])
def predict():
    # Обрабатываем запрос, выделяем параметры
    data = request.json
    start = data['start']
    end = data['end']
    period = data['period']
    # Офрмируем период выборки
    start = datetime.datetime.strptime(start, '%Y-%m-%d')
    end = datetime.datetime.strptime(end, '%Y-%m-%d')
    date_range = pd.Series([start, end])
    date_range = pd.to_datetime(date_range)
    request_index = pd.DataFrame(index = pd.DatetimeIndex(date_range)).asfreq(freq = 'D').index
    # Формируем датафрейм с необходимыми дополнительными признаками
    test_df = pd.DataFrame(index = pd.DatetimeIndex([start, end]))
    test_df = test_df.asfreq(freq = 'D')
    test_df['Date'] = pd.to_datetime(test_df.index.to_pydatetime())
    test_df['holyday'] = test_df['Date'].apply(holydays_add)
    test_df['day_after_holydays'] = test_df['Date'].apply(day_after_holydays_add)
    test_df['day_before_holydays'] = test_df['Date'].apply(day_before_holydays_add)
    test_df['COVID'] = test_df['Date'].apply(get_covid)
    test_df['day_of_week'] = test_df.index.day_name()
    test_df = pd.get_dummies(test_df, columns = ['day_of_week'])
    test_df = test_df.drop(['Date'], axis = 1)
    moon_df = pd.read_csv('./model/moon.csv', sep=';', index_col = 'date', parse_dates = ['date'])
    test_df = pd.merge(test_df, moon_df, how='left', left_index=True, right_index=True)
    # Загружаем модель
    print('loading...')
    with open('./model/model.pkl', 'rb') as pkl_file: 
        model = pickle.load(pkl_file)
    print(model)
	# Делаем предсказание
    test_df['Прогноз поступивших'] = model['model'].predict(test_df)
    test_df['Прогноз поступивших'] = model['trend_remover'].revers_transform(
            test_df,
            target = 'Прогноз поступивших'
        )
    test_df['Прогноз поступивших'] = test_df['Прогноз поступивших'].apply(lambda x: x if x >= 0 else 0)
    # Обрабатываем прогноз для передачи в html
    fig, ax = plt.subplots(figsize = (18,12))
    if period == 'month':
        months_names = {
            1: 'Январь',
            2: 'Февраль',
            3: 'Март',
            4: 'Апрель',
            5: 'Май',
            6: 'Июнь',
            7: 'Июль',
            8: 'Август',
            9: 'Сентябрь',
            10: 'Октябрь',
            11: 'Ноябрь',
            12: 'Декабрь',
        }
        test_df = test_df.loc[request_index].resample('M').agg(sum)
        sns.barplot(x = ['{} - {}'.format(months_names[x.month], x.year) for x in test_df.index], y = round(test_df['Прогноз поступивших']), ax = ax)
        ax.set_xlabel("Месяц")
        ax.tick_params(axis='x', rotation=90)
    elif period == 'day':
        test_df = test_df.loc[request_index]
        sns.lineplot(round(test_df['Прогноз поступивших']), ax = ax)
        ax.set_xlabel("Дата")
    elif period == 'week':
        test_df = test_df.loc[request_index].resample('W').agg(sum)
        sns.barplot(x = ['{} - {}'.format(x.week, x.year) for x in test_df.index], y = round(test_df['Прогноз поступивших']), ax = ax)
        ax.set_xlabel("Неделя")
        ax.tick_params(axis='x', rotation=90)
    ax.set_title("График поступивших")
    ax.set_ylabel("Количество поступивших")
    test_df['Прогноз поступивших'] = round(test_df['Прогноз поступивших'])
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    # Переводим получившиеся график и таблицу с данными в html
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    table = pd.DataFrame(test_df["Прогноз поступивших"]).to_html()
    to_send = '<div style = "display: block">\
                    <div style = "overflow-y: auto; width: 400px; height:800px; position:relative; left: 0px;float: left;display: block; ">' + table + '</div>\
                    <div style ="position:relative; right: 0px;max-width: 1500px;display: block;float: left;"><img style ="width: 1300px;" src="data:image/png;base64,'+ encoded + '"/><div>\
                </div>'
    # Формируем ответ и отправляем его
    response = jsonify({'prediction': to_send})
    return  response

# Функция для добалвения заголовков в ответ
@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response

# Запуск веб приложения
if __name__ == '__main__':
    
    app.run(host = '0.0.0.0', port=5000)