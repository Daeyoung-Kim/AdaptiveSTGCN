import numpy as np
import pandas as pd
import os
import glob
import re

folder = '/Users/user/Desktop/TermProjectCode/01DataProcessing/01_Feature Matrix/@Transit_data'
Months = os.listdir(folder)
Months.reverse()

Raw = []
Errors = []
Errors2 = []

for i in range(len(Months)) :
  try :
    raw = pd.read_csv(folder + '/' + Months[i], encoding='cp949', index_col='사용일자')
    Raw.append(raw)
  except :
    Errors.append(Months[i])

for i in range(len(Errors)) : 
  try :
    pd.read_csv(folder + '/' + Errors[i], encoding='utf-8')
    raw_data = pd.read_csv(folder + '/' + Errors[5], encoding='utf-8').reset_index()
    raw_data.columns = ['사용일자', '노선명', '역명', '승차총승객수', '하차총승객수', '등록일자', 'NaN']
    raw_data = raw_data.drop(columns = [ 'NaN'])
    raw_data = raw_data.set_index('사용일자')
    Raw.append(raw_data)
  except : 
    Errors.append(Errors[i])
        
def make_timeseries_dataset(Raw, data_contents) :
  """ST-GNN 인풋을 위한 데이터셋을 생성하는 프로세스로, 인풋으로는 합치고자 하는 달(Months)과, 생성하고자하는 데이터 종류(type)으로 나뉜다. 이때 type는 : 전체승객수, 승차총승객수, 하차총승객수 """

  months_df = []

  # 월별 데이터 합치기
  for i in range(len(Raw)) :
    month = Raw[i].reset_index()
    month = month.drop(columns=['등록일자'])
    month['전체승객수'] = month['승차총승객수'] + month['하차총승객수'] # 전체승객수 데이터셋

    Day = month['사용일자'].unique() # 일자별 데이터
    month = month.set_index('사용일자')
    days_df = []

    # 일별 데이터 합치기
    for j in range(len(Day)) : 
      day = month.loc[Day[j]]
      day = day.set_index(keys=['역명', '노선명'])
      day = day[[data_contents]]
      day = day.rename(columns = {data_contents : str(Day[j])})
      day = day.sort_index()
      if day.index.is_unique == False : # 중복되는 인덱스 제거
        day = day.groupby(level=[0, 1]).max()
      days_df.append(day)

    output_days = pd.concat(days_df, axis=1, join='outer') # 일자별 취합
    months_df.append(output_days)

  output_months = pd.concat(months_df, axis=1, join='outer') # 월별 취합

  # 같은 '역명'이지만, 이름이 바뀐 역 데이터 취합
  
  return output_months

def duplicate_data_processing(data) :
  """생성된 데이터는 역 이름이 바뀜에 따라 중복되는 데이터들 존재(ex: 숭실대입구, 숭실대입구(살피재)), 이 데이터들을 '역명'을 기준으로 합치는 함수"""

  # (1) 중복되는 역 이름 추출
  data_reset = data.reset_index()
  station_names = data_reset['역명'].unique()
  tofilter2 = [x for x in station_names if "(" and ")" in x] # 괄호 있는 역
  tofilter1 = [re.sub(r'\([^)]*\)', "", x) for x in tofilter2] # 괄호 없는 역
  removal = [14, 22, 29, 34, 51, 52, 53, 59] # tofilter 중에 괄호만 있고 중복되지는 않는 역 인덱스
  tosubtract1 = [tofilter1[x] for x in removal]
  tosubtract2 = [tofilter2[x] for x in removal]
  filter1 = sorted(list(set(tofilter1) - set(tosubtract1)))
  filter2 = sorted(list(set(tofilter2) - set(tosubtract2)))

  # (2) 중복 데이터 취합하기
  update = []

  for i in range(len(filter1)) :
    try :
      filtered_data = data.loc[[filter1[i], filter2[i]]].reset_index()
      filtered_line = filtered_data['노선명'].unique() # 해당역에 포함된 노선 개수

      for k in range(len(filtered_line)) :
        objective_data = filtered_data[filtered_data['노선명']==filtered_line[k]]
        concat_data = pd.DataFrame(objective_data.sum(), columns=['0']).transpose()
        concat_data['역명'] = filter1[i]
        concat_data['노선명'] = filtered_line[k]
        concat_data = concat_data.set_index(keys=['역명', '노선명'])
        update.append(concat_data)

    except :
      print('error : {}'.format(i))

  toreplace = pd.concat(update, axis=0, join='outer').sort_index()

  # (3) 원본 데이터프레임 'data'로부터 대상 목록 제거하기
  filter = sorted(list(set(filter1) | set(filter2)))
  data_filtered = data.drop(filter)

  # (4) 3번으로부터 제거된 데이터프레임 'data_filtered'와 새로 끼워넣을 'toreplace' 합치기
  final = pd.concat([data_filtered, toreplace], axis=0, join='outer')
  final = final.groupby(['역명']).sum()

  return final

data = make_timeseries_dataset(Raw, '전체승객수')

# (1) 중복되는 역 이름 추출
data_reset = data.reset_index()
station_names = data_reset['역명'].unique()
tofilter2 = [x for x in station_names if "(" and ")" in x] # 괄호 있는 역
tofilter1 = [re.sub(r'\([^)]*\)', "", x) for x in tofilter2] # 괄호 없는 역
removal = [14, 22, 29, 34, 51, 52, 53, 59] # tofilter 중에 괄호만 있고 중복되지는 않는 역 인덱스
tosubtract1 = [tofilter1[x] for x in removal]
tosubtract2 = [tofilter2[x] for x in removal]
filter1 = sorted(list(set(tofilter1) - set(tosubtract1)))
filter2 = sorted(list(set(tofilter2) - set(tosubtract2)))

# (2) 중복 데이터 취합하기
update = []

for i in range(len(filter1)) :
  try :
    filtered_data = data.loc[[filter1[i], filter2[i]]].reset_index()
    filtered_line = filtered_data['노선명'].unique() # 해당역에 포함된 노선 개수

    for k in range(len(filtered_line)) :
      objective_data = filtered_data[filtered_data['노선명']==filtered_line[k]]
      concat_data = pd.DataFrame(objective_data.sum(), columns=['0']).transpose()
      concat_data['역명'] = filter1[i]
      concat_data['노선명'] = filtered_line[k]
      concat_data = concat_data.set_index(keys=['역명', '노선명'])
      update.append(concat_data)

  except :
    print('error : {}'.format(i))

toreplace = pd.concat(update, axis=0, join='outer').sort_index()

filter = sorted(list(set(filter1) | set(filter2)))
data_filtered = data.drop(filter)

final = pd.concat([data_filtered, toreplace], axis=0, join='outer')

# final.to_csv('./01 Data Processing/01_Feature Matrix/221103_DataProcessing_Station_Line/Station.Line2Time_Total.csv')
final.to_csv('./01 Data Processing/01_Feature Matrix/ABCD.csv')