import numpy as np
import pandas as pd
import math

# 1. Adjacency Matrix
Adj_1 = pd.read_csv('./Data/Matrix_data/DistMatrix1.csv',encoding='CP949')
Adj_2 = pd.read_csv('./Data/Matrix_data/DistMatrix2.csv',encoding='CP949')
Adj_3 = pd.read_csv('./Data/Matrix_data/DistMatrix3.csv',encoding='CP949')
Adj_4 = pd.read_csv('./Data/Matrix_data/DistMatrix4.csv',encoding='CP949')
Adj_5 = pd.read_csv('./Data/Matrix_data/DistMatrix5.csv',encoding='CP949')
Adj_6 = pd.read_csv('./Data/Matrix_data/DistMatrix6.csv',encoding='CP949')

Adj_1 = Adj_1.rename(columns = {'Unnamed: 0': 'index'}).set_index('index')
Adj_2 = Adj_2.rename(columns = {'Unnamed: 0': 'index'}).set_index('index')
Adj_3 = Adj_3.rename(columns = {'Unnamed: 0': 'index'}).set_index('index')
Adj_4 = Adj_4.rename(columns = {'Unnamed: 0': 'index'}).set_index('index')
Adj_5 = Adj_5.rename(columns = {'Unnamed: 0': 'index'}).set_index('index')
Adj_6 = Adj_6.rename(columns = {'Unnamed: 0': 'index'}).set_index('index')

# 사용할 함수
# 플로이드 워셜 알고리즘 

def Floyd_Warshall(InitialAdjacencyMatrix) :
  """모든 정점 사이의 최단경로를 찾는 탐색 알고리즘"""
  
  n = len(InitialAdjacencyMatrix)
  INF = np.inf
  dist = [[INF] * n for i in range(n)]

  # c최단경로를 담는 배열 초기화
  for i in range(n) :
    for j in range(n) :
      dist[i][j] = InitialAdjacencyMatrix[i][j]

  # 초기화된 배열을 기반으로, 최단경로 업데이트
  for k in range(n) : # 거치는 점
    for i in range(n) : # 시작점
      for j in range(n) : # 끝점
         dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
 
  return dist

def SetInitialAdjacencyMatrix(StationLine_DataFrame):
  """지하철 노선도를 인접행렬로 변환, 1개 노선에 대해서만 수행"""
  n = len(StationLine_DataFrame)
  Station_list = list(StationLine_DataFrame.reset_index()['역명'])
  
  # 초기화된 인접행렬
  Line_Matrix = [[0] * n for i in range(n)]
  AMatrix_DF = pd.DataFrame(Line_Matrix)
  AMatrix_DF.columns = Station_list
  AMatrix_DF['index'] = Station_list
  AMatrix_DF = AMatrix_DF.set_index('index')

  # 초기 인접행렬
  for i in range(n) :
    for j in range(n) :
      if j == i :
        AMatrix_DF.loc[Station_list[i], Station_list[j]] = 0
      elif j == i + 1 :
        AMatrix_DF.loc[Station_list[i], Station_list[j]] = StationLine_DataFrame.loc[Station_list[j], '역간거리(km)']
      elif j == i - 1 :
        AMatrix_DF.loc[Station_list[i], Station_list[j]] = StationLine_DataFrame.loc[Station_list[j+1], '역간거리(km)']
      else :
        AMatrix_DF.loc[Station_list[i], Station_list[j]] = np.inf

  return AMatrix_DF

# 2단계. 초기 인접행렬을 최종 인접행렬 데이터프레임으로 변환

def Inital2Final_AdjacencyMatrix(InitialAdjacencyMatrix) :
  """초기 인접행렬 DF를 넣어 최종 인접행렬 DF로 변환"""

  A = pd.DataFrame(Floyd_Warshall(InitialAdjacencyMatrix.to_numpy())) # 최종 인접행렬로 변환
  
  stn_order_row = list(InitialAdjacencyMatrix.reset_index()['index'])
  stn_order_col = InitialAdjacencyMatrix.columns

  A.columns = stn_order_col
  A['index'] = stn_order_row
  A = A.set_index('index')

  return A

# 초기 인접거리행렬
for i in range(1,len(Adj_1)):
  for j in range(i+1):
    if(Adj_1.iloc[j,i] != np.nan):
      Adj_1.iloc[i,j] = Adj_1.iloc[j,i]

for i in range(1,len(Adj_2)):
  for j in range(i+1):
    if(Adj_2.iloc[j,i] != np.nan):
      Adj_2.iloc[i,j] = Adj_2.iloc[j,i]

for i in range(1,len(Adj_3)):
  for j in range(i+1):
    if(Adj_3.iloc[j,i] != np.nan):
      Adj_3.iloc[i,j] = Adj_3.iloc[j,i]

for i in range(1,len(Adj_4)):
  for j in range(i+1):
    if(Adj_4.iloc[j,i] != np.nan):
      Adj_4.iloc[i,j] = Adj_4.iloc[j,i]

for i in range(1,len(Adj_5)):
  for j in range(i+1):
    if(Adj_5.iloc[j,i] != np.nan):
      Adj_5.iloc[i,j] = Adj_5.iloc[j,i]

for i in range(1,len(Adj_6)):
  for j in range(i+1):
    if(Adj_6.iloc[j,i] != np.nan):
      Adj_6.iloc[i,j] = Adj_6.iloc[j,i]

# 6호선 응암순환선 전처리 (양방향 순환 -> 단방향 순환으로 바꿔준다)
Adj_1.iloc[154,157] = np.nan
Adj_1.iloc[62,154] = np.nan 
Adj_1.iloc[155,62] = np.nan 
Adj_1.iloc[61,155] = np.nan 
Adj_1.iloc[156,61] = np.nan 
Adj_1.iloc[157,156] = np.nan

Adj_2.iloc[154,157] = np.nan 
Adj_2.iloc[62,154] = np.nan
Adj_2.iloc[155,62] = np.nan
Adj_2.iloc[61,155] = np.nan
Adj_2.iloc[156,61] = np.nan
Adj_2.iloc[157,156] = np.nan

Adj_3.iloc[154,157] = np.nan 
Adj_3.iloc[62,154] = np.nan
Adj_3.iloc[155,62] = np.nan
Adj_3.iloc[61,155] = np.nan
Adj_3.iloc[156,61] = np.nan
Adj_3.iloc[157,156] = np.nan

Adj_4.iloc[154,157] = np.nan 
Adj_4.iloc[62,154] = np.nan
Adj_4.iloc[155,62] = np.nan
Adj_4.iloc[61,155] = np.nan
Adj_4.iloc[156,61] = np.nan
Adj_4.iloc[157,156] = np.nan

Adj_5.iloc[154,157] = np.nan 
Adj_5.iloc[62,154] = np.nan
Adj_5.iloc[155,62] = np.nan
Adj_5.iloc[61,155] = np.nan
Adj_5.iloc[156,61] = np.nan
Adj_5.iloc[157,156] = np.nan

Adj_6.iloc[154,157] = np.nan 
Adj_6.iloc[62,154] = np.nan
Adj_6.iloc[155,62] = np.nan
Adj_6.iloc[61,155] = np.nan
Adj_6.iloc[156,61] = np.nan
Adj_6.iloc[157,156] = np.nan

Adj_1 = Adj_1.fillna(np.inf)
Adj_2 = Adj_2.fillna(np.inf)
Adj_3 = Adj_3.fillna(np.inf)
Adj_4 = Adj_4.fillna(np.inf)
Adj_5 = Adj_5.fillna(np.inf)
Adj_6 = Adj_6.fillna(np.inf)

CompAdj_1 = Inital2Final_AdjacencyMatrix(Adj_1)
CompAdj_2 = Inital2Final_AdjacencyMatrix(Adj_2)
CompAdj_3 = Inital2Final_AdjacencyMatrix(Adj_3)
CompAdj_4 = Inital2Final_AdjacencyMatrix(Adj_4)
CompAdj_5 = Inital2Final_AdjacencyMatrix(Adj_5)
CompAdj_6 = Inital2Final_AdjacencyMatrix(Adj_6)

FinalAdj1 = CompAdj_1
FinalAdj2 = CompAdj_2
FinalAdj3 = CompAdj_3
FinalAdj4 = CompAdj_4
FinalAdj5 = CompAdj_5
FinalAdj6 = CompAdj_6

# 인접행렬 얻기 위해 STGCN의 weighted matrix 조건 적용
import math

for i in range(len(FinalAdj1)):
  for j in range(len(FinalAdj1)):
    if(i == j or (math.exp(-(CompAdj_1.iloc[i,j]/10)**2) < 0.5)):
      FinalAdj1.iloc[i,j] = 0
    else:
      FinalAdj1.iloc[i,j] = math.exp(-(CompAdj_1.iloc[i,j]/10)**2)

for i in range(len(FinalAdj2)):
  for j in range(len(FinalAdj2)):
    if(i == j or (math.exp(-(CompAdj_2.iloc[i,j]/10)**2) < 0.5)):
      FinalAdj2.iloc[i,j] = 0
    else:
      FinalAdj2.iloc[i,j] = math.exp(-(CompAdj_2.iloc[i,j]/10)**2) 

for i in range(len(FinalAdj3)):
  for j in range(len(FinalAdj3)):
    if(i == j or (math.exp(-(CompAdj_3.iloc[i,j]/10)**2) < 0.5)):
      FinalAdj3.iloc[i,j] = 0
    else:
      FinalAdj3.iloc[i,j] = math.exp(-(CompAdj_3.iloc[i,j]/10)**2) 

for i in range(len(FinalAdj4)):
  for j in range(len(FinalAdj4)):
    if(i == j or (math.exp(-(CompAdj_4.iloc[i,j]/10)**2) < 0.5)):
      FinalAdj4.iloc[i,j] = 0
    else:
      FinalAdj4.iloc[i,j] = math.exp(-(CompAdj_4.iloc[i,j]/10)**2) 

for i in range(len(FinalAdj5)):
  for j in range(len(FinalAdj5)):
    if(i == j or (math.exp(-(CompAdj_5.iloc[i,j]/10)**2) < 0.5)):
      FinalAdj5.iloc[i,j] = 0
    else:
      FinalAdj5.iloc[i,j] = math.exp(-(CompAdj_5.iloc[i,j]/10)**2) 

for i in range(len(FinalAdj6)):
  for j in range(len(FinalAdj6)):
    if(i == j or (math.exp(-(CompAdj_6.iloc[i,j]/10)**2) < 0.5)):
      FinalAdj6.iloc[i,j] = 0
    else:
      FinalAdj6.iloc[i,j] = math.exp(-(CompAdj_6.iloc[i,j]/10)**2)

FinalAdj1 = FinalAdj1.sort_values('index').transpose().sort_index()
FinalAdj2 = FinalAdj2.sort_values('index').transpose().sort_index()
FinalAdj3 = FinalAdj3.sort_values('index').transpose().sort_index()
FinalAdj4 = FinalAdj4.sort_values('index').transpose().sort_index()
FinalAdj5 = FinalAdj5.sort_values('index').transpose().sort_index()
FinalAdj6 = FinalAdj6.sort_values('index').transpose().sort_index()

FinalAdj1.to_csv('./Data/Matrix_base/AdjMat01.csv')
FinalAdj2.to_csv('./Data/Matrix_base/AdjMat02.csv')
FinalAdj3.to_csv('./Data/Matrix_base/AdjMat03.csv')
FinalAdj4.to_csv('./Data/Matrix_base/AdjMat04.csv')
FinalAdj5.to_csv('./Data/Matrix_base/AdjMat05.csv')
FinalAdj6.to_csv('./Data/Matrix_base/AdjMat06.csv')
            
# 2. Feature Matrix
df = pd.read_csv('./Data/StationTotal.csv').drop_duplicates()
df1_2 = df[df.columns[2:1797]].transpose().sort_index().transpose() # 20150101~20191130까지의 데이터 수집

df2 = pd.concat([df[df.columns[0:2]],df1_2],axis=1).fillna(0).reset_index(drop=True)
df2 = df2.sort_values(by=['역명']).reset_index(drop=True)

# 이수역, 청량리역, 잠실새내역 역명 일치
df2.loc[565,'역명']='이수'

df2.iloc[557,2:] = df2.iloc[557,2:] + df2.iloc[559,2:]
df2.iloc[558,2:] = df2.iloc[558,2:] + df2.iloc[560,2:]
df2 = df2.drop([559,560])
df2.loc[[557, 558],'역명']='청량리'

df2.iloc[503,2:] = df2.iloc[503,2:] + df2.iloc[379,2:]
df2 = df2.drop(379)

# 노선별로 정리
df2 = df2.sort_values(by=['노선명','역명']).reset_index(drop=True)

index_1to9 = []

for i in range(len(df2)):
  index_1to9.append(df2.iloc[i,1] in ['1호선','2호선','3호선','4호선','5호선','6호선','7호선','8호선','9호선','9호선2단계','9호선2~3단계'])

df3_01 = df2.iloc[index_1to9]

index4 = []

for i in range(len(df3_01)):
  if df3_01.iloc[i,0] in ['신내','까치울','부천종합운동장','춘의','신중동','부천시청','상동','삼산체육관',
                       '굴포천','부평구청','산성','남한산성입구','단대오거리','신흥','수진','모란','언주',
                       '삼성중앙','봉은사','삼전','석촌고분','송파나루','한성백제','둔촌오륜','중앙보훈병원']:
    index4.append(i)
    
df3_1 = df3_01.drop(index4,axis=0)

df3_02 = df2[df2['노선명']=='분당선']
df3_2 = df3_02.iloc[[1,2,3,6,7,14,16,18,19,21,25,34]]

df3_03 = df2[(df2['노선명']=='경부선')]
df3_3 = df3_03.iloc[[0,3,7,8,10,16,25,26,28,31]]

df3_04 = df2[(df2['노선명']=='경인선')]
df3_4 = df3_04.iloc[[1,2,14,15]]

df3_05 = df2[(df2['노선명']=='경의선')]
df3_5 = df3_05.iloc[[0,5,10,13,26,28 ]]

df3_06 = df2[(df2['노선명']=='중앙선')]
df3_6 = df3_06.iloc[[5,6,17,20]]

df3_07 = df2[(df2['노선명']=='경원선')]
df3_7 = df3_07.iloc[[1,3,6,7,12,14,15,17,19,20,21,22,23,25,27,28,29]]

DF4 = pd.concat([df3_1,df3_2,df3_3,df3_4,df3_5,df3_6,df3_7],axis=0).groupby('역명').sum()
DF4.columns = pd.to_datetime(DF4.columns, format='%Y-%m-%d')

# Change point 1 |   before: 20150103~20150327, after: 20160102~20160325
FeatureMat1 = pd.concat([DF4[pd.date_range("2015-01-03",periods=84)],DF4[pd.date_range("2016-01-02",periods=84)]],axis=1)

# Change point 2 |   before: 20160206~20160429, after: 20170204~20170428
FeatureMat2 = pd.concat([DF4[pd.date_range("2016-02-06",periods=84)],DF4[pd.date_range("2017-02-04",periods=84)]],axis=1)

# Change point 3 |   before: 20170610~20170901, after: 20180609~20180831
FeatureMat3 = pd.concat([DF4[pd.date_range("2017-06-10",periods=84)],DF4[pd.date_range("2018-06-09",periods=84)]],axis=1)

# Change point 4 |   before: 20171102~20171201, after: 20181101~20181130
FeatureMat4 = pd.concat([DF4[pd.date_range("2017-11-02",periods=30)],DF4[pd.date_range("2018-11-01",periods=30)]],axis=1)

# Change point 5 |   before: 20181001~20181031, after: 20190930~20190930
FeatureMat5 = pd.concat([DF4[pd.date_range("2018-10-01",periods=31)],DF4[pd.date_range("2019-09-30",periods=31)]],axis=1)

# Transform Feature Matrix to Input of Adaptive ST-GCN
FeatureMat1.to_csv('./Data/Matrix_base/FeatureMat01.csv')
FeatureMat2.to_csv('./Data/Matrix_base/FeatureMat02.csv')
FeatureMat3.to_csv('./Data/Matrix_base/FeatureMat03.csv')
FeatureMat4.to_csv('./Data/Matrix_base/FeatureMat04.csv')
FeatureMat5.to_csv('./Data/Matrix_base/FeatureMat05.csv')