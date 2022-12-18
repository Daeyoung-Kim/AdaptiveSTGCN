import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model, svm, ensemble
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings(action='ignore')

"""
Adaptive ST-GNN의 성능을 비교하기 위한 baseline performance를 위한 모듈.
같은 머신 러닝모델들(LR, SVM, RF)과의 위해 ST-GNN과 같은 출처의 데이터를 사용.
"""

"""
** PART 1, Adaptive ST-GNN의 인풋데이터 MLR에 적합하게 가공**
- **Y**  = 회귀를 위한 y 값 (Numerical)
- **X1** = 지나는 노선 개수 (Numerical)
- **X2** = 해당되는 지하철 노선 One-hot Vector < 인접행렬의 특성 반영 (Categorical)
- **X3** = 1/주요 업무지구(종로/강남/여의도)와의 거리 < 인접행렬의 특성 반영 (Numerical)
"""

# Y
# Import DataFrame
origin = pd.read_csv("./Data/Matrix_base/FeatureMat05.csv")
origin = origin.set_index(keys=['역명'])

# Split DataFrame for 2018/2019
year_18Oct = origin.iloc[:, :31]
year_19Oct = origin.iloc[:, 31:]

def build_feature_Y(year):
    """기존의 년도별 10월 일별 승하차 데이터를 요일별 평균으로 취합하는 함수."""

    if year == 2018:
        data = year_18Oct
    elif year == 2019:
        data = year_19Oct

    # Day Splitters for days of the week
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    alldays = [i for i in range(31)]
    monday = np.array([7 * i for i in range(5)])
    weekdays = [monday]

    for i in range(1, 7):
        day = sorted(list(set(monday + i) & set(alldays)))
        weekdays.append(day)
    weekdays[0] = weekdays[0].tolist()

    # 기존 데이터프레임을 요일별 평균값으로 변환
    DayAverage_df = []
    for i in range(len(weekdays)):
        day_av = data.iloc[:, weekdays[i]].mean(axis=1).to_frame().rename(columns={0: day_names[i]})
        DayAverage_df.append(day_av)

    result = pd.concat(DayAverage_df, axis=1)
    result = result.reset_index().rename(columns={'역명': 'station'}).set_index('station')

    return result


# X
# Import DataFrame
station2line = pd.read_csv("./Data/Station2Line.csv").set_index(keys=['역명', '노선명'])
adjmat_befor = pd.read_csv("./Data/Matrix_base/AdjMat05.csv").set_index(keys=['Unnamed: 0'])
adjmat_after = pd.read_csv("./Data/Matrix_base/AdjMat06.csv").set_index(keys=['Unnamed: 0'])

def build_feature_staion_data(data):
    """지하철 역명과 노선명을 기반으로 X1, X2 생성"""
    # 지하철 노선명 정제
    problm_stn = ['경원선', '경부선', '장항선', '경인선', '경의선', '중앙선', '공항철도 1호선', '과천선', '안산선', '일산선', '9호선2단계']
    mended_stn = ['1호선', '1호선', '1호선', '1호선', '경의중앙선', '경의중앙선', '공항철도', '4호선', '4호선', '3호선', '9호선']
    fix_dict = dict(zip(problm_stn, mended_stn))

    data = data.reset_index().set_index('노선명').drop('9호선2~3단계', axis=0)  # 중복된 9호선 열 삭제
    data = data.reset_index().set_index(keys=['역명', '노선명'])

    orign_line_list = data.reset_index()['노선명'].tolist()
    fixed_line_list = []
    for i in range(len(orign_line_list)):
        if orign_line_list[i] in problm_stn:
            item = fix_dict[orign_line_list[i]]
            fixed_line_list.append(item)
        elif orign_line_list[i] == '9호선2~3단계':
            fixed_line_list.append(orign_line_list[i])
        else:
            fixed_line_list.append(orign_line_list[i])

    data['노선'] = fixed_line_list
    data = data.reset_index().set_index(keys=['역명', '노선']).drop('노선명', axis='columns', inplace=False)

    # One-hot vector을 위한 지하철노선별 컬럼 추가
    lines_name = sorted(list(set(fixed_line_list)))
    for i in range(len(lines_name)):
        data[lines_name[i]] = 0

    # 모든 역을 돌면서 해당 노선에 대한 one-hot vector 생성
    stations = list(data.reset_index()['역명'])
    for i in range(len(stations)):
        single_stn = list(data.loc[stations[i]].index)
        for j in range(len(single_stn)):
            data.loc[stations[i], single_stn[j]].loc[:, single_stn[j]] = 1
    data['transfer_count'] = list(data.sum(axis=1))  # 추후 환승노선 개수 세기 위한 특성
    data = data.sort_index()

    # 노선별로 분리되어 있는 데이터를 하나의 역 데이터로 합침
    unique_stn = sorted(list(set(list(data.reset_index()['역명']))))
    total_df = []
    for i in range(len(unique_stn)):
        stn_data = pd.DataFrame(data.loc[unique_stn[i]].sum()).T
        stn_data['station'] = unique_stn[i]
        total_df.append(stn_data)

    result = pd.concat(total_df).set_index('station')

    return result

def build_feature_X(data, year):
    """
    인풋으로 들어오는 year에 따라 X1, X2, X3 생성하고,
    Regression을 위한 X 데이터를 아웃풋으로 내놓는다.
    """

    data = build_feature_staion_data(station2line)
    unique_stn = list(data.index)
    alter_stn = ['종합운동장', '석촌', '올림픽공원']
    target_stn = ['종로3가', '강남', '여의도']

    if year == 2018:
        # X1 X2 데이터를 Change Point 이전으로 보정
        data.loc[alter_stn, '9호선'] = 0
        data.loc[alter_stn, 'transfer_count'] = 1

        # X3 생성, 9호선 확장 이전/이후의 네트워크를 기준으로
        target_dist = adjmat_befor[target_stn]

    elif year == 2019:
        # X3 생성, 9호선 확장 이전/이후의 네트워크를 기준으로
        target_dist = adjmat_after[target_stn]

    else:
        print('Error')

    target_dist['station'] = list(target_dist.index)
    target_dist = target_dist.set_index('station')

    # 기존에 작업했던 X1+X2와 X3 결합
    data_X = pd.merge(data.reset_index(), target_dist.reset_index(), how='inner').set_index('station')

    return data_X

# X+Y
def input4regression(year):
    """최종적으로 Regression에 들어갈 데이터 생성"""

    X = build_feature_X(station2line, year)
    Y = build_feature_Y(year)

    Data4Regression = pd.merge(Y, X, left_index=True, right_index=True, how='right').fillna(0)

    return Data4Regression

target_hocs = ['석촌', '올림픽공원', '종합운동장', '잠실', '송파', '둔촌동', '방이', '잠실새내', '삼성', '선정릉']
Data4Regression18 = input4regression(2018)
Data4Regression19 = input4regression(2019)


"""
** PART 2, 베이스라인모델(MLR, SVM, RF)들과 비교
"""

class model:
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    def __init__(self):
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.index = 0
        self.model = 0
        self.acc = 0

    def get_data(self, data, index):
        """요일 코드를 지정해 데이터를 불러온다. Train Test Split은 9:1비율."""
        len_train = int(len(data) * 0.9)
        self.X_train = data.iloc[:, 7:].iloc[:len_train]
        self.X_test = data.iloc[:, 7:].iloc[len_train:]
        self.y_train = data[[model.days[index]]].iloc[:len_train]
        self.y_test = data[[model.days[index]]].iloc[len_train:]
        self.index = index

    def linear_regression(self):
        """선형회귀"""
        # 모델 학습
        line = linear_model.LinearRegression(normalize=True, fit_intercept=True)
        line.fit(self.X_train, self.y_train)
        self.model = line

        # 성능저장
        acc = line.score(self.X_test, self.y_test)
        self.acc = acc

    def support_vector_regression(self):
        """서포트벡터머신 회귀"""
        # 모델학습
        svr = svm.SVR()
        svr.fit(self.X_train, self.y_train)
        self.model = svr

        # 성능저장
        acc = svr.score(self.X_test, self.y_test)
        self.acc = acc

    def random_forest_regression(self):
        """랜덤포레스트 회귀"""
        # 모델학습
        rf = ensemble.RandomForestRegressor()
        rf.fit(self.X_train, self.y_train)
        self.model = rf

        # 성능저장
        acc = rf.score(self.X_test, self.y_test)
        self.acc = acc

    def targetted_prediction(self, test_dataframe):
        """신규 데이터 예측"""

        target = test_dataframe.loc[target_hocs].iloc[:, 7:]

        items = target_hocs
        preds = []
        reals = list(test_dataframe.loc[target_hocs].iloc[:, self.index])

        # 최종 예측
        for i in range(len(items)):
            data = np.array(target.iloc[i]).reshape(1, -1)
            pred = self.model.predict(data).ravel()[0]
            preds.append(pred)

        # 실제값과 예측값을 절대 오차 데이터프레임화
        daytype = model.days[self.index]
        target_dataframe = pd.DataFrame(preds)
        target_dataframe.columns = [daytype]
        target_dataframe[daytype] = abs(target_dataframe[daytype] - reals) * (1 / 10000)

        return target_dataframe

def average_acc(data_before, data_after, regr_option, day_option):
    """각 모델별 예측을 월~금에 대해 반복 수행하고, 주중의 예측데이터와 실제 데이터의 절대오차를 데이터프레임으로 정리"""
    pred_data = []
    accurcy_data = []

    if day_option == 'weekday':
        for i in range(5):
            regr = model()
            regr.get_data(data_before, i)
            if regr_option == 'linear_regression':
                regr.linear_regression()
            elif regr_option == 'support_vector_regression':
                regr.support_vector_regression()
            elif regr_option == 'random_forest_regression':
                regr.random_forest_regression()
            day = regr.targetted_prediction(data_after)
            pred_data.append(day.iloc[:, 0])
            accurcy_data.append(regr.acc)

    elif day_option == 'weekend':
        for i in range(5, 7):
            regr = model()
            regr.get_data(data_before, i)
            if regr_option == 'linear_regression':
                regr.linear_regression()
            elif regr_option == 'support_vector_regression':
                regr.support_vector_regression()
            elif regr_option == 'random_forest_regression':
                regr.random_forest_regression()
            day = regr.targetted_prediction(data_after)
            pred_data.append(day.iloc[:, 0])
            accurcy_data.append(regr.acc)

    # 예측값
    pred_df = pd.concat(pred_data, axis=1)
    pred_df['stations'] = target_hocs
    pred_df = pred_df.set_index('stations')
    pred_df = pd.DataFrame(pred_df.mean(axis=1), columns=[day_option])

    # 평균정확도
    temp = np.array(accurcy_data)
    average_acc = temp.mean()

    return pred_df, average_acc

# (1) Multiple_Linear_Regression 예측 오차
weekday_df, weekday_acc = average_acc(Data4Regression18, Data4Regression19, 'linear_regression', 'weekday')
weekend_df, weekend_acc = average_acc(Data4Regression18, Data4Regression19, 'linear_regression', 'weekend')
MLR_final = pd.concat([weekday_df, weekend_df], axis=1)

# (2) Support_Vector_Regression 예측 오차
weekday_df, weekday_acc = average_acc(Data4Regression18, Data4Regression19, 'support_vector_regression', 'weekday')
weekend_df, weekend_acc = average_acc(Data4Regression18, Data4Regression19, 'support_vector_regression', 'weekend')
SVR_final = pd.concat([weekday_df, weekend_df], axis=1)

# (3) Random_Forest_Regression 예측 오차
weekday_df, weekday_acc = average_acc(Data4Regression18, Data4Regression19, 'random_forest_regression', 'weekday')
weekend_df, weekend_acc = average_acc(Data4Regression18, Data4Regression19, 'random_forest_regression', 'weekend')
RF_final = pd.concat([weekday_df, weekend_df], axis=1)

# Final Result
AbosoluteError = pd.concat([MLR_final, SVR_final, RF_final], axis=1)
AbosoluteError.columns = ['MLR_weekday', 'MLR_weekend', 'SVR_weekday', 'SVR_weekend', 'RF_weekday', 'RF_weekend']
AbosoluteError.to_csv('./Result/Baseline_comparison.csv', encoding='cp949')
print(AbosoluteError)

