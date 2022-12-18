# **Using GNN to Predict Passenger Usage After Subway Line Extension**

## **Summary**
Application of Spatio-Temporal GCN to predict passenger usage after subway line extension.    
Source codes are based on PyTorch implementation of ST-GCN  
(https://github.com/hazdzz/STGCN).

## **Dataset Source**
Passenger usages of stations   
(https://data.seoul.go.kr/dataList/OA-12914/S/1/datasetView.do#)  
Distance between stations
(https://www.data.go.kr/tcs/dss/selectDataSetList.do?keyword=역간거리&brm=&svcType=&instt=&recmSe=N&conditionType=init&extsn=&kwrdArray=)  

## **How to Run Code**
Run files in the following order.

`python DataLoader.ipynb`  
`python MatrixLoader.py`  
`python main.py`  
`python Baseline_comparison.py`

## **Requirements**
This code is available at Python >= 3.6, but 3.8.1 is recommended.  
To install required packages:  

`pip install -r requirements.txt`