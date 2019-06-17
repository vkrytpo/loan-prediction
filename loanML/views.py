from django.http import HttpResponse
from django.shortcuts import redirect
from django.template.response import TemplateResponse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
def predictor():
    train = pd.read_csv("Train.csv")
    train_original=train.copy()
    train['Dependents'].replace('3+', 3,inplace=True)
    train['Loan_Status'].replace('N', 0,inplace=True)
    train['Loan_Status'].replace('Y', 1,inplace=True)
    train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
    train['Married'].fillna(train['Married'].mode()[0], inplace=True)
    train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
    train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
    train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
    train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
    train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
    train['LoanAmount_log'] = np.log(train['LoanAmount'])
    train=train.drop('Loan_ID',axis=1)
    X = train.drop('Loan_Status',1)
    y = train.Loan_Status
    X=pd.get_dummies(X)
    train=pd.get_dummies(train)
    x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model
def train_model(request):
    global g_model
    g_model=predictor()
    return redirect('/')

def ml(request):
    try:
    	x=g_model
    except:
    	return redirect('/train')
    suff_label={
        'Property_Area':['Rural' ,'Urban','Semi Urban'],
        'Dependents':['0','1','2','3+'] ,
        'Education':['Graduate','Not Graduate'],
        'Self_Employed':['No','Yes'],
        'Gender':['Female','Male'],
        'Married':['No','Yes'],
	'Credit_History':['No','Yes']
	}
    suff_list={
        'Property_Area':['Property_Area_Rural' ,'Property_Area_Urban','Property_Area_Semiurban'],
        'Dependents':['Dependents_0','Dependents_1','Dependents_2','Dependents_3'] ,
        'Education':['Education_Graduate','Education_Not_Graduate'],
        'Self_Employed':['Self_Employed_No','Self_Employed_Yes'],
        'Gender':['Gender_Female','Gender_Male'],
        'Married':['Married_Yes','Married_No']}
    input_area =['Gender', 'Married', 'Dependents', 'Education',
        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Property_Area']
    txt_item=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term',]
    ctx={}
    val={}
    res=0
    form={}
    if request.POST:
       for item in input_area:
            val[item]= int(request.POST[item])
       form = val
       res=modify(val,suff_list)
    ctx={'txt_item':txt_item ,'res':res,'suff_label':suff_label,'form':form }
    return TemplateResponse( request ,'file.html',ctx)
def modify(val ,suff_list):
    dummy_val ={
    'ApplicantIncome':0,
    'CoapplicantIncome':0,
    'LoanAmount':0,
    'Loan_Amount_Term':0,
    'Credit_History':0,
    'LoanAmount_log':0,
    'Gender_Female':0,
    'Gender_Male':0,
    'Married_No':0,
    'Married_Yes':0,
    'Dependents_3':0,
    'Dependents_0':0,
    'Dependents_1':0,
    'Dependents_2':0,
    'Education_Graduate':0,
    'Education_Not_Graduate':0,
    'Self_Employed_No':0,
    'Self_Employed_Yes':0,
    'Property_Area_Rural':0,
    'Property_Area_Semiurban':0,
    'Property_Area_Urban':0}
    for item in val.keys():
        if item in suff_list.keys():
            dummy_val[suff_list[item][int(val[item])-1]]=1
        else:
            dummy_val[item]=val[item]
    dummy_val['LoanAmount_log'] = np.log(dummy_val['LoanAmount'])
    return predict_ans(dummy_val)
def predict_ans(dummy_val):
    res_arr=g_model.predict(pd.DataFrame(dummy_val,index=['0']))
    res1 = bool(res_arr)
    print('status : ',res1)
    return res1
