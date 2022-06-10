import pickle

import altair
import joblib

import requests
import streamlit as st
from cryptography.hazmat.primitives.hashes import MD5
from keras.preprocessing import image
from streamlit_option_menu import option_menu
import pandas as pd
import cv2
from PIL import Image, ImageEnhance
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import  LogisticRegression  #逻辑回归
from sklearn.tree import DecisionTreeClassifier       #决策树
from sklearn.neighbors import  KNeighborsClassifier   #k近邻
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  #线性判别分析
from sklearn.naive_bayes import GaussianNB            #朴素贝叶斯
from sklearn.svm import SVC                           #支持向量机
from tensorflow.python.keras.models import load_model
from wordcloud import WordCloud
import glob

st.set_option('deprecation.showPyplotGlobalUse', False)
matplotlib.use('Agg')
# st.set_option('deprecation.showPyplotGlobalUse', False)
# emotions_emoji_dict = {"anger": "🤬🤬", "disgust": "🤮", "fear": "😱", "happy": "🤗", "joy": "😂", "neutral": "😐",
#                            "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"}



diabetes_model = pickle.load(
    open(r'C:\Users\zzy\PycharmProjects\MultipleDiseasePre\multipleDisease\diabetes_model.sav', 'rb'))
    # open(r'diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(
    open(r'C:\Users\zzy\PycharmProjects\MultipleDiseasePre\multipleDisease\heart_disease_model.sav', 'rb'))
    # open(r'heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(
    open(r'C:\Users\zzy\PycharmProjects\MultipleDiseasePre\multipleDisease\parkinsons_model.sav', 'rb'))
    # open(r'parkinsons_model.sav', 'rb'))

def convert_df(df):
     return df.to_csv().encode('utf-8')


def plot(kind):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt = df[selected_columns_names].plot(kind=kind)
    st.write(plt)
    st.pyplot()


with st.sidebar:
    selected = option_menu('多疾病预测系统',
                               ['任意表格数据分析','糖尿病预测', '心脏病预测', '帕金森预测','猴痘','脑肿瘤预测'],
                               icons=['border-all','activity', 'heart-fill', 'person-fill',"bricks","exclamation-circle-fill"],
                           )
    st.info('[©Developed by zbossz,here is my gmail:2430871434zzy@gmail.com]')
    st.info('[©本web由zbossz开发,这是我的QQ邮箱:430561907@qq.com]')

if selected =='任意表格数据分析':
    st.title("任意表格数据图像分析")
    activities2 = ["探索性数据分析","图像分析","建立模型"]
    choice3 = st.selectbox("选择操作",activities2)
    if choice3 == "探索性数据分析":
        data = st.file_uploader("上传数据集",type=["csv","txt"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(10))
            col1,col2 = st.columns(2)
            with col1:
                if st.checkbox("展示数据的规格"):
                    st.write(df.shape)
                    label = st.text_input("请输入您的数据的结果列")
                    if label != "":
                        st.write(df.iloc[:,int(label)].value_counts())
            with col2:
                if st.checkbox("展示数据的列名"):
                    st.write(df.columns.to_list())
            if st.checkbox("显示数据概要"):
                st.write(df.describe())

            if st.checkbox("显示该数据有没有重复数据"):
                st.dataframe(df[df.duplicated()])
            if st.checkbox("选择要展示的列的数据"):
                selected_columns = st.multiselect("选择要展示的列",df.columns.to_list())
                new_df = df[selected_columns]
                st.dataframe(new_df)
    elif choice3 == "图像分析":
        data = st.file_uploader("上传数据集", type=["csv", "txt"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(10))
            if st.checkbox("相关性分析"):
                fig, ax = plt.subplots()
                st.write(sns.heatmap(df.corr(),annot=True))
                st.pyplot(fig)
            if st.checkbox("饼状图分析"):
                fig1,ax = plt.subplots()
                all_columns = df.columns.to_list()
                columns_to_plot = st.selectbox("选一列作为饼状图分析数据",all_columns)
                pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct = "%1.1f%%")
                st.write(pie_plot)
                st.pyplot(fig1)
            all_columns_names = df.columns.tolist()
            type_of_plot = st.selectbox("选择绘图的种类:",["区域图","条状图","线型图","直方图","盒装图","kde"])
            selected_columns_names = st.multiselect("选择要画的数据列",all_columns_names)
            if st.button("生成图像"):
                if type_of_plot =="区域图":
                    data = df[selected_columns_names]
                    st.area_chart(data)
                elif type_of_plot =="条状图":
                    data = df[selected_columns_names]
                    st.bar_chart(data)
                elif type_of_plot =="线型图":
                    data = df[selected_columns_names]
                    st.line_chart(data)
                elif type_of_plot == "直方图":
                    plot("hist")
                elif type_of_plot =="盒装图":
                    plot("box")
                elif type_of_plot =="kde":
                    plot("kde")
    elif choice3 == "建立模型":
        data = st.file_uploader("上传数据集", type=["csv", "txt"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(10))
            all_columns = df.columns.to_list()
            label =st.text_input("请输入您得数据集得标签列是多少(数字)，最后一列可用-1表示，以此类推，默认是最后一列")
            if label !="":
                label1 = int(label)
                X = df.iloc[:,0:label1]
                Y = df.iloc[:,label1]
                seed = 7
                models = []
                models.append(("逻辑回归模型",LogisticRegression()))
                models.append(("线性判别分析模型",LinearDiscriminantAnalysis()))
                models.append(("k近邻模型",KNeighborsClassifier()))
                models.append(("决策树模型",DecisionTreeClassifier()))
                models.append(("朴素贝叶斯模型",GaussianNB()))
                models.append(("支持向量机模型",SVC()))

                model_names=[]
                model_mean = []
                model_std =[]
                all_models =[]
                score = 'accuracy'
                for name,model in models:
                    kfold = model_selection.KFold(n_splits=10,random_state=seed,shuffle=True)
                    cv_results = model_selection.cross_val_score(model,X,Y,cv=kfold,scoring=score)
                    model_names.append(name)
                    model_mean.append(cv_results.mean())
                    model_std.append(cv_results.std())
                    accuracy_results = {"模型名称":name,"模型精度":cv_results.mean(),"模型标准差":cv_results.std()}
                    all_models.append(accuracy_results)
                if st.checkbox("结果表格"):
                    st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns=["模型名称              ","模型精度         ","模型标准差        "]))

                if st.checkbox("结果json"):
                    st.json(all_models)
elif (selected == '糖尿病预测'):
    st.dataframe(pd.read_csv(r'C:\Users\zzy\PycharmProjects\MultipleDiseasePre\multipleDisease\diabetes.csv'))
    # st.dataframe(pd.read_csv(r'diabetes.csv'))
    st.title('糖 尿 病 预 测 系 统 基 于 机 器 学 习')
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('怀孕次数')
    with col2:
        Glucose = st.text_input('血糖水平')
    with col3:
        BloodPressure = st.text_input('血压水平')
    with col1:
        SkinThickness = st.text_input('皮脂厚度')
    with col2:
        Insulin = st.text_input('胰岛素水平')
    with col3:
        BMI = st.text_input('体重指数')
    with col1:
        DiabetesPredigreeFunction = st.text_input('糖尿病前期功能')
    with col2:
        Age = st.text_input('年龄')
    diab_dignosis = ''
    st.warning("必须键入所有数据,才能点预测,数据可使用上述表格的数据！")
    if (st.button('预测')):
        prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                                              BMI, DiabetesPredigreeFunction, Age]])
        if (prediction[0] == 1):
            diab_dignosis = '数据人员为糖尿病患者'
        else:
            diab_dignosis = '数据人员非糖尿病患者'
            st.balloons()
    st.success(diab_dignosis)

elif (selected == '心脏病预测'):
    st.dataframe(pd.read_csv(r'C:\Users\zzy\PycharmProjects\MultipleDiseasePre\multipleDisease\heart.csv'))
    # st.dataframe(pd.read_csv(r'heart.csv'))
    st.title('心 脏 病 预 测 系 统 基 于 机 器 学 习')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('年龄')

    with col2:
        sex = st.text_input('性别')

    with col3:
        cp = st.text_input('胸痛类型')

    with col1:
        trestbps = st.text_input('静息血压')

    with col2:
        chol = st.text_input('血清胆固醇（mg/dl）')

    with col3:
        fbs = st.text_input('空腹血糖>120毫克/分升')

    with col1:
        restecg = st.text_input('静息心电图结果')

    with col2:
        thalach = st.text_input('达到最大心率')

    with col3:
        exang = st.text_input('运动性心绞痛')

    with col1:
        oldpeak = st.text_input('运动诱发ST段压低').strip()

    with col2:
        slope = st.text_input('峰值运动ST段斜率')

    with col3:
        ca = st.text_input('透视着色的主要血管')

    with col1:
        thal = st.text_input('thal：0=正常；1=固定缺陷；2=可逆缺陷')

    heart_diagnosis = ''

    st.warning("必须键入所有数据,才能点预测,数据可使用上述表格的数据！")
    # st.error("心脏病预测暂时有问题！！不可用！！")
    if st.button('预测'):
        array1 = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        array2 = np.array(array1, dtype=float)
        heart_prediction = heart_disease_model.predict([array2])
        if (heart_prediction[0] == 1):
            heart_diagnosis = '数据人员患有心脏病'
        else:
            heart_diagnosis = '数据人员没有心脏病'
            st.balloons()

    st.success(heart_diagnosis)

elif (selected == '帕金森预测'):
    st.dataframe(pd.read_csv(r'C:\Users\zzy\PycharmProjects\MultipleDiseasePre\multipleDisease\parkinsons.csv'))
    # st.dataframe(pd.read_csv(r'parkinsons.csv'))
    st.title('帕 金 森 预 测 系 统 基 于 机 器 学 习')
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)平均人声基频')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)最大人声基频')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)最小人声基频')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)基频变化的测量方法')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)基频变化的测量方法')

    with col1:
        RAP = st.text_input('MDVP:RAP基频变化的测量方法')

    with col2:
        PPQ = st.text_input('MDVP:PPQ基频变化的测量方法')

    with col3:
        DDP = st.text_input('Jitter:DDP基频变化的测量方法')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer\n\n')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)幅度变化的测量')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3幅度变化的测量')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5幅度变化的测量')

    with col3:
        APQ = st.text_input('MDVP:APQ幅度变化的测量')

    with col4:
        DDA = st.text_input('Shimmer:DDA幅度变化的测量')

    with col5:
        NHR = st.text_input('NHR语音状态中噪声与音调分量比的测量')

    with col1:
        HNR = st.text_input('HNR语音状态中噪声与音调分量比的测量')

    with col2:
        RPDE = st.text_input('RPDE非线性动态复杂性度量')

    with col3:
        DFA = st.text_input('DFA信号分形缩放指数')

    with col4:
        spread1 = st.text_input('spread1基本频率变化的非线性测量')

    with col5:
        spread2 = st.text_input('spread2基本频率变化的非线性测量')

    with col1:
        D2 = st.text_input('D2非线性动态复杂性度量')

    with col2:
        PPE = st.text_input('PPE基本频率变化的非线性测量')

    parkinsons_diagnosis = ''
    st.warning("必须键入所有数据,才能点预测,数据可使用上述表格的数据！")

    if st.button("预测"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                                                           Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR,
                                                           RPDE,
                                                           DFA, spread1, spread2, D2, PPE]])

        if (parkinsons_prediction[0] == 1):
            parkinsons_diagnosis = "数据人员患有帕金森"
        else:
            parkinsons_diagnosis = "数据人员没有帕金森"
            st.balloons()

    st.success(parkinsons_diagnosis)
elif selected == '猴痘':
    monkey_pox_data = pd.read_csv(r"C:\Users\zzy\PycharmProjects\MultipleDiseasePre\multipleDisease\monkey_pox_data.csv")
    # monkey_pox_data = pd.read_csv("monkey_pox_data.csv")
    st.title("猴痘数据分析")
    st.header("数据每日更新")
    st.dataframe(monkey_pox_data)
    csv = convert_df(monkey_pox_data)
    st.download_button(
        label="下载猴痘数据",
        data=csv,
        file_name='monkey_pox_data.csv',
        mime='text/csv',
    )
    col1,col2 = st.columns([1,2])
    with col1:
        if st.checkbox("查看数据集的形状"):
            st.success(monkey_pox_data.shape)
    with col2:
        if st.checkbox("查看目前数据有猴痘病患的国家"):
            st.write(monkey_pox_data['Country'].unique())
    with col1:
        if st.checkbox("查看有多少个国家有猴痘病患"):
            st.success(len(monkey_pox_data['Country'].unique()))
    with col2:
        if st.checkbox("查看各个国家对应的病患数量"):
            st.write(monkey_pox_data['Country'].value_counts())

    choice = st.selectbox("请选择需查看的图形化界面:",['请选择','查看各个国家对应的病患数量','所有猴痘病例状况的统计','查看前N个最大病例数国家的病例状况','查看症状词云图','查看每日疾病数据'])
    if choice == '查看各个国家对应的病患数量':
        plt.figure(figsize=(40,30))
        fig3, ax = plt.subplots()
        plt.xticks(rotation=90)
        st.write(sns.countplot(x='Country',data=monkey_pox_data))
        st.pyplot(fig3)
    elif choice == '所有猴痘病例状况的统计':
        plt.figure(figsize=(40, 30))
        fig3, ax = plt.subplots()
        st.write(sns.countplot(x='Status', data=monkey_pox_data))
        st.pyplot(fig3)
    elif choice == '查看前N个最大病例数国家的病例状况':
        plt.figure(figsize=(40, 30))
        fig, ax = plt.subplots()
        number = st.number_input("请输入您想查看的国家数目",min_value=1,max_value=len(monkey_pox_data['Country'].unique()),step=1)
        top_countries = list(monkey_pox_data['Country'].value_counts().to_frame().nlargest(number,'Country').index)
        top_countries_df = monkey_pox_data[monkey_pox_data['Country'].isin(top_countries)]
        st.dataframe(top_countries_df)
        st.write(sns.catplot(x='Country',kind="count" ,data=top_countries_df,hue="Status"))
        st.pyplot()
    elif choice == '查看症状词云图':
        fig1, ax1 = plt.subplots()
        text = ' '.join(monkey_pox_data['Symptoms'].fillna('0').to_list())
        myworcloud = WordCloud().generate(text)
        ax1 = plt.imshow(myworcloud,interpolation="bilinear")
        plt.axis("off")
        st.pyplot(fig1)
    elif choice == '查看每日疾病数据':
        df = monkey_pox_data.groupby('Date_confirmation')['Status'].size()
        fig , ax  = plt.subplots()
        ax = df.plot(kind='bar')
        st.pyplot(fig)

        fig1, ax1 = plt.subplots()
        ax1 = pd.to_datetime(monkey_pox_data['Date_confirmation']).dt.day_name().value_counts().plot(kind='bar')
        st.pyplot(fig1)
elif selected == '脑肿瘤预测':
    model = load_model(r"C:\Users\zzy\PycharmProjects\MultipleDiseasePre\multipleDisease\best_brain_model.h5")
    # model = load_model(r"best_brain_model.h5")
    img = st.file_uploader("请上传图片",type=['jpg', 'png', 'jpeg'])
    if img is not None:
        img = Image.open(img)
        st.image(img)
        img = img.resize((224,224))
        i = image.img_to_array(img)/255
        if i.shape == (224,224,1):
            i = cv2.cvtColor(i,cv2.COLOR_GRAY2BGR)
        input_arr = np.array([i])
        st.write(input_arr.shape)
        predict_x = model.predict(input_arr)
        classes_x = np.argmax(predict_x, axis=1)

        prediction = ''
        if classes_x == 0:
            prediction = "该图片数据为脑肿瘤患者"
            st.success(prediction)
        else:
            prediction = "该图片数据为健康状态的大脑"
            st.success(prediction)
