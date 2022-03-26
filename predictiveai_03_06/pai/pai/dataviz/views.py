import os
import io
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse, Http404
from django.core.files.storage import FileSystemStorage
from django.shortcuts import redirect
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from plotly.offline import plot
import plotly.graph_objs as go
import pandas as pd
from django.views.generic import TemplateView # Import TemplateView
import csv
import shutil
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
import seaborn as sns; sns.set()
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import time
import random as r
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression




flag='no'
target_col=""
primary_col=""
sl_type =""
category_col=[]
no_imp_feat=[]
category_features=[]
numeric_features=[]
SafeNumeric=[]


from django.views.generic import View
from django.shortcuts import redirect
from django.contrib import messages
from django.core.mail import send_mail

class SendFormEmail(View):
    def  get(self, request):

        # Get the form data
        name = request.GET.get('name', None)
        email = request.GET.get('email', None)
        message = request.GET.get('message', None)
        send_mail(
            'Subject - PredictiveAi User Feedback',


            'Email-address:\n'+email+'\n'+'User name:' + name + '\n' +'msg:'+ message,
            'predictiveai.tech@gmail.com', # Admin
            [
                'maheshsappani@gmail.com',
            ]
        )
        send_mail(
            'Help-Desk @PredictiveAi.tech ',
            "Hi "+name+""" ,\nThanks for contact us on PredictiveAi.tech
             Don't fired up!!.I am here to help you.
             Our team memember will contact you within a day\n
            \nCheer to the echo !\nMahesh @ PredictiveAi.tech team.
            \nThis emails is designed to hope you there is a active team behind the PredictiveAi.tech
            \nPredictiveAi.Tech: Open-Source Platform for Predictive-Analytics
            \nweb link: https://predictiveai.tech/
            \nlinkidin: https://www.linkedin.com/company/predictiveai-tech/\n
            \nEmail: predictiveai.tech@gmail.com"
            \nPh:+91 8925284860""",

            'predictiveai.tech@gmail.com', # Admin
            [
                email,
            ]
        )

        # Redirect to same page after form submit
        messages.success(request, ('!! Email sent successfully !!'))
        return redirect('contact')


def divide_feat(fname):
    try:
        global category_col,category_features,numeric_features,target_col,primary_col,SafeNumeric
        df = pd.read_csv(os.path.join(settings.MEDIA_ROOT, fname + '.csv'))
        category_features= [f for f in df.columns if df[f].dtype == object]

        if len(category_col)>0 and category_col[0] != '':
            for col in category_col:
                df[col] = df[col].astype('object')
            category_features=category_features+category_col
            category_features=list(set(category_features))
        numeric_features = [f for f in df.columns if df[f].dtype != object]

        numeric_features = set(numeric_features) - set(category_features)
        numeric_features=list(numeric_features)
        SafeNumeric=list(numeric_features)
        if target_col in SafeNumeric:
            SafeNumeric.remove(target_col)
        if primary_col in SafeNumeric:
            SafeNumeric.remove(primary_col)
        print(numeric_features)
        print(category_features)
        print(SafeNumeric)

    except Exception as e:
        print(e)

class FaqPageView(TemplateView):
    template_name = "faq.html"

class HomePageView(TemplateView):
    template_name = "index2.html"


def downloadOriginal(request, fName):
    file_path = os.path.join(settings.MEDIA_ROOT, 'original/' + fName + '.csv')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            return response
    raise Http404


def downloadProcessed(request, fName):
    file_path = os.path.join(settings.MEDIA_ROOT, fName + '.csv')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            return response
    raise Http404


def downloadInfo(request, fName):
    file_path = os.path.join(settings.MEDIA_ROOT, 'info/' + fName + '.csv')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            return response
    raise Http404


def upload(request):
    if request.method == 'POST':
        try:
            print("hai***************")
            x=os.path.join(settings.MEDIA_ROOT)
            shutil.rmtree(x)
            #x=x+'/original'
            #os.mkdir(x)
            #y=os.path.join(settings.PLOT_ROOT)
            list( map( os.unlink, (os.path.join( settings.PLOT_ROOT,f) for f in os.listdir(settings.PLOT_ROOT)) ) )
            #shutil.rmtree(y)
            #os.mkdir('/plot')
            print("hello***************")
        except Exception as e:
            print(e)


        uploadedFile = request.FILES['document']
        arr = uploadedFile.name.rsplit('.', 1)
        if arr[1] not in ['csv', 'xls', 'xlsx', 'json']:
            context = {
                'msg':'*Kindly upload .csv,.xml,.json,.xlsx formated file'
            }
            return render(request, 'upload.html', context)
        fs = FileSystemStorage()
        name = fs.save('paiData.csv', uploadedFile)
        fs.save('original/' + name, uploadedFile)
        arr[0] = name.rsplit('.', 1)[0]
        if arr[1] not in ['xls', 'xlsx']:
            f = open(os.path.join(settings.MEDIA_ROOT, name), "r")
            data = f.read().replace("–", "-")
            f.close()
            f = open(os.path.join(settings.MEDIA_ROOT, name), "w")
            f.write(data)
            f.close()
            f = open(os.path.join(settings.MEDIA_ROOT, 'original/' + name), "w")
            f.write(data)
            f.close()
        if arr[1] != 'csv':
            if arr[1] == 'xlsx' or arr[1] == 'xls':
                df = pd.read_excel(os.path.join(settings.MEDIA_ROOT, name))
            elif arr[1] == 'json':
                try:
                    df = pd.read_json(os.path.join(settings.MEDIA_ROOT, name))
                except:
                    df = pd.read_json(os.path.join(settings.MEDIA_ROOT, name), lines=True)
            os.remove(os.path.join(settings.MEDIA_ROOT, name))
            df = df.drop_duplicates()
            df.to_csv(os.path.join(settings.MEDIA_ROOT, name), index=False)
            os.remove(os.path.join(settings.MEDIA_ROOT, 'original/' + name))
            df.to_csv(os.path.join(settings.MEDIA_ROOT, 'original/' + name), index=False)
        return redirect('/info/' + arr[0] + '/')
    context = {
        'fName': '',
    }
    return render(request, 'upload.html', context)

def info(request,fName):
    global target_col
    global primary_col
    global sl_type
    global category_col

    df = pd.read_csv(os.path.join(settings.MEDIA_ROOT, fName + '.csv'))

    featureList = list(df)
    featureValues = ''
    count = ''
    context = {
        'fName': fName,
        'featureValues': featureValues,
        'count': count,
        'featureList': featureList,
        'div': '',

    }
    if request.method == 'POST':
        if 'dviz' in request.POST:
            if 'yes' == request.POST.get('dviz'):
                target_col = request.POST['tc']
                category_col = request.POST.getlist('cc')
                primary_col = request.POST['pc']
                sl_type = request.POST['ty']
                divide_feat(fName)
                return redirect('/visualizer3/' + fName + '/')
            else:
                target_col= request.POST['tc']
                primary_col= request.POST['pc']
                sl_type = request.POST['ty']
                category_col = request.POST.getlist('cc')
                divide_feat(fName)
                return redirect('/preprocess/' + fName + '/')
    return render(request, 'info.html', context)

def preprocess(request, fName):
    global no_imp_feat
    global category_features
    global target_col
    print(target_col)


    df = pd.read_csv(os.path.join(settings.MEDIA_ROOT, fName + '.csv'))
    divide_feat(fName)
    print("///////////")
    print(category_features)

    name = []


    if request.method == 'POST':
        name = request.POST.getlist('todelete')
    for feature in name:
        try:
            del df[feature]
        except Exception as e:
            print(e)
    df.to_csv(os.path.join(settings.MEDIA_ROOT, fName + '.csv'), index=False)
    df.head(50).to_csv("data1.csv")
    import io
    buf = io.StringIO()
    df.info(buf=buf)
    s = buf.getvalue()
    l = s.split("\n")
    l1, l2 = l[1], l[2]
    del l[0:3]
    l3 = []
    f = list(df)
    for (i, j) in zip(f, l):
        l4 = []
        l4.append(i)
        k = j.split(" ")
        l4.append(k[-1])
        l4.append(k[-2])
        l4.append(k[-3])
        l3.append(l4)
    clm_list = l3

    l4=[]
    l4.append(l1)
    l4.append(l2)
    df.describe().to_csv("my_description.csv")



    with open('my_description.csv', newline='') as f:
        reader = csv.reader(f)
        data1 = list(reader)
    with open('data1.csv', newline='') as f:
        reader = csv.reader(f)
        data2 = list(reader)
    data2h=data2[0]
    data2.pop(0)
    data1h=data1[0]
    data1.pop(0)
    clm_list1 = list(df)
    no_imp_feat=list(clm_list1)
    try:
        no_imp_feat.remove(target_col)
        no_imp_feat.remove(primary_col)
    except Exception as e:
        print(e)
    finally:
        context = {
            'clm_list': clm_list,
            'clm_list1': clm_list1,
            'l1':l1,
            'l2':l2,
            'data1h':data1h,
            'des_list':data1,
            'data2h': data2h,
            'des_list2': data2,
            'no_imp_feat':no_imp_feat,
            'fName': fName,
        }
        return render(request, 'preprocess.html', context)

def cleaning(request, fName):
    try:
        featureName = request.GET['f']
    except:
        return redirect('/preprocess/' + fName + '/')
    df = pd.read_csv(os.path.join(settings.MEDIA_ROOT, fName + '.csv'))

    if request.method == 'POST':
        tonull = request.POST.getlist('nulls')
        if request.POST.get('nullTreat') == 'treat':
            median = df.loc[:, featureName].median()[0]
            for value in tonull:
                df[featureName][df[featureName] == value] = np.nan
                try:
                    df[featureName][df[featureName] == float(value)] = np.nan
                except Exception as e:
                    print(e)
            df[featureName].fillna(median, inplace=True)

        elif request.POST.get('nullTreat') == 'ignore':
            for value in tonull:
                df = df[df[featureName] != value]
                try:
                    df = df[df[featureName] != float(value)]
                except Exception as e:
                    print(e)
            df = df.dropna(subset=[featureName])

    df.to_csv(os.path.join(settings.MEDIA_ROOT, fName + '.csv'), index=False)
    count_nan = len(df) - df[featureName].count()
    total = len(df)
    df = df[featureName].value_counts()
    df = df.sort_index(axis=0)
    features = df.index.get_values()
    cnt = list(df)
    fList = list(zip(features, cnt))
    fList = sorted(fList, key=lambda x: x[0])
    context = {
        'fName': fName,
        'featureName': featureName,
        'featureList': fList,
        'nullValues': count_nan,
        'total': total,
    }
    return render(request, 'cleaning.html', context)

def normalization(request, fName):
    df = pd.read_csv(os.path.join(settings.MEDIA_ROOT, fName + '.csv'))
    msg = ''
    msg1 = ''
    global numeric_features
    global SafeNumeric
    divide_feat(fName)
    print(SafeNumeric)

    if request.method == 'POST':
        try:
            method = request.POST.get('norm')
            if method == 'minMax':

                scaler = MinMaxScaler()
                scaler.fit(df[SafeNumeric])

                scaled = scaler.transform(df[SafeNumeric])

                for i, col in enumerate(SafeNumeric):
                    df[col] = scaled[:, i]

                df[SafeNumeric] = df[SafeNumeric].astype(float).round(decimals=3)
                msg = 'Numerical Data Normalized using Min-Max'
            elif method == 'zScore':

                scaler = StandardScaler()
                scaler.fit(df[SafeNumeric])
                scaled = scaler.transform(df[SafeNumeric])

                for i, col in enumerate(SafeNumeric):
                    df[col] = scaled[:, i]

                msg = 'Numerical Data Normalized using Z-Score'
            elif method == 'rscal':
                r_scaler = preprocessing.RobustScaler()
                scaled=r_scaler.fit_transform(df[SafeNumeric])
                for i, col in enumerate(SafeNumeric):
                    df[col] = scaled[:, i]

                msg = 'Numerical Data Normalized using Robust-Scaler'
            elif method == 'maxabs':
                n_scaler = preprocessing.MaxAbsScaler()
                scaled = n_scaler.fit_transform(df[SafeNumeric])

                for i, col in enumerate(SafeNumeric):
                    df[col] = scaled[:, i]

                msg = 'Numerical Data Normalized using MaxAbs_scaler'
            else:
                msg1 = '*Please Select a Method for Normalization'
        except Exception as e:
            print(e)
            msg1 = '*Failed to complete Normalization'
    df.to_csv(os.path.join(settings.MEDIA_ROOT, fName + '.csv'), index=False)



    context = {
        'fName': fName,

        'msg': msg,
        'msg1': msg1,
    }
    return render(request, 'normalization.html', context)



def labelEncoding(request, fName):
    df = pd.read_csv(os.path.join(settings.MEDIA_ROOT, fName + '.csv'))

    features = list(df)
    if request.method == 'POST':
        try:
            for ele in features:
                process = request.POST.get(ele)
                if process == 'label':
                    labelencoder = LabelEncoder()
                    df[ele] = labelencoder.fit_transform(df[ele])
                elif process == 'oneHot':
                    one_hot = pd.get_dummies(df[ele], prefix='type_' + ele)
                    df = df.drop(ele, axis=1)
                    df = df.join(one_hot)
                elif process == 'helmert':
                    ce_helmert = ce.HelmertEncoder(cols=[ele],drop_invariant=True)
                    hel=ce_helmert.fit_transform(df[ele])
                    df=df.drop(ele,axis=1)
                    df=df.join(hel)
                elif process == 'ceb':
                    ce_backward = ce.BackwardDifferenceEncoder(cols=[ele],drop_invariant=True)
                    ceb=ce_backward.fit_transform(df[ele])
                    df=df.drop(ele,axis=1)
                    df=df.join(ceb)
                elif process == 'binary':
                    ce_binary = ce.BinaryEncoder(handle_unknown='ignore')

                    cebi = ce_binary.fit_transform(df[ele])

                    df = df.drop(ele, axis=1)
                    df = df.join(cebi)
        except Exception as e:
            print(e)
        finally:
            pass
    df.to_csv(os.path.join(settings.MEDIA_ROOT, fName + '.csv'), index=False)
    dataType = []
    features = list(df)
    for ele in features:
        dType = str(df[ele].dtype)
        if dType == 'object':
            dataType.append('Categorical')
        else:
            dataType.append('Numeric')
    fList = list(zip(features, dataType))
    context = {
        'fName': fName,
        'featureList': fList,
    }
    return render(request, 'labelEncoding.html', context)


def view(request, fName):
    df = pd.read_csv(os.path.join(settings.MEDIA_ROOT, fName + '.csv'))
    import csv
    with open('data1.csv', newline='') as f:
        reader = csv.reader(f)
        data2 = list(reader)
    data2h = data2[0]
    data2.pop(0)

    context = {
        'fName': fName,
        'head': data2h,
        'values': data2,
    }
    return render(request, 'view.html', context)



def visualize1(request, fName):

    df = pd.read_csv(os.path.join(settings.MEDIA_ROOT, fName + '.csv'))
    global category_features
    global numeric_features
    global flag
    global target_col
    divide_feat(fName)
    flag='no'

    clm_list1 = list(df)

    try:
        numeric_features.remove(primary_col)
    except Exception as e:
        print(e)

    if request.method == 'POST':
        if request.POST.get("form_type") == 'bar_0':
            method = request.POST.get('vid')
            cat_x=request.POST.get('x')
            num_y=request.POST.getlist('y')
            l1=[]
            fig = go.Figure()
            for i in num_y:
                x0=tuple(df[cat_x])
                y0=tuple(df[i])
                a=go.Bar(name=i, x=x0, y=y0)
                l1.append(a)
            fig = go.Figure(data=l1)
            fig.update_layout(autosize=False, width=500, height=500,
                                  margin=dict(l=50, r=50, b=100, t=100, pad=4), )
            #fig.update_layout(barmode='group')
            #fig.update_traces(textposition='outside')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
            plot_div = plot(fig, output_type='div', filename='elevations-3d-surface')
            script = plot_div
            print(cat_x,num_y)
            context = {
                'fName': fName,
                'div': '',
                'cfeat':category_features,
                'nfeat':numeric_features,
                'script':script,

            }
            return render(request, 'visualizer3.html', context)
        elif request.POST.get("form_type") == 'scatter_0':
            print("form2")
            #os.remove("output.png")
            method = request.POST.get('vid')
            num_x = request.POST.get('x')
            num_y = request.POST.get('y')
            cat_z = request.POST.get('c')

            fy=os.path.join(settings.PLOT_ROOT)
            print(fy)


            timestr = time.strftime("%H%M%S")
            fx = 'output'+timestr+'.png'
            fy=fy+'/'+fx
            print(fy)
            try:
                sns.scatterplot(x=num_x, y=num_y, hue=cat_z, data=df)
                plt.savefig(fy)
                flag="yes"
                plt.close()
            except Exception as e:
                print(e)
                flag="no"

                #ax.savefig("dataviz/static/output.png")
            plt.close()
            context = {
                'fName': fName,
                'div': '',
                'cfeat': category_features,
                'nfeat': numeric_features,
                'flag':flag,
                'path1':fx,

            }
            flag='no'
            return render(request, 'visualizer3.html', context)
        elif request.POST.get("form_type") == 'scatter_1':
            method = request.POST.get('vid')
            num_x=request.POST.get('x')
            num_y=request.POST.getlist('y')
            l1=[]
            fig = go.Figure()
            for i in num_y:
                x0=tuple(df[num_x])
                y0=tuple(df[i])
                a=go.Scatter(name=i, x=x0, y=y0,mode='markers')
                l1.append(a)
            fig = go.Figure(data=l1)
            fig.update_layout(autosize=False, width=500, height=500,
                                  margin=dict(l=50, r=50, b=100, t=100, pad=4),paper_bgcolor="LightSteelBlue", )
            fig.update_traces(textposition='top center')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            plot_div = plot(fig, output_type='div', filename='elevations-3d-surface')
            script = plot_div
            context = {
                'fName': fName,
                'div': '',
                'nfeat':numeric_features,
                'cfeat': category_features,
                'flag': flag,
                'script':script,

            }
            return render(request, 'visualizer3.html', context)
        elif request.POST.get("form_type") == 'box_0':
            method = request.POST.get('vid')
            num_y=request.POST.getlist('y')
            l1=[]
            c=['indianred','lightseagreen','DarkSlateGrey','MediumPurple']
            fig = go.Figure()
            for i in num_y:
                r1=r.randint(0,3)
                y0 = tuple(df[i])
                a=go.Box(name=i,y=y0, marker_color=c[r1])
                l1.append(a)
            fig = go.Figure(data=l1)
            fig.update_layout(autosize=False, width=500, height=500,
                                  margin=dict(l=50, r=50, b=100, t=100, pad=4),paper_bgcolor="LightSteelBlue", )

            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            plot_div = plot(fig, output_type='div', filename='elevations-3d-surface')
            script = plot_div
            context = {
                'fName': fName,
                'div': '',
                'flag': flag,
                'script':script,
                'cfeat': category_features,
                'nfeat': numeric_features,

            }
            return render(request, 'visualizer3.html', context)
        elif request.POST.get("form_type") == 'pair_0':
            # os.remove("output.png")
            method = request.POST.get('vid')
            fy = os.path.join(settings.PLOT_ROOT)
            timestr = time.strftime("%H%M%S")
            fx = 'output'+timestr+'.png'
            fy = fy + '/' + fx
            print("///////////**********")
            print(target_col)
            t=target_col


            try:
                sns.pairplot(df, hue=t)
                plt.savefig(fy)
                flag = "yes"
                print("///////////**********")
            except Exception as e:
                print(e)
                print("**********")
                flag = "no"

                # ax.savefig("dataviz/static/output.png")
            plt.close()
            context = {
                'fName': fName,
                'div': '',
                'cfeat': category_features,
                'nfeat': numeric_features,
                'flag': flag,
                'path1': fx,

            }
            flag = 'no'
            return render(request, 'visualizer3.html', context)
        elif request.POST.get("form_type") == 'heat_0':
            method = request.POST.get('vid')
            fy = os.path.join(settings.PLOT_ROOT)
            timestr = time.strftime("%H%M%S")
            fx = 'output'+timestr+'.png'
            fy = fy + '/' + fx

            try:
                f, ax = plt.subplots(figsize=(9, 9))
                sns.heatmap(df.corr(), annot=True, linewidths=0.5, linecolor="red", fmt='.2f', ax=ax)
                plt.savefig(fy)
                flag = "yes"

            except Exception as e:
                print(e)

                flag = "no"


            plt.close()
            context = {
                'fName': fName,
                'div': '',
                'cfeat': category_features,
                'nfeat': numeric_features,
                'flag': flag,
                'path1': fx,


            }
            flag = 'no'
            return render(request, 'visualizer3.html', context)
        elif request.POST.get("form_type") == 'lm_0':

            method = request.POST.get('vid')
            num_x = request.POST.get('x')
            num_y = request.POST.get('y')


            fy=os.path.join(settings.PLOT_ROOT)
            timestr = time.strftime("%H%M%S")
            fx = 'output'+timestr+'.png'
            fy=fy+'/'+fx

            try:
                sns.lmplot(x=num_x, y=num_y, data=df)
                plt.savefig(fy)
                flag="yes"
                plt.close()
            except Exception as e:
                print(e)
                flag="no"

                #ax.savefig("dataviz/static/output.png")
            plt.close()
            feat = list(df)

            context = {
                'fName': fName,
                'feat': feat,
                'div': '',
                'cfeat': category_features,
                'nfeat': numeric_features,
                'flag':flag,
                'path1':fx,

            }
            flag='no'
            return render(request, 'visualizer3.html', context)
        elif request.POST.get("form_type") == 'count_0':

            method = request.POST.get('vid')
            num_x = request.POST.get('x')

            fy = os.path.join(settings.PLOT_ROOT)
            timestr = time.strftime("%H%M%S")
            fx = 'output'+timestr+'.png'
            fy = fy + '/' + fx

            try:
                sns.set(style="darkgrid")
                sns.countplot(df[num_x])
                plt.savefig(fy)
                flag = "yes"
                plt.close()
            except Exception as e:
                print(e)
                flag = "no"

                # ax.savefig("dataviz/static/output.png")
            plt.close()
            feat = list(df)

            context = {
                'fName': fName,
                'div': '',
                'feat': feat,
                'cfeat': category_features,
                'nfeat': numeric_features,
                'flag': flag,
                'path1': fx,

            }
            flag = 'no'
            return render(request, 'visualizer3.html', context)
    feat=list(df)

    context = {
        'fName': fName,
        'cfeat': category_features,
        'nfeat': numeric_features,
        'feat':feat,
        'flag':flag,
    }
    return render(request, 'visualizer3.html', context)

def featureSelection(request, fName):
    df = pd.read_csv(os.path.join(settings.MEDIA_ROOT, fName + '.csv'))
    selectedFeature = ''
    max_leaf_nodes = None
    criterion = 'gini'
    if request.method == 'POST':
        selectedFeature = request.POST.getlist('selected-feature')
        criterion = request.POST['criterion']
        try:
            max_leaf_nodes = int(request.POST['max_leaf_nodes'])
        except:
            pass
        name = request.POST.getlist('toDelete')
        if len(name) > 0:
            for ele in name:
                try:
                    del (df[ele])
                except:
                    pass
    df.to_csv(os.path.join(settings.MEDIA_ROOT, fName + '.csv'), index=False)
    featureNames = list(df)
    fList = []
    if len(selectedFeature) > 0:
        selectedFeature = selectedFeature[0]
        if len(selectedFeature) > 0:
            train, test = train_test_split(df, test_size=0.2)
            clf = DecisionTreeClassifier(criterion=criterion, max_leaf_nodes=max_leaf_nodes)
            clf = clf.fit(train.drop([selectedFeature], axis=1).values, train[selectedFeature].values)
            features = train.drop([selectedFeature], axis=1).columns
            dataType = clf.feature_importances_
            fList = list(zip(features, dataType))
            fList = sorted(fList, key=lambda tup: tup[1])
    context = {
        'fName': fName,
        'featureList': fList,
        'featureNames': featureNames,
        'selectedFeature': selectedFeature,
        'max_leaf_nodes': max_leaf_nodes,
    }
    return render(request, 'featureSelection.html', context)


def cluster(request, fName):
    from sklearn.cluster import KMeans
    featureNames = []
    centers = []
    n_clusters = ''
    if request.method == 'POST':
        n_clusters = int(request.POST['clusters'])
        df = pd.read_csv(os.path.join(settings.MEDIA_ROOT, fName + '.csv'))
        featureNames = list(df)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df)
        centers = kmeans.cluster_centers_
    context = {
        'featureNames': featureNames,
        'fName': fName,
        'centers': centers,
        'n_clusters': n_clusters,
    }
    return render(request, 'cluster.html', context)

def testfile(uploadedFile):
    arr = uploadedFile.name.rsplit('.', 1)
    fs = FileSystemStorage()
    name = fs.save('paitestData.csv', uploadedFile)
    fs.save('original/' + name, uploadedFile)
    arr[0] = name.rsplit('.', 1)[0]
    if arr[1] not in ['xls', 'xlsx']:
        f = open(os.path.join(settings.MEDIA_ROOT, name), "r")
        data = f.read().replace("–", "-")
        f.close()
        f = open(os.path.join(settings.MEDIA_ROOT, name), "w")
        f.write(data)
        f.close()
        f = open(os.path.join(settings.MEDIA_ROOT, 'original/' + name), "w")
        f.write(data)
        f.close()
    if arr[1] != 'csv':
        if arr[1] == 'xlsx' or arr[1] == 'xls':
            df = pd.read_excel(os.path.join(settings.MEDIA_ROOT, name))
        elif arr[1] == 'json':
            try:
                df = pd.read_json(os.path.join(settings.MEDIA_ROOT, name))
            except:
                df = pd.read_json(os.path.join(settings.MEDIA_ROOT, name), lines=True)
        os.remove(os.path.join(settings.MEDIA_ROOT, name))
        df = df.drop_duplicates()
        df.to_csv(os.path.join(settings.MEDIA_ROOT, name), index=False)
        os.remove(os.path.join(settings.MEDIA_ROOT, 'original/' + name))
        df.to_csv(os.path.join(settings.MEDIA_ROOT, 'original/' + name), index=False)
    return arr[0]
def predictionCC(request, fName):

    global target_col
    global primary_col
    global sl_type
    flagCC=0
    fileD=0
    flagR=0
    divide_feat(fName)
    print(target_col)
    print("=====+++++++")
    fy = os.path.join(settings.PLOT_ROOT)
    print(fy)
    timestr = time.strftime("%H%M%S")
    fx = 'PredictiveAiResult'+timestr+'.docx'

    fy = fy + '/' + fx
    algoC = ['Logistic Regression', 'KNN', 'SVM', 'Decision Tree', 'Gaussian Naive Bayes', 'Random Forest Classifier']
    algoR = ['Linear Regression', 'GradientBoosting Regression']
    print(sl_type)
    df = pd.read_csv(os.path.join(settings.MEDIA_ROOT, fName + '.csv'))
    y=df[target_col]
    df=df.drop([target_col],axis=1)
    originalFeats=df
    df=df.drop([primary_col],axis=1)
    x=df
    xfeatList=list(df)


    algo=[]
    if sl_type=='classification':
        algo=algoC
    else:
        algo=algoR

    if request.method == 'POST':
        pcaValue = request.POST.get('pcaV')
        algoValue = request.POST.get('algoV')
        try:
            uploadedFile = request.FILES['document2']
            arr = uploadedFile.name.rsplit('.', 1)
            if arr[1] not in ['csv', 'xls', 'xlsx', 'json']:
                context = {
                    'msg': 'Not supported file format' + arr[1] + 'upload .xmls/.csv/.xml/.json. formated file'
                }
                return render(request, 'prediction.html', context)
            #print(uploadedFile)
            tfname=testfile(uploadedFile)
            print(tfname)
            testdf = pd.read_csv(os.path.join(settings.MEDIA_ROOT, tfname + '.csv'))
            if set(testdf) != set(originalFeats):
                context = {
                    'msg': "** Test Data should contain "+str(list(originalFeats))+" features",
                    'fName': fName,
                    'algo': algo,
                    'feature': xfeatList,
                }
                return render(request, 'prediction.html', context)

            try:
                flagCC=1
                testid=testdf[primary_col]
                testdf=testdf.drop([primary_col],axis=1)
            except Exception as e:
                print(e)
                context = {
                    'msg': e,
                    'fName': fName,
                    'algo': algo,
                    'feature': xfeatList,
                }
                return render(request, 'prediction.html', context)
        except:
            pass
            testdf=[]
            flagCC=0

            for i in xfeatList:
                i = request.POST.get(i)
                testdf.append(i)
            testdf=np.reshape(testdf, (1,-1))
            testdf=testdf.astype(np.float64)

        print(pcaValue,algoValue)
        if algoValue=='KNN':
            try:
                knn2 = KNeighborsClassifier()
                param_grid = {'n_neighbors': np.arange(1, 20), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
                knn_gscv = GridSearchCV(knn2, param_grid, cv=2)
                knn_gscv.fit(x, y)
                y_pred = knn_gscv.predict(testdf)
                y_pred=np.array(y_pred).tolist()
                paras=str(knn_gscv.best_params_)
                accur=str(round(knn_gscv.best_score_*100,2))
                file1 = open(fy, "w")
                file1.write("Algorithm:"+algoValue)
                file1.write("\nHyper Tuning Parameters:"+paras)
                file1.write("\nModel Accuracy:"+accur)
                file1.write("\n\nThank you for visiting us.")
                file1.write("\nContact Us:predictiveai.tech@gmail.com\nPh:+91 9994381272")
                file1.write("\nTest Result:\n\n")
                if flagCC==1:
                    a = list(testid)
                    b = y_pred
                    p = zip(a, b)
                    s=""
                    for a, b in p:
                        s=s+str(a)+" "+str(b)+"\n"
                    file1.write(s)
                else:
                    file1.write(str(y_pred))

                file1.close()
                fileD=1


            except Exception as e:
                error1:e
                print(e)
                context = {
                    'msg': e,
                    'fName': fName,
                    'algo': algo,
                    'feature': xfeatList,
                }
                return render(request, 'prediction.html', context)
        elif algoValue == 'Random Forest Classifier':
            try:
                model = RandomForestClassifier()
                params = {'criterion': ['gini', 'entropy'],
                          'n_estimators': [10, 15, 20, 25, 30],
                          'min_samples_leaf': [1, 2, 3],
                          'min_samples_split': [3, 4, 5, 6, 7],
                          'random_state': [123],
                          'n_jobs': [-1]}
                random_c = GridSearchCV(model, param_grid=params)
                random_c.fit(x, y)
                y_pred = random_c.predict(testdf)
                y_pred = np.array(y_pred).tolist()
                paras = str(random_c.best_params_)
                accur = str(round(random_c.best_score_ * 100, 2))
                file1 = open(fy, "w")
                file1.write("Algorithm:" + algoValue)
                file1.write("\nHyper Tuning Parameters:" + paras)
                file1.write("\nModel Accuracy:" + accur)
                file1.write("\n\nThank you for visiting us.")
                file1.write("\nContact Us:predictiveai.tech@gmail.com\nPh:+91 9994381272")
                file1.write("\nTest Result:\n\n")
                if flagCC == 1:
                    a = list(testid)
                    b = y_pred
                    p = zip(a, b)
                    s = ""
                    for a, b in p:
                        s = s + str(a) + " " + str(b) + "\n"
                    file1.write(s)
                else:
                    file1.write(str(y_pred))

                file1.close()
                fileD = 1
            except Exception as e:
                error1: e
                print(e)
                context = {
                    'msg': e,
                    'fName': fName,
                    'algo': algo,
                    'feature': xfeatList,
                }
                return render(request, 'prediction.html', context)
        elif algoValue == 'SVM':
            try:

                svm_c = svm.SVC()
                params = {'C': [6, 7, 8, 9, 10, 11, 12],
                          'kernel': ['linear', 'rbf']}
                svm_c = GridSearchCV(svm_c, param_grid=params)
                svm_c.fit(x, y)
                y_pred = svm_c.predict(testdf)
                y_pred = np.array(y_pred).tolist()
                paras = str(svm_c.best_params_)
                accur = str(round(svm_c.best_score_ * 100, 2))
                file1 = open(fy, "w")
                file1.write("Algorithm:" + algoValue)
                file1.write("\nHyper Tuning Parameters:" + paras)
                file1.write("\nModel Accuracy:" + accur)
                file1.write("\n\nThank you for visiting us.")
                file1.write("\nContact Us:predictiveai.tech@gmail.com\nPh:+91 9994381272")
                file1.write("\nTest Result:\n\n")
                if flagCC == 1:
                    a = list(testid)
                    b = y_pred
                    p = zip(a, b)
                    s = ""
                    for a, b in p:
                        s = s + str(a) + " " + str(b) + "\n"
                    file1.write(s)
                else:
                    file1.write(str(y_pred))

                file1.close()
                fileD = 1
            except Exception as e:
                error1: e
                print(e)
                context = {
                    'msg': e,
                    'fName': fName,
                    'algo': algo,
                    'feature': xfeatList,
                }
                return render(request, 'prediction.html', context)
        elif algoValue == 'Decision Tree':
            try:

                model = DecisionTreeClassifier(random_state=1234)
                # Hyper Parameters Set
                params = {'max_features': ['auto', 'sqrt', 'log2'],'criterion':['gini', 'entropy'],'splitter':['best', 'random'],
                          'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                          'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                          'random_state': [123]}
                # Making models with hyper parameters sets
                decision_c = GridSearchCV(model, param_grid=params)
                decision_c.fit(x, y)
                y_pred = decision_c.predict(testdf)
                y_pred = np.array(y_pred).tolist()
                paras = str(decision_c.best_params_)
                accur = str(round(decision_c.best_score_ * 100, 2))
                file1 = open(fy, "w")
                file1.write("Algorithm:" + algoValue)
                file1.write("\nHyper Tuning Parameters:" + paras)
                file1.write("\nModel Accuracy:" + accur)
                file1.write("\n\nThank you for visiting us.")
                file1.write("\nContact Us:predictiveai.tech@gmail.com\nPh:+91 9994381272")
                file1.write("\nTest Result:\n\n")
                if flagCC == 1:
                    a = list(testid)
                    b = y_pred
                    p = zip(a, b)
                    s = ""
                    for a, b in p:
                        s = s + str(a) + " " + str(b) + "\n"
                    file1.write(s)
                else:
                    file1.write(str(y_pred))

                file1.close()
                fileD = 1
            except Exception as e:
                error1: e
                print(e)
                context = {
                    'msg': e,
                    'fName': fName,
                    'algo': algo,
                    'feature': xfeatList,
                }
                return render(request, 'prediction.html', context)
        elif algoValue == 'Logistic Regression':
            try:
                clf = LogisticRegression()
                grid_values = {'multi_class':['auto', 'ovr', 'multinomial'],'C': [0.001, .009, 0.01, .09, 1, 5, 10, 25]}
                scorer = sklearn.metrics.make_scorer(sklearn.metrics.f1_score, average='weighted')
                logistic_c = GridSearchCV(clf, param_grid=grid_values,scoring=scorer)
                logistic_c.fit(x, y)
                y_pred = logistic_c.predict(testdf)
                y_pred = np.array(y_pred).tolist()
                paras = str(logistic_c.best_params_)
                accur = str(round(logistic_c.best_score_ * 100, 2))
                file1 = open(fy, "w")
                file1.write("Algorithm:" + algoValue)
                file1.write("\nHyper Tuning Parameters:" + paras)
                file1.write("\nModel Accuracy:" + accur)
                file1.write("\n\nThank you for visiting us.")
                file1.write("\nContact Us:predictiveai.tech@gmail.com\nPh:+91 9994381272")
                file1.write("\nTest Result:\n\n")
                if flagCC == 1:
                    a = list(testid)
                    b = y_pred
                    p = zip(a, b)
                    s = ""
                    for a, b in p:
                        s = s + str(a) + " " + str(b) + "\n"
                    file1.write(s)
                else:
                    file1.write(str(y_pred))

                file1.close()
                fileD = 1
            except Exception as e:
                error1: e
                print(e)
                context = {
                    'msg': e,
                    'fName': fName,
                    'algo': algo,
                    'feature': xfeatList,
                }
                return render(request, 'prediction.html', context)
        elif algoValue == 'Gaussian Naive Bayes':
            try:

                gnb = GaussianNB()
                X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2,
                                                                    random_state=109)
                gnb.fit(X_train,y_train)
                y_pred=gnb.predict(X_test)
                acc=metrics.accuracy_score(y_test, y_pred)
                y_pred = gnb.predict(testdf)
                y_pred = np.array(y_pred).tolist()
                accur = str(round(acc* 100, 2))
                file1 = open(fy, "w")
                file1.write("Algorithm:" + algoValue)
                file1.write("\nModel Accuracy:" + accur)
                file1.write("\n\nThank you for visiting us.")
                file1.write("\nContact Us:predictiveai.tech@gmail.com\nPh:+91 9994381272")
                file1.write("\nTest Result:\n\n")
                if flagCC == 1:
                    a = list(testid)
                    b = y_pred
                    p = zip(a, b)
                    s = ""
                    for a, b in p:
                        s = s + str(a) + " " + str(b) + "\n"
                    file1.write(s)
                else:
                    file1.write(str(y_pred))

                file1.close()
                fileD = 1
            except Exception as e:
                error1: e
                print(e)
                context = {
                    'msg': e,
                    'fName': fName,
                    'algo': algo,
                    'feature': xfeatList,
                }
                return render(request, 'prediction.html', context)
        elif algoValue == 'GradientBoosting Regression':
            try:
                model = GradientBoostingRegressor()
                parameters = {'learning_rate': [0.01, 0.02, 0.03],
                              'subsample': [0.9, 0.5, 0.2],
                              'n_estimators': [100, 500, 1000],
                              'max_depth': [4, 6, 8]
                              }
                grid_r = GridSearchCV(estimator=model, param_grid=parameters, cv=2)
                grid_r.fit(x, y)
                y_pred = grid_r.predict(testdf)
                y_pred = np.array(y_pred).tolist()
                paras = str(grid_r.best_params_)
                accur = str(round(grid_r.best_score_ * 100, 2))
                file1 = open(fy, "w")
                file1.write("Algorithm:" + algoValue)
                file1.write("\nHyper Tuning Parameters:" + paras)
                file1.write("\nModel Accuracy:" + accur)
                file1.write("\n\nThank you for visiting us.")
                file1.write("\nContact Us:predictiveai.tech@gmail.com\nPh:+91 9994381272")
                file1.write("\nTest Result:\n\n")
                if flagCC == 1:
                    a = list(testid)
                    b = y_pred
                    p = zip(a, b)
                    s = ""
                    for a, b in p:
                        s = s + str(a) + " " + str(b) + "\n"
                    file1.write(s)
                else:
                    file1.write(str(y_pred))

                file1.close()
                fileD = 1
            except Exception as e:
                error1: e
                print(e)
                context = {
                    'msg': e,
                    'fName': fName,
                    'algo': algo,
                    'feature': xfeatList,
                }
                return render(request, 'prediction.html', context)
        elif algoValue == 'Linear Regression':
            try:

                reg_r = LinearRegression()
                parameters = {'fit_intercept':['bool', 'optional','True'],'normalize':['bool','False']

                              }
                reg_r = GridSearchCV(estimator=reg_r, param_grid=parameters, cv=2)
                reg_r.fit(x, y)
                y_pred = reg_r.predict(testdf)
                y_pred = np.array(y_pred).tolist()
                paras = str(reg_r.best_params_)
                accur = str(round(reg_r.best_score_ * 100, 2))
                file1 = open(fy, "w")
                file1.write("Algorithm:" + algoValue)
                file1.write("\nHyper Tuning Parameters:" + paras)
                file1.write("\nModel Accuracy:" + accur)
                file1.write("\n\nThank you for visiting us.")
                file1.write("\nContact Us:predictiveai.tech@gmail.com\nPh:+91 9994381272")
                file1.write("\nTest Result:\n\n")
                if flagCC == 1:
                    a = list(testid)
                    b = y_pred
                    p = zip(a, b)
                    s = ""
                    for a, b in p:
                        s = s + str(a) + " " + str(b) + "\n"
                    file1.write(s)
                else:
                    file1.write(str(y_pred))

                file1.close()
                fileD = 1
            except Exception as e:
                error1: e
                print(e)
                context = {
                    'msg': e,
                    'fName': fName,
                    'algo': algo,
                    'feature': xfeatList,
                }
                return render(request, 'prediction.html', context)

    context = {
        'fName': fName,
        'algo': algo,
        'feature':xfeatList,
        'fileD':fileD,
        'path2':fx,


    }
    #return redirect('/prediction.html/' + fName + '/')
    return render(request, 'prediction.html', context)


"""
sending dictionary and access from html
    featuresPred = list(df)
   for i in featuresPred:
        l1 = []
        l = []
        a = df[i].min()
        l1.append(a)
        b = df[i].max()
        l1.append(b)

        dataCol[i] = l1

html page:
{% for f, values in dataCol.items %}
<h1>{{ values.0 }}</h1>

 <div class="input-field text-white ">
     <label style="color:white; line-height: 1.8; text-align:left;font-size:17px; font-family: 'Comic Sans MS';" > {{ f }}:</label>
                   <p class="range-field">
                        <input type="range" id="{{ f }}" name="{{ f }}" min="{{ values.0 }}" max="{{ values.1 }}" value="0" step="1">

                  </div><br>
       {% endfor %}

"""

def classify(request, fName):
    import datetime
    now = datetime.datetime.now()
    context = {
        'fName': fName,
        'time': now,
    }
    return render(request, 'classify.html', context)
