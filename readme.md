<!-- TOC -->

- [支持向量机分类](#支持向量机分类)
- [支持向量机 网格搜索](#支持向量机-网格搜索)
- [临近法](#临近法)
- [决策树](#决策树)
- [随机森林](#随机森林)
- [bagging方法](#bagging方法)
- [Gradient Boosting方法](#gradient-boosting方法)
- [xgboost 方法](#xgboost-方法)

<!-- /TOC -->

```python
import pandas as pd
import numpy as np
import  matplotlib.pyplot as  plt
import time
```


```python
df=pd.read_excel("all_data_meituan.xlsx")[["comment","star"]]
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comment</th>
      <th>star</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>还行吧，建议不要排队那个烤鸭和羊肉串，因为烤肉时间本来就不够，排那个要半小时，然后再回来吃烤...</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>去过好几次了 东西还是老样子 没增添什么新花样 环境倒是挺不错 离我们这也挺近 味道还可以 ...</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>一个字：好！！！ #羊肉串# #五花肉# #牛舌# #很好吃# #鸡软骨# #拌菜# #抄河...</td>
      <td>50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>第一次来吃，之前看过好多推荐说这个好吃，真的抱了好大希望，排队的人挺多的，想吃得趁早来啊。还...</td>
      <td>20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>羊肉串真的不太好吃，那种说膻不膻说臭不臭的味。烤鸭还行，大虾没少吃，也就到那吃大虾了，吃完了...</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (17400, 2)




```python
df['sentiment']=df['star'].apply(lambda x:1 if x>30 else 0)
df=df.drop_duplicates() ## 去掉重复的评论
df=df.dropna()
```


```python
X=pd.concat([df[['comment']],df[['comment']],df[['comment']]])
y=pd.concat([df.sentiment,df.sentiment,df.sentiment])
X.columns=['comment']
X.reset_index
X.shape
```




    (3138, 1)




```python
import jieba
def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))
X['cut_comment']=X["comment"].apply(chinese_word_cut)
X['cut_comment'].head()
```

    Building prefix dict from the default dictionary ...
    Loading model from cache C:\Users\FRED-H~1\AppData\Local\Temp\jieba.cache
    Loading model cost 0.651 seconds.
    Prefix dict has been built succesfully.
    




    0    还行 吧 ， 建议 不要 排队 那个 烤鸭 和 羊肉串 ， 因为 烤肉 时间 本来 就 不够...
    1    去过 好 几次 了   东西 还是 老 样子   没 增添 什么 新花样   环境 倒 是 ...
    2    一个 字 ： 好 ！ ！ ！   # 羊肉串 #   # 五花肉 #   # 牛舌 #   ...
    3    第一次 来 吃 ， 之前 看过 好多 推荐 说 这个 好吃 ， 真的 抱 了 好 大 希望 ...
    4    羊肉串 真的 不太 好吃 ， 那种 说 膻 不 膻 说 臭 不 臭 的 味 。 烤鸭 还 行...
    Name: cut_comment, dtype: object




```python
from sklearn.model_selection import  train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,random_state=42,test_size=0.25)
```


```python
def get_custom_stopwords(stop_words_file):
    with open(stop_words_file,encoding="utf-8") as f:
        custom_stopwords_list=[i.strip() for i in f.readlines()]
    return custom_stopwords_list
```


```python
stop_words_file = "stopwords.txt"
stopwords = get_custom_stopwords(stop_words_file)
stopwords[-10:]
```




    ['100', '01', '02', '03', '04', '05', '06', '07', '08', '09']




```python
from sklearn.feature_extraction.text import  CountVectorizer
vect=CountVectorizer()
vect
```




    CountVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), preprocessor=None, stop_words=None,
            strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
            tokenizer=None, vocabulary=None)




```python
vect.fit_transform(X_train["cut_comment"])
```




    <2353x1965 sparse matrix of type '<class 'numpy.int64'>'
    	with 20491 stored elements in Compressed Sparse Row format>




```python
vect.fit_transform(X_train["cut_comment"]).toarray().shape
```




    (2353, 1965)




```python
# pd.DataFrame(vect.fit_transform(X_train["cut_comment"]).toarray(),columns=vect.get_feature_names()).iloc[:10,:22]
# print(vect.get_feature_names())
# #  数据维数1956，不算很大（未使用停用词）
```


```python
vect = CountVectorizer(token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',stop_words=frozenset(stopwords)) # 去除停用词
pd.DataFrame(vect.fit_transform(X_train['cut_comment']).toarray(), columns=vect.get_feature_names()).head()
# 1691 columns,去掉以数字为特征值的列，减少了三列编程1691 
# max_df = 0.8 # 在超过这一比例的文档中出现的关键词（过于平凡），去除掉。
# min_df = 3 # 在低于这一数量的文档中出现的关键词（过于独特），去除掉。
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amazing</th>
      <th>happy</th>
      <th>ktv</th>
      <th>pm2</th>
      <th>一万个</th>
      <th>一个多</th>
      <th>一个月</th>
      <th>一串</th>
      <th>一人</th>
      <th>一件</th>
      <th>...</th>
      <th>麻烦</th>
      <th>麻酱</th>
      <th>黄喉</th>
      <th>黄桃</th>
      <th>黄花鱼</th>
      <th>黄金</th>
      <th>黑乎乎</th>
      <th>黑椒</th>
      <th>黑胡椒</th>
      <th>齐全</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1691 columns</p>
</div>




```python
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn import  metrics
svc_cl=SVC()
pipe=make_pipeline(vect,svc_cl)
pipe.fit(X_train.cut_comment, y_train)
```




    Pipeline(memory=None,
         steps=[('countvectorizer', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), preprocessor=None,
            stop_words=...,
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False))])




```python
y_pred = pipe.predict(X_test.cut_comment)
metrics.accuracy_score(y_test,y_pred)
```




    0.6318471337579618




```python
metrics.confusion_matrix(y_test,y_pred)
```




    array([[  0, 289],
           [  0, 496]], dtype=int64)



####  支持向量机分类


```python
from sklearn.svm import SVC
svc_cl=SVC() # 实例化
pipe=make_pipeline(vect,svc_cl)
pipe.fit(X_train.cut_comment, y_train)
y_pred = pipe.predict(X_test.cut_comment)
metrics.accuracy_score(y_test,y_pred)
```




    0.6318471337579618



#### 支持向量机 网格搜索


```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import  Pipeline
# svc=SVC(random_state=1)
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
tfidf=TfidfTransformer()
# ('tfidf',
#                       TfidfTransformer()),
#                      ('clf',
#                       SGDClassifier(max_iter=1000)),
# svc=SGDClassifier(max_iter=1000)
svc=SVC()
# pipe=make_pipeline(vect,SVC)
pipe_svc=Pipeline([("scl",vect),('tfidf',tfidf),("clf",svc)])
para_range=[0.0001,0.001,0.01,0.1,1.0,10,100,1000]
para_grid=[
    {'clf__C':para_range,
    'clf__kernel':['linear']},
    {'clf__gamma':para_range,
    'clf__kernel':['rbf']}
]
```


```python
gs=GridSearchCV(estimator=pipe_svc,param_grid=para_grid,cv=10,n_jobs=-1)
```


```python
gs.fit(X_train.cut_comment,y_train)
```




    GridSearchCV(cv=10, error_score='raise',
           estimator=Pipeline(memory=None,
         steps=[('scl', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), preprocessor=None,
            stop_words=frozenset({'...,
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False))]),
           fit_params=None, iid=True, n_jobs=-1,
           param_grid=[{'clf__C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000], 'clf__kernel': ['linear']}, {'clf__gamma': [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000], 'clf__kernel': ['rbf']}],
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)




```python
gs.best_estimator_.fit(X_train.cut_comment,y_train)
```




    Pipeline(memory=None,
         steps=[('scl', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), preprocessor=None,
            stop_words=frozenset({'...,
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False))])




```python
y_pred = gs.best_estimator_.predict(X_test.cut_comment)
metrics.accuracy_score(y_test,y_pred)
```




    0.9503184713375796



#### 临近法


```python
from sklearn.neighbors import  KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
pipe=make_pipeline(vect,knn)
pipe.fit(X_train.cut_comment, y_train)
```




    Pipeline(memory=None,
         steps=[('countvectorizer', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), preprocessor=None,
            stop_words=...owski',
               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
               weights='uniform'))])




```python
y_pred = pipe.predict(X_test.cut_comment)
metrics.accuracy_score(y_test,y_pred)
```




    0.7070063694267515




```python
metrics.confusion_matrix(y_test,y_pred)
```




    array([[ 87, 202],
           [ 28, 468]], dtype=int64)



#### 决策树


```python
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(criterion='entropy',random_state=1)
```


```python
pipe=make_pipeline(vect,tree)
pipe.fit(X_train.cut_comment, y_train)
y_pred = pipe.predict(X_test.cut_comment)
metrics.accuracy_score(y_test,y_pred)
```




    0.9388535031847134




```python
metrics.confusion_matrix(y_test,y_pred)
```




    array([[256,  33],
           [ 15, 481]], dtype=int64)



#### 随机森林


```python

from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(criterion='entropy',random_state=1,n_jobs=2)
pipe=make_pipeline(vect,forest)
pipe.fit(X_train.cut_comment, y_train)
y_pred = pipe.predict(X_test.cut_comment)
metrics.accuracy_score(y_test,y_pred)
# 加上tfidf反而准确率96.5至95.0，
```




    0.9656050955414013




```python
metrics.confusion_matrix(y_test,y_pred)
```




    array([[265,  24],
           [  3, 493]], dtype=int64)



#### bagging方法


```python

from sklearn.ensemble import BaggingClassifier
bag=BaggingClassifier(base_estimator=tree,
                     n_estimators=10,
                     max_samples=1.0,
                     max_features=1.0,
                     bootstrap=True,
                     bootstrap_features=False,
                     n_jobs=1,random_state=1)
pipe=make_pipeline(vect,tfidf,bag)
pipe.fit(X_train.cut_comment, y_train)
y_pred = pipe.predict(X_test.cut_comment)
metrics.accuracy_score(y_test,y_pred)  #  没用转化td-idf 93.2%, 加上转化步骤，准确率提升到95.5
```




    0.9554140127388535




```python
metrics.confusion_matrix(y_test,y_pred)
```




    array([[260,  29],
           [  6, 490]], dtype=int64)


####  Gradient Boosting方法


```python
from sklearn.ensemble import GradientBoostingClassifier
grd = GradientBoostingClassifier(learning_rate=0.18,max_depth=10,n_estimators=240,random_state=42,max_features='sqrt',subsample=0.9,
                                min_impurity_decrease=0.01)
                                
print(grd)
# Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.
# Choosing max_features < n_features leads to a reduction of variance and an increase in bias.降低过拟合，但是可能会增加偏差，降低方差（对应的过拟合）
pipe=make_pipeline(vect,tfidf,grd)
pipe.fit(X_train.cut_comment, y_train)
y_pred = pipe.predict(X_test.cut_comment)
metrics.accuracy_score(y_test,y_pred)
```

    GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.18, loss='deviance', max_depth=10,
                  max_features='sqrt', max_leaf_nodes=None,
                  min_impurity_decrease=0.01, min_impurity_split=None,
                  min_samples_leaf=1, min_samples_split=2,
                  min_weight_fraction_leaf=0.0, n_estimators=240,
                  presort='auto', random_state=42, subsample=0.9, verbose=0,
                  warm_start=False)
    




    0.96560509554140128




```python
metrics.confusion_matrix(y_test,y_pred)
```




    array([[265,  24],
           [  3, 493]], dtype=int64)



#### xgboost 方法


```python
from xgboost import XGBClassifier   
# sklearn API 类似于导入的从skearn中导入某个算法，然后再进行实例化即可，初始化算法的时候可以修改默认参数
from xgboost import plot_importance
x_train_vect=vect.fit_transform(X_train["cut_comment"])
x_test_vect= vect.transform(X_test["cut_comment"])
clf = XGBClassifier(
silent=1 ,#设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
# #nthread=4,# cpu 线程数 默认最大
learning_rate= 0.20, # 学习率
min_child_weight=0.5, 
# # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
# #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
# #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
gamma=0.1,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
subsample=0.7, # 随机采样训练样本 训练实例的子采样比
max_depth=15,
max_delta_step=0,#最大增量步长，我们允许每个树的权重估计。
colsample_bylevel=0.7, # Subsample ratio of columns for each split, in each level.
colsample_bytree=0.6, # 生成树时进行的列采样 
reg_lambda=0.04,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越保守
reg_alpha=0.05, # L1 正则项参数，参数越大，模型越保守
### 正则化是在梯度提升树种没有的，这是xgboost与GB方法的区别之一。
scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重=sum(负类样本)/sum(正类样本)
# objective= 'reg:logistic', #多分类的问题 指定学习任务和相应的学习目标
objective='binary:logistic' ,
# #num_class=10, # 类别数，多分类与 multisoftmax 并用
n_estimators=900, #树的个数
random_state=42
# #eval_metric= 'auc'
)
# xgb_model=XGBClassifier()
# clf = GridSearchCV(xgb_model, {'max_depth': [4, 6,8,10],
#                                'n_estimators': [50, 100, 200,400,600],
#                                'gamma':[0.1,0.12,0.15,0.18,0.2],
#                               'subsample':[0.5,0.6,0.7,0.8,0.9,1.0],
#                               'learning_rate':[0.1,0.15,0.2],
#                               'reg_lambda':[0.2,0.4,0.6,0.8]}, verbose=1,
#                               n_jobs=2)
clf.fit(x_train_vect,y_train,eval_metric=['auc','error'])
# clf.fit(x_train_vect,y_train,eval_metric=['auc','error'])
# print(clf.best_score_)
# print(clf.best_params_)


# #获取验证集合结果
# # evals_result = clf.evals_result()
# y_true, y_pred = y_test, clf.predict(x_test_vect)
# print("Accuracy : %d" % metrics.accuracy_score(y_true, y_pred))
```




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.7,
           colsample_bytree=0.6, gamma=0.1, learning_rate=0.2,
           max_delta_step=0, max_depth=15, min_child_weight=0.5, missing=None,
           n_estimators=900, n_jobs=1, nthread=None,
           objective='binary:logistic', random_state=42, reg_alpha=0.05,
           reg_lambda=0.04, scale_pos_weight=1, seed=None, silent=1,
           subsample=0.7)




```python
y_pred=clf.predict(x_test_vect)
metrics.accuracy_score(y_test,y_pred)
```




    0.94777070063694269




```python
metrics.confusion_matrix(y_test,y_pred)
```




    array([[260,  29],
           [ 12, 484]], dtype=int64)