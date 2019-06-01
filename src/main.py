import fool
import pandas as pd
import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  
from pyltp import Segmentor
from pyltp import Parser
from pyltp import Postagger
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np


'''
实体统一
对同一实体具有多个名称的情况进行统一，将多种称谓统一到一个实体上，
比如：杭州阿里巴巴集团和阿里巴巴是一个实体
'''
 
def main_extract(company_name,d_city_province,suffixs,scopes):
    """输入公司名，返回会被统一的简称
    分析公司全称的组成方式，将“地名”、“公司主体部分”、“公司后缀”区分开，并制定一些启发式的规则
    一个公司名可能有0-n个地名+0-n个字符+0-1个经营范围+0-n个后缀
    简称的组成
    1.有中间字符，直接取中间字符
    2.没有中间字符，且地名和经营范围有一个也不存在，则取整个公司名，否则取地名和经营范围的组合
    Args：
        company_name  输入的公司名
        d_city_province 地区
        suffixs 后缀
        scopes 经营范围
    Returns:
        公司的简称
    """
    
    place = ''
    scope = ''
    middle = ''
    suffix = ''

    # 分词
    name_list = jieba.lcut(company_name.replace(' ', ''))
    
    # 从前面分离出地名
    for word in name_list:
        if word not in d_city_province:
            break
        else:
            place += word
            name_list.remove(word)
    
    # 去除强后缀
    if len(name_list):
        word = name_list.pop()
        if word not in suffixs:
            name_list.append(word)
    
    findscope = False
    # 从后面开始遍历直到遇到经营范围
    while len(name_list):
        word = name_list.pop()
        # 如果是经营范围，则保留且结束遍历
        if word in scopes:
            findscope = True
            # 如果是股份直接丢弃
            if word == '股份':
                continue
            # 有多个经营范围时取最前面的，并把之前选到的给后缀部分
            suffix = scope + suffix
            scope = word
        elif findscope:
            middle = word
            break
        else:
            suffix = word + suffix

    # 剩余部分为中间字段
    middle = ''.join(name_list) + middle
    
    if len(middle) != 0:
        result = middle
        if scope == '银行':
            result += scope
    elif len(place) == 0 or len(scope) == 0:
        result = place + scope + suffix
    else:
        result = place + scope
    '''
    # 如果前三个有任意一个长度为0，
    if len(place) == 0 or len(middle) == 0 or len(scope) == 0:
        # 如果middle和scope都为0.则需要加上后缀
        if len(middle) == 0 and len(scope) == 0:
            result = place + suffix
        else:
            result = place + middle + scope
    else:
        result = place + middle
    '''
    
    return result


# 打开有关公司名称前后缀和经营范围的文件
with open('../data/dict/stopwords.txt', encoding='utf8') as f:
    stop_word = f.read().split()
with open('../data/dict/company_business_scope.txt', encoding='utf8') as f:
    scopes = f.read().split()
with open('../data/dict/co_City_Dim.txt', encoding='utf8') as f:
    citys = f.read().split()
with open('../data/dict/co_Province_Dim.txt', encoding='utf8') as f:
    provinces = f.read().split()   
with open('../data/dict/company_suffix.txt', encoding='utf8') as f:
    suffixs = f.read().split()  
    suffixs = suffixs[1:]
    
citys.extend(provinces)
d_city_province = citys


'''
实体识别
'''
sample_data = pd.read_csv('../data/info_extract/train_data.csv', encoding = 'utf-8', header=0)
y = sample_data.loc[:,['tag']]
train_num = len(sample_data)
test_data = pd.read_csv('../data/info_extract/test_data.csv', encoding = 'utf-8', header=0)
sample_data = pd.concat([sample_data.loc[:, ['id', 'sentence']], test_data])
sample_data['ner'] = None
# 将提取的实体以合适的方式在‘ner’列中并统一编号，便于后续使用
# 将句子中的实体改为n号企业
# 维护n号企业与实际企业名的字典
index = 0
alias2company_dict = {}
company_index = 1
company2alias_dict = {}
f_user_dict = open('../data/user_dict.txt', 'w', encoding='utf8')
alias = ''
for i, row in sample_data.iterrows():
    sentence = row['sentence']
    tmp, ner__ = fool.analysis(sentence)
    ner_ = [list(entity) for entity in ner__[0]]
    for entity in ner_:
        if entity[2] == 'company':
            company_ = main_extract(entity[3],d_city_province,suffixs,scopes)
            if company_ not in company2alias_dict:
                alias = '%d号企业'%company_index
                company2alias_dict[company_] = alias
                alias2company_dict[alias] = company_
                company_index += 1
                f_user_dict.write(alias+'\n')
            # 先将句子中的实体名称进行替换
            alias = company2alias_dict[company_]
            # sentence = sentence.replace(entity[3], alias)
            entity[3] = company_
    
    # 测试发现ltp的用户自定义词典效果很差，并不是用户提供的词典优先
    # sample_data.iat[index, 1] = sentence
    sample_data.iat[index, 2] = ner_
    index += 1     
    
f_user_dict.close()


'''
关系抽取
目标：借助句法分析工具，和实体识别的结果，以及文本特征，基于训练数据抽取关系。

抽取股权交易关系，选出投资方，被投资方。
'''

# 提取文本tf-idf特征
# 去除停用词，并转换成tfidf向量。

def get_tfidf_feature():
    '''获取tfidf特征
    先对每个样本进行分句，并且分句结果中的公司名称全部换成前一步中的公司简名
    Returns：
        整个样本的tfidf结果
    '''
    segmentor = Segmentor()  # 初始化实例
    segmentor.load_with_lexicon('e:/ltp_data_v3.4.0/cws.model', '../data/user_dict.txt')  # 加载模型
    text = []
    for i, row in sample_data.iterrows():
        words = []
        sentence = row['sentence']
        start = 0
        end = 0
        for entity in row['ner']:
            end = entity[0]
            words.extend(segmentor.segment(sentence[start:end]))
            words.append(entity[3])
            start = entity[1] - 1
        if end < len(sentence):
            words.extend(segmentor.segment(sentence[start:len(sentence)]))
            
        text.append(' '.join(words))

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(text)
    
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf_feature = transformer.fit_transform(X.toarray())
    
    segmentor.release() 
    
    return tfidf_feature


tfidf_feature = get_tfidf_feature()
tfidf_feature = pd.DataFrame(tfidf_feature.toarray())

# 提取句法特征

# 投资关系关键词
# 可以结合投资关系的触发词建立有效特征
key_words = ["收购","竞拍","转让","扩张","并购","注资","整合","并入","竞购","竞买","支付","收购价","收购价格","承购","购得","购进",
             "购入","买进","买入","赎买","购销","议购","函购","函售","抛售","售卖","销售","转售"]

key_words_dict = {}
# 用来记录未知关键字
key_words_dict2 = {'index':len(key_words)}
for i,value in enumerate(key_words):
    key_words_dict[value] = i

postagger = Postagger() # 初始化实例
postagger.load_with_lexicon('e:/ltp_data_v3.4.0/pos.model', '../data/user_dict.txt')  # 加载模型
segmentor = Segmentor()  # 初始化实例
segmentor.load_with_lexicon('e:/ltp_data_v3.4.0/cws.model', '../data/user_dict.txt')  # 加载模型
parser = Parser() # 初始化实例
parser.load('e:/ltp_data_v3.4.0/parser.model')  # 加载模型
# labeller = SementicRoleLabeller()
# labeller.load('e:/ltp_data_v3.4.0/pisrl.model')

def get_parse_feature(s):
    """句法特征提取
    对语句进行句法分析，并返回句法结果，
    提取的特征为
        0-company个数，1-两个company的距离，2-两个company的句法距离，3~4-两个company与触发词之间的距离
        5~8-company的前后词依存句法，9~10-依存句法,11-关键词 
    Args：
        待分析句法特征的一行，
    Returns：
        分析结果
    """
    features = [0] * 12
    sentence = s['sentence']
    ner_list = s['ner']
    company_set = set()
    
    features[5:11] = 'N'*6
    if (len(sentence)>2000):
        return features, ('', '')
    
    # 统计company个数，少于2个不做后面的分析，计算x0
    company_list = []
    for entity in ner_list:
        if entity[2] == 'company':
            if entity[3] not in company_set:
                company_set.add(entity[3])
            company_list.append([entity[3], entity[0], entity[1]])
            
    features[0] = len(company_set)        
    if features[0] < 2:
        return features, ('', '')
    
    # 按照之前的实体识别进行再一次的分词，
    words = []
    start = 0
    end = 0
    current_company = 0
    for entity in ner_list:
        end = entity[0]
        words.extend(segmentor.segment(sentence[start:end]))
        words.append(entity[3])
        start = entity[1] - 1
        # 记录当前词的位置
        if entity[2] == 'company':
            company_list[current_company].insert(0, len(words)-1)
            current_company += 1
    if end < len(sentence):
        words.extend(segmentor.segment(sentence[start:len(sentence)]))

    # 根据分词结果做词性标注
    pos = postagger.postag(words)
    # 异常情况，这组数据作为无效值处理
    if (len(pos) == 0):
        features[0] = 0
        return features, ('', '')
    # 做句法分析
    parse_result = parser.parse(words, pos)   
    # roles = labeller.label(words, pos, parse_result)
    # 记录下一个company数组，
    # 每个成员变量带有0当前分词位置，1实体名，2起点，3终点    
    trigger_index = 0
    index = 0
    features[11] = -1
    for word in words:
        # 找核心词，即被标记为HED的词，但不一定是触发词，
        if parse_result[index].relation == 'HED':
            trigger_index = index
        # 查看关键字是否在word中，如果在则记录到
        if word in key_words_dict:
            features[11] = key_words_dict[word]
        index += 1
    
    # 如果触发词不在当前key_words_dict中,查看其是否在2表中，还不在则加一个新的
    if features[11] == -1:
        if words[trigger_index] in key_words_dict2:
            features[11] = key_words_dict2[words[trigger_index]]
        else:
            current_index = key_words_dict2['index']
            features[11] = current_index
            key_words_dict2[words[trigger_index]] = current_index
            key_words_dict2['index'] = current_index + 1
    
    # 查找company位置，优先句法类型SBV或者VOB关系的，其次是有关系的，其次选取较近的，遍历company数组，
    # sbv关系加50分，位置距离减距离分，有关系加50分
    # 有多个时，选取最近的两个
    for company in company_list:
        match = 0
        if parse_result[company[0]].relation == "SBV" or parse_result[company[0]].relation == "VOB":
            match += 50
        if parse_result[company[0]].head == trigger_index:
            match += 50
        match = match - abs(company[0] - trigger_index)
        company.append(match)
        
    company_list.sort(key=lambda company : company[4], reverse=True)
    company1 = company_list[0]
    # company2要求与company1的名称不同
    for i in range(1,len(company_list)):
        if company_list[i][1] == company1[1]:
            continue
        else:
            company2 = company_list[i]
            break
    
    # 计算x1
    features[1] = min(abs(company1[2] - company2[3]), abs(company1[3] - company2[2]))
    
    # 计算x3和x4
    features[3] = company1[0] - trigger_index
    features[4] = company2[0] - trigger_index
    
    # 计算x2
    features[2] = company1[0] - company2[0]
    
    # 计算x5-x8
    if company1[0]-1 >= 0:
        features[5] = parse_result[company1[0]-1].relation
    if company1[0]+1 < len(words):
        features[6] = parse_result[company1[0]+1].relation
    if company2[0]-1 >= 0:
        features[7] = parse_result[company2[0]-1].relation
    if company2[0]+1 < len(words):
        features[8] = parse_result[company2[0]+1].relation   
    
    # 计算x9-x10，该依存句法可能不是与触发词的关系
    features[9] = parse_result[company1[0]].relation
    features[10] = parse_result[company2[0]].relation
    

    return features,(company1[1], company2[1])


# 遍历sample_data,生成二维数组
parse_feature = []
company_pairs = []
for i, row in sample_data.iterrows():
    feature, company_pair = get_parse_feature(row)
    parse_feature.append(feature)
    company_pairs.append(company_pair)


postagger.release() 
segmentor.release() 
parser.release() 

parse_feature = pd.DataFrame(parse_feature)

# 汇总词频特征和句法特征
# 将字符型变量转换为onehot形式
# 拼接前转化为onehot
encoder = LabelEncoder()  
enc = OneHotEncoder()
for i in range(5,11):
    to_place = encoder.fit_transform(parse_feature.iloc[:,i].values)
    parse_feature = parse_feature.drop(i, axis=1,inplace=False)
    parse_feature.insert(i, i, to_place)
    
    
whole_feature = pd.concat([parse_feature, tfidf_feature], axis=1)
train_x = whole_feature.iloc[:train_num]
test_x = whole_feature.iloc[train_num:]


# 建立分类器进行分类，使用sklearn中的分类器，不限于LR、SVM、决策树等
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import classification_report
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score


class RF:
    def __init__(self):      
        self.scaler = preprocessing.StandardScaler()
        
    def black_box_function(self, n_estimators_, max_depth_, random_state_):
        clf = RandomForestClassifier(n_estimators=int(n_estimators_), 
                                     max_depth=int(max_depth_), 
                                     random_state=int(random_state_))
        score = cross_val_score(clf, self.train_x_scale, self.train_y, cv=5, scoring='f1_micro')

        return np.mean(score)

    def train(self, train_x, train_y):
        self.scaler.fit(train_x)
        self.train_x_scale = self.scaler.transform(train_x)
        self.train_y = train_y

        optimizer = BayesianOptimization(
            self.black_box_function,
            {'n_estimators_': (1190, 1190),
             'max_depth_': (29, 29),
             'random_state_': (5, 5)})

        optimizer.maximize(
            init_points=3,
            n_iter=2,
        )
        
        print(optimizer.max)
        model = RandomForestClassifier(n_estimators=int(optimizer.max['params']['n_estimators_']),
                                     max_depth=int(optimizer.max['params']['max_depth_']),
                                     random_state=int(optimizer.max['params']['random_state_']))
        '''
        model = RandomForestClassifier(n_estimators=1190,
                                     max_depth=29,
                                     random_state=5)
        '''
        model.fit(self.train_x_scale, self.train_y)
        return model
        
    def predict(self, clf, test_x):
        test_x_scale = self.scaler.transform(test_x)
        predict = clf.predict(test_x_scale)
        predict_prob = clf.predict_proba(test_x_scale)
        
        return predict, predict_prob
    
    def report(self, clf):
        print(classification_report(self.train_y, clf.predict(self.train_x_scale)))

 
rf = RF()
model = rf.train(train_x.values, y.values.ravel())
rf.report(model)

# 预测
predict, predict_prob = rf.predict(model, test_x.values)

# 存储提取的投资关系实体对，本次关系抽取不要求确定投资方和被投资方，仅确定实体对具有投资关系即可
"""
以如下形式存储，转为dataframe后写入csv文件：
[
    [九州通,江中药业股份有限公司],
    ...
]
"""
company_pair_df = pd.DataFrame(columns=['company1', 'company2'])
index = 0

for i in range(len(y)):
    if y.iloc[i][0] == 1 and company_pairs[i][0] != '':
        company_pair_df.loc[index] = company_pairs[i]
        index += 1
        

for i in range(len(predict)):
    if predict[i] == 1 and company_pairs[i + 7000][0] != '':
        company_pair_df.loc[index] = company_pairs[i + 7000]
        index += 1

print(company_pair_df)
company_pair_df.to_csv('../submit/company_pair.csv', sep=',', header=True, index=True)

# 存储进图数据库
from py2neo import Node, Relationship, Graph

graph = Graph(
    "http://localhost:7474", 
    username="neo4j", 
    password="123456"
)

company_pair_df = pd.read_csv('company_pair.csv', encoding='utf-8')

node_dict = {}

for i,row in company_pair_df.iterrows():
    if row['company1'] not in node_dict:
        node_dict[row['company1']] = Node('Company', name=row['company1'])
    if row['company2'] not in node_dict:
        node_dict[row['company2']] = Node('Company', name=row['company2'])
    
    # 本次不区分投资方和被投资方
    r = Relationship(node_dict[row['company1']], 'INVEST', node_dict[row['company2']])

    s = node_dict[row['company1']] | node_dict[row['company2']] | r
    graph.create(s)
    #r = Relationship(node_dict[row['company2']], 'INVEST', node_dict[row['company1']])
    #s = node_dict[row['company1']] | node_dict[row['company2']] | r
    #graph.create(s) 