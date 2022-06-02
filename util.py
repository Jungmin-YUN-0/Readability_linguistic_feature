# Linguistic_feature 
import pandas as pd
import numpy as np

# SVM
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc

# GRU
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Dropout
#from keras.layers.embeddings import Embedding
from tensorflow.keras.layers import Embedding
from tensorflow import keras

#============================================================================================#

def Linguistic_feature():    
    # for training dataset   
    train_lf = pd.read_csv(r"./onestopeng/linguistic_feature/train_lf.csv")
    # for test dataset
    test_lf = pd.read_csv(r"./onestopeng/linguistic_feature/test_lf.csv")

    cols = ['0','1','2','3','4','5','6','7','9','10'] # cols = ['8']는 우선 제외.. #7:84
    
    train_lf['lf']=train_lf[cols].apply(lambda row: ':'.join(row.values.astype(str)), axis=1)
    train_lf['lf'].replace('\[','', regex=True, inplace=True)
    train_lf['lf'].replace('\]','', regex=True, inplace=True)
    train_lf['lf'].replace(r'^:\w','', regex=True, inplace=True) #train_lf['lf'].replace(r'^:','', regex=True, inplace=True)

    test_lf['lf']=test_lf[cols].apply(lambda row: ':'.join(row.values.astype(str)), axis=1)
    test_lf['lf'].replace('\[','', regex=True, inplace=True)
    test_lf['lf'].replace('\]','', regex=True, inplace=True)
    test_lf['lf'].replace(r'^:\w','', regex=True, inplace=True)

    def list_split(df):
        df = df.str.split(', ')
        df_lst = df.tolist()
        for i in range(0,len(df_lst)):
            for j in range(0, len(df_lst[i])):
                df_lst[i][j]=str(df_lst[i][j]).split(':')[0]
        df_lst_np = np.array(df_lst, dtype=np.float64)
        #print("linguistic feature (training dataset) : ", df_lst_np.shape)
        return df_lst_np

    train_lf_np = list_split(train_lf['lf'])
    test_lf_np = list_split(test_lf['lf'])
    
    return train_lf_np, test_lf_np

    '''
    train_lf['lf'] = train_lf['lf'].str.split(', ')
    train_lf_lst = train_lf['lf'].tolist()
    for i in range(0,len(train_lf_lst)):
        for j in range(0, len(train_lf_lst[i])):
            train_lf_lst[i][j]=str(train_lf_lst[i][j]).split(':')[0]
    train_lf_np = np.array(train_lf_lst, dtype=np.float64)
    print("linguistic feature (training dataset) : ", train_lf_np.shape)

    test_lf['lf'] = test_lf['lf'].str.split(', ')
    test_lf_lst = test_lf['lf'].tolist()
    for i in range(0,len(test_lf_lst)):
        for j in range(0, len(test_lf_lst[i])):
            test_lf_lst[i][j]=str(test_lf_lst[i][j]).split(':')[0]
    test_lf_np = np.array(test_lf_lst, dtype=np.float64)
    print("linguistic feature (test dataset) : ", test_lf_np.shape)    
    '''
#============================================================================================#  

def SVM(train_feature, train_label, test_feature, test_label):
    # SVM model
    SVMmodel = SVC(kernel ='linear', C = 0.5)

    # Training with SVM
    SVMmodel.fit(train_feature, train_label)

    # Evaluation with SVM
    y_pred_bert_words_svm = SVMmodel.predict(test_feature)
    y_prob_bert_words_svm = SVMmodel.decision_function(test_feature)

    def printResult(y_pred, y_prob):
        acc = accuracy_score(test_label, y_pred)
        # Result
        print("Accuracy: {:.4f}".format(acc*100),end='\n\n')
        cm = confusion_matrix(test_label,y_pred)
        print('Confusion Matrix:\n', cm)
        print(classification_report(test_label,y_pred))

    #     # Plot
    #     fpr, tpr, thresholds = roc_curve(test_label, y_prob)
    #     roc_auc = auc(fpr, tpr)
    #     print ("Area under the ROC curve : %f" % roc_auc)
    #     plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver operating characteristic')
    #     plt.plot(fpr, tpr, color='red', label='ROC curve (area = %0.2f)' % roc_auc)
    #     plt.legend(loc='lower right')

    # Result
    printResult(y_pred_bert_words_svm, y_prob_bert_words_svm)

#============================================================================================#

def GRU(train_feature, train_label):
    EMBEDDING_DIM = 32
    vocab_size = 30522
    max_length=768 #128

    model_gru = Sequential()
    #model_gru.add(keras.Input(shape=(510, 768)))
    ##model_gru.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length))
    model_gru.add(GRU(units=256,input_shape = (510,768),recurrent_dropout=0.2))
    #model_gru.add(GRU(units=1024,dropout=0.2,input_shape = (510,768),recurrent_dropout=0.2)))
    model_gru.add(Dropout(0.2))
    model_gru.add(Dense(64, activation='relu'))
    model_gru.add(Dense(1, activation='sigmoid'))

    '''
    model = keras.Sequential()
    model.add(keras.Input(shape=(150,150,3)))
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    '''

    # try using different optimizers and different optimizer configs
    model_gru.compile(loss='mse', optimizer='adam', metrics=['accuracy']) #, loss='mse' loss='binary_crossentropy

    print(model_gru.summary())

    model_gru.fit(train_feature, train_label, batch_size=128, epochs=10, validation_split=0.2, verbose=2)
