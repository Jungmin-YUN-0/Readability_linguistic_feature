import dill
import numpy as np
from util import SVM, Linguistic_feature

class Classification():
    def __init__(self, data, classification, linguistic_feature):
        DATA = data
        CLASSIFICATION = classification
        LINGUISTIC_FEATURE = linguistic_feature

        data = dill.load(open(DATA, 'rb'))

        bert_word_training_features = np.asarray(data['features']['f_train'])
        bert_word_test_features = np.asarray(data['features']['f_test'])

        train_label = data['labels']['l_train']
        test_label = data['labels']['l_test']

    # print(bert_word_training_features.shape, bert_word_test_features.shape, len(train_label), len(test_label))
    # (510, 768) (57, 768) 510 57
        if LINGUISTIC_FEATURE == 'FALSE':
            ##
            if CLASSIFICATION == "svm":
                SVM(bert_word_training_features, train_label, bert_word_test_features, test_label)
            #elif CLASSIFICATION == 'gru':
                #pass
                
        elif LINGUISTIC_FEATURE == 'TRUE':
            train_lf_np, test_lf_np = Linguistic_feature()
            
            bert_word_training_lf = np.concatenate([bert_word_training_features, train_lf_np], axis=1)
            #print("bert_word embedding with linguistic feature (training dataset): ", bert_word_training_lf.shape)
            bert_word_test_lf = np.concatenate([bert_word_test_features, test_lf_np], axis=1)
            #print("bert_word embedding with linguistic feature (test dataset): ", bert_word_test_lf.shape)

            if CLASSIFICATION == "svm":
                SVM(bert_word_training_lf, train_label, bert_word_test_lf, test_label)
        
