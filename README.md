### 이독성 판별을 위한 사전학습 언어모델

**Hierachical Language Information-BERT(HLI-BERT) + Lingustic Feature**



![image](https://github.com/Jungmin-YUN-0/Readability_linguistic_feature/assets/76892989/fe45dd13-a5dd-4398-87bf-dd8ea26652e6)


**#1 extract_linguistic_feature.ipynb**  
linguistic feature 추출 (linfeat)

**#2 main.py**  
main.py is for model training and inference.


Example command lines:

```Python
python main.py -data [data_path] -cls [svm / gru] -lingf [TRUE / FALSE]
```

**Arguments are as follows:**

* data: directory of dataset
* cls: You can choose between ```svm``` and ```gru``` for classification.
* lingf: whether to apply linguistic feature








