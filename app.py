# @author: Xinxin Tang
# email: xinxin.tang92@gmail.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from flask import Flask, request
app = Flask(__name__)

import json
import tensorflow as tf
import TF_iris_data as iris_data
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri


def py_model():
    '''DNN model 2 layers with 10 neurons'''
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # The DNN Model
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=3,
        model_dir='./py_model')

    return classifier, train_x, train_y, test_x, test_y


def py_train():
    # Train the model.
    classifier, train_x, train_y, test_x, test_y = py_model()
    classifier.train(
        input_fn=lambda:iris_data.train_input_fn(train_x, train_y, 100), steps=1000)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:iris_data.eval_input_fn(test_x, test_y, 100))
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


def py_pred(predict):
    classifier, _, _, _, _ = py_model()
    predictions = classifier.predict(
            input_fn=lambda:iris_data.eval_input_fn(predict,labels=None,batch_size=1))

    res = []
    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        pre = "Prediction is " + iris_data.SPECIES[class_id]
        res.append(pre)
        # print(pred_dict)
    return res


def prediction(test_data):
    robjects.r('''predict.dnn <- function(model, data = X.test) {
                new.data <- data.matrix(data)
                hidden.layer <- sweep(new.data %*% model$W1 ,2, model$b1, '+')
                hidden.layer <- pmax(hidden.layer, 0)
                score <- sweep(hidden.layer %*% model$W2, 2, model$b2, '+')
                score.exp <- exp(score)
                probs <-sweep(score.exp, 1, rowSums(score.exp), '/') 
                labels.predicted <- max.col(probs)
                return(labels.predicted)
         }
               g <- function(test) {
                        load("./iris.rda")
                        labels.dnn <- predict.dnn(ir.model,test)
                        }''')

    test = pd.DataFrame(test_data)
    test = pandas2ri.py2ri(test)
    r_g = robjects.globalenv['g']
    pred = (r_g(test))
    expected = ['Setosa', 'Versicolor', 'Virginica']
    num = [1, 2, 3]
    dic = dict(zip(num, expected))
    pred_list = ["Prediction is " + dic[x] for x in pred]
    return pred_list


@app.route("/tensorflow-iris/predict", methods=['POST'])
def tf_iris():
    data = json.loads(request.data)
    res = py_pred(data)
    return json.dumps(res)


@app.route("/r-iris/predict", methods=['POST'])
def r_iris():
    data = json.loads(request.data)
    p = prediction(data)
    return json.dumps(p)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(debug=True, host='0.0.0.0')

