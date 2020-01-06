---
title:  "Text classification"
date:   2019-08-21 22:17:56 -0400
categories: nlp
---

This is a walkthrough of my attempts to run text classification at my work.

# Background
Recently, people have been trying to organize and classify cyber threat data. There's this thing called the **MAEC (Malware Attribute Enumeration and Characterization)** vocabulary, which is basically a way to talk about malware which computers can understand (think XML).

# The problem
There's a lot of papers on cyber data. We want to read through them faster, so we want to use computers to help understand them.

# The data
[SemEval 2018, Task 8][SecureNLP]. The authors annotate sentences from cybersecurity reports with tags of varying levels of complexity. The hardest ones are malware attributes from - you guessed it - MAEC!

[SecureNLP]: https://www.aclweb.org/anthology/S18-1113
<!-- ![sample annotation](/images/securenlp_demo.png) -->
<img src="/images/securenlp_demo.PNG" alt="drawing" width="500"/>


# The goal
Build a model that classifies these attributes from the text.

The straightforward approach is to classify sentences. But there's a catch - attributes annotations are attached to the "action word" of a sentence which may sometimes have multiple. I thought the easiest way was just to throw away those sentences.

I simplified the task to just predicting Capabilities, which are the highest tier of attributes.

# Data analysis

The most annoying part is the data preprocessing, which I'll skip. Let's examine the class labels and frequencies.

<!-- ![classes](/images/securenlp_classes.png) -->
<!-- ![class_freqs](/images/securenlp_class_distro.png) -->
<img src="/images/securenlp_classes.PNG" alt="drawing" width="500"/>
<img src="/images/securenlp_class_distro.PNG" alt="drawing" width="500"/>


Well there's our first problem. Not only is the dataset small, but its highly skewed. Either way, let's proceed with a simple baseline - bag-of-words with a linear support vector machine, trained using stochastic gradient descent. Let's evaluate it using a support-weighted F1 score, which will give the higher frequency classes more representation.

```python
def run_pipeline(train_data, train_labels, test_data, test_labels, model):
    
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1))), ('tfidf', TfidfTransformer()),
                             ('clf', model)])
    
    text_clf = text_clf.fit(train_data, train_labels)
    predicted = text_clf.predict(test_data)

    p,r,f = precision_score(test_labels, predicted, average='weighted'), recall_score(test_labels, predicted, average='weighted'), f1_score(test_labels, predicted, average='weighted')
    print(p,r,f)

model = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5)
run_pipeline_tune(X_train, y_train, X_test, y_test, model)
```

Our result? 0.594. Not so bad, but there's evidently room for improvement. We can visualize the learning curve of the model to get a sense of its behavior.

<img src="/images/text_classify_perf.PNG" alt="drawing" width="500"/>

The gap between the training and validation curves is a classic sign of overfitting. So how can we improve it?

Overfitting is typically addressed in 4 ways:
1. Train on more data
2. Simplify/regularize model
3. Reduce feature dimensions
4. Feature engineering

3 is interesting, because there's a super popular idea for reducing the input dimensions, called word embeddings.

To be continued...