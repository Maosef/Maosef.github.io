---
# layout: page
title: Portfolio
permalink: /portfolio/
---


## Things I've done


# Text classifier
Classify sentences with malware attributes. Built with Pandas and scikit-learn.
![performance](/images/text_classify_perf.PNG)

**More details**: I wrote an [article][text classify] on this. 

[text classify]: https://maosef.github.io/nlp/2019/08/22/text-classify.html

# Co-occurence network
Visualization of named-entity co-occurences extracted from documents. Nodes are words, edges are weighted by frequency. Built with spaCy, networkx, and Plotly.
![Conet](/images/conet2.png)

**More details**: This wasn't new or difficult to do, but it looked cool. I was trying to build a pipeline for automated knowledge base construction... still getting there. One extension would be to run community detection algorithms on it.


<!-- # Trading Algorithm Pipeline
Made with the Smith Investment Fund. Performs data ingestion, fundamental factor calculations, and backtesting. 

![trading performance](/images/trading_perf.PNG) -->


# FINRA Advisor network
Visualize financial advisors and their connections via concurrent employment. Made during Bitcamp 2019.
![FINRA network](/images/finra_network.PNG)
[Link to the code][finra-network]

**More details**: My first experiment with network visualization. Network theory is just a prefect combination of cool and pretty. Being able to animate this over time is the next big step.

<!-- # Bookshelf
Sees bookshelf, recommends book titles. Made during PennApps.
[Link to the code][bookshelf] -->

# Document topic visualizer
Ingest a bunch of text, run topic modelling algorithms on it, visualize the results. Built with Gensim and Plotly.
![Topic model](/images/topic_model.PNG)
**More details**: the topic model algorithm I used is called **Latent Dirichlet Allocation (LDA)**, which is really cool. Graphical models and probability just pop up everywhere. There are a bunch of extensions to it, which would be fun to try and implement later.

# Spatio-temporal action detection
Classify actions in a video. Localize it over space and time. Built with Tensorflow and OpenCV.
![Action detection](/images/action_detection.png)
**More details**: this sad image doesn't do justice to how much work goes into action detection. There's the space part, which uses cool object detection algorithms. Then there's the time part, which is weird and nobody really knows how to use it. It gets even crazier when you want real-time, which goes into tracking algorithms. I need to find and upload a video clip later.

# Quadratic Sieve
Implemented a fast number factorization algorithm. [Link to the code][quadratic-sieve]

**More details**: this was one of my first big projects. There's terribly inefficient parts of it (it's written in pure Python), but it was a fun experience. Also, number theory is absolutely wack.

[finra-network]: https://github.com/Maosef/Bitcamp_2019
[bookshelf]: https://github.com/Maosef/PennApps_XIX
[quadratic-sieve]: https://github.com/Maosef/Quadratic-Sieve
