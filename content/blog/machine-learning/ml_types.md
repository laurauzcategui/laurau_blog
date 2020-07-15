---
title: "Types of Machine Learning"
categories: ["machine-learning"]
tags: ["Machine Learning", "AI"]
date: 2020-07-02T16:32:58+01:00
draft: false
---
   ![](/img/ml_types_small.jpg)

It's been a while since I wrote my first post on [What is Machine Learning (ML) and How Programming paradigms have changed over time](https://towardsdatascience.com/on-the-journey-to-machine-learning-ai-6059ffa87d5f) and describing some use cases/applications. This time, I am sharing how Machine Learning and AI can be seen from different perspectives, specifically covering the following two areas:

1. How much human interaction is involved in the training process of different ML algorithms.
2. How the training is performed.
 
---
 
Before moving into each of these areas, let's clarify a few concepts around the Machine Learning Process. If you are familiar with how Machine Learning works, you can skip this section.
 
A high-level definition of Machine Learning can be seen as:

 - Given some *data* representative of an area ( sales, politics, education) you are analysing and an *algorithm*, the ability of a computer can *learn* from this data and detect certain *patterns* on it. Followed by being able to tell (*predict* ) you and determine the type of new data or at least an approximation of it. 
 
Take this concept with a pinch of salt, as there is too much involved in the background about how ML is actually performed.
 
In other words, the computer is able to detect *patterns* by *training* ( learning ) from the input data. This process is highly iterative and needs lots of tuning. For example: It needs to check how far or close the prediction is from the real value, then correct itself by adjusting its parameters until reaching a point where the model is certainly accurate enough to be used.
 
Ok, now that we have an overview of the process. Let's jump into types of Machine Learning.
 
## ML Algorithms and Human intervention.
 
Machine Learning systems on this area could be seen as the amount of ***"Supervision" a.k.a Human Interaction*** those will have over the training process. These are divided in 3 main categories, I will try to illustrate the following definitions with examples.
 
## 1. Supervised Learning
 
Imagine you are the owner of a local bookshop.

{{< figure src="/img/bookshop.jpg" height=200 width=200 class="imgs" >}}

 Your daughter Ana is a Data Scientist and she has offered to take books dataset recorded in your inventory system and implement a new system to speed up registration of new books arriving for sale. 

Ana knows some characteristics of the books:
- Genre ( Fiction, Non-Fiction, Fantasy, Thriller )
- Hardcover, if a books is available on Hardcover
- ISBN, the commercial id for the book.
- Title, Number of Pages
- Extract of the book or Cover picture
- Author
 
Now, Imagine having to allocate one book in a bookshelf. This is easy if you look up the metadata online and place the book on the Fiction shelf.
 
However, in reality, if you receive upto 200 books per day, you cannot do this job on your own, manually entering data on the system is error prone and you might end up putting books in the wrong shelf. This misplacement might end up in lower revenue as when a new customer comes incomes in and leaves unsatisfied because they cannot find the book even though it was actually on the store.

Ana's new system is as easy as feeding the system with an extract or the cover of the book you are looking for and can tell you which shelf this should be placed on.
 
This particular example is called: ***Classification***, because the system is just helping you to classify (organise) some data based on certain characteristics that you as user presented to it.
 
A real example of this kind of system is: **Google Photos**.
 
Yet another type of Supervised algorithm is about predicting a numeric value, given a set of features or characteristics called predictors.
 
Now, you might actually ask how this happens?
 
An example of this is using ***Regression***. A common and widely used algorithm known as Linear Regression and the purpose is to predict a continuous value as a result.
 
This actually translates to having a set of features or variables, and a set of labels that match this input features and all you want the algorithm to do is to learn how to *"fit"* the weights (*parameters*) associated with these features to give the approximate value that is closer to the real value.
 
## 2. Unsupervised Learning.

***Unsupervised*** algorithms cover all cases when we do not have a label or real value to compare against. Instead we do have sets of data, and the model will be trained and predicted based on how the data is ***grouped*** together, how it detects patterns and if it shows certain behaviours.
 
A typical example for this type of Machine Learning is ***Clustering***, where you group your data based on similarity. Real examples include Recommender Systems such as:
 
- Retailer websites, like Amazon and Zalando.
- Media / Streaming systems, like Netflix, Youtube among others.
 
There are lots of other algorithms that are used within this type, for example Anomaly detection which might cover credit card fraud or account takeover situations that can be prevented and predicted with the use of ML.
 
Yet another of my favourite types of unsupervised learning algorithms that I've discovered recently are those used for data visualization, like [t-SNE](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) or t-Distributed Stochastic Neighbor Embedding. 

I have used t-SNE in a couple of pet projects for *sentiment analysis visualization* and you could actually see the clusters forming within itself in high dimensional space using at the same time dimensionality reduction.
 
## 3. Reinforcement Learning.
 
Many books and websites refer to these algorithms like *"the beast"* but I like thinking about it about the top of the ice cream. Currently being a trending topic due to the achievements that are being accomplished in the area, which I will mention in a bit.
 
 
What would you do in this case? Most likely, you will play a couple of times, trying to investigate what are the best movements and the best route you can use to finally rescue the dogs from the shelter. This is similar to what a reinforcement algorithm will do, you can think of it as the following:
 
1. You are given an *environment*. (space in the context of the video game), representative of the state of certain variables on the space. Like roadblocks or crickets falling from the sky.
2. Actions to be performed during the event. ( movements, like going up, down, throw a ball to builders ), it's worth noticing that some actions could be ***rewarded*** as "good", and some others as "bad" meaning it will decrease your ability to achieve the goal.
3. Agents (*roles in the video game, rescuer or, builder*), the ones who will perform the action to achieve the goal at the minimum cost possible.
 
The way it works is the Agent observes the environment, tries to perform actions, and based on these actions will get rewarded for them or not. Potentially the Agents will learn on their own what is the best approach/strategy to pursue in order to get the maximum amount of points (rewards).
 
One of the best systems I've seen so far built with it are:
 
- **Hide and Seek**, developed by OpenAI. Where they use a multi-agent approach teaching those to play hide and seek. There are 2 agents "hiders" and "seekers" and by observing the environment they were able to learn behaviour/actions that weren't even provided, like using obstacles to pass barriers. If you are curious about how it works in depth you can watch the video below and the **[paper](https://arxiv.org/abs/1909.07528)**.
 
   {{< youtube kopoLzvh5jY >}}
   <br>
 
- **Video Games**, like StarCraf or AlphaStar. The agents play against humans and indeed it beat the best world player or beat other agents in a matter or couple of seconds. In order to achieve this, the group that developed AlphaStar didn't only use Reinforcement learning but others like using Supervised learning to train the neural networks that will create the instructions to play the game. If you are curious about how it done, you can check their blog post **[here](https://deepmind.com/blog/article/alphastar-mastering-real-time-strategy-game-starcraft-ii)**
 
   ![](/img/starcraft.gif)
 
We have learned by now different types of Machine Learning systems based on how little or much the human interaction with those systems is applied. Now we can go ahead and explore the next 2.
 
----
 
## ML Algorithms and Training process.
 
In previous sections, we outlined that the training process is how your algorithm will "learn" the best parameters to make a prediction. Having said that, training itself is an art and can be performed differently depending on the use case, resources available, and the data itself.
 
This is divided in two:
 
1. Batch Learning
2. Online Learning, also known as Incremental Learning.
 
Let's see how those work:
 
### 1. Batch Learning.

Batch learning is about training your model with all the data that is available at once rather than doing it incrementally. Usually performing this action takes a lot of time. Imagine having to train Terabytes if not bigger of data all at once. But why would you need to do that?
Due to the nature of the Business or use case, for example, reports or decisions that are delivered with a certain frequency like weekly/monthly might not need training on a real-time basis.
This type of training is usually done offline, as it takes a very long time (hours/days/weeks) to complete.
 
#### How does training occur?
 
1. The model is trained.
2. The model gets deployed and launched to Production.
3. The model is running continuously without further "learning".
 
All this process is called offline learning. Now you might wonder, what happens if my data or use case has changed? You need to train your model again against incorporating the new data or features.
 
A practical example, in my previous job models were deployed to production in a batch learning fashion and some models didn't need to be refreshed so often, so It could happen once every week or so. Yet another reason to generate a new model was by looking at metrics deployed in the monitoring system and checking if the performance of the model was being degraded.
 
These types of models are tightly related to resources such as IO,CPU, Memory or Network among others, so having this in mind before deciding on which approach to take is hugely important. Nowadays this might be really expensive but at the same time, you could take advantage of Cloud platforms offering solutions out of the box for doing this, such as: [AWS Batch ](https://docs.aws.amazon.com/batch/latest/userguide/what-is-batch.html),[Azure Machine Learning - Batch Prediction](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-run-batch-predictions-designer), or [Google Cloud AI Platform](https://cloud.google.com/ai-platform/training/docs). 

Moving on, let’s talk now about Online Learning

### Online Learning

Have you wondered how Netflix, Disney, Amazon Prime Video are recommending you what to watch? Or Why Zalando and Amazon keep telling you to buy these amazing trousers that might go with a  pair of white shoes? You are probably telling, yeah Recommender Systems, but more than that, how it can be done so quickly? How does the system adapt so quickly to change? 

You probably got it right again, this is done because Training occurs on the fly, meaning data is processed as it's arriving to the system. This approach is suitable for systems that are receiving data continuously, like retailers, or systems that need to adapt to changes quickly as well like a News website, or Stock market. 

In terms of how training is performed is as follows: 

1. Data is chunked into pieces or mini-batches.
2. Train the model ( continuously). 
3. Continuous evaluation / monitoring. 
4. Deployment to production.

One of the advantages of doing online learning is that if you don’t have enough computation resources or storage, you can discard the data as you train. Saving you then a huge deal of resources. On the other hand, if you need to replay the data you might want to store it for a certain amount of time. 

As with every system we deal with, this type of training has its strengths and weaknesses. One of the downsides of dealing with this approach is that Performance of the model might degrade quickly or eventually as the data might change quickly you might notice drops in prediction accuracy at some point in time. For this reason and -barely a topic I have seen talked about often- is having in place good monitoring systems. Those will help your team or company to prevent and understand when to start changing or tuning the models to make them effective.

__________

## Highlights

You have come this far and I hope you have enjoyed it. Initially, we talked about how machine learning works to then dive into how Machine Learning is divided according to the perspective of Humans interact with the Algorithms. Ultimately, I tried to describe as much as possible each of these types Supervised, Unsupervised, and Reinforcement Learning with cool examples and applications I have seen around.
I'll probably start posting more practical things that I've learned on the journey and would love to share. #SharingIsCaring.

Stay well and see you soon. 

