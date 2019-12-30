---
title: "On the Journey to Machine Learning / AI"
date: 2019-12-30T20:34:53Z
draft: false
---

# On the Journey to Machine Learning / AI

This is my first blog on Machine Learning (ML) and my journey through it. I’ve been quite interested in all things data from the very beginning of my career as a Software Engineer.

If you are interested in following baby steps on the journey to ML, its fundamentals and how you could start building projects with it, tag along and I’ll do my best to explain what I’ve learned so far.

I knew the future will be all about Data from the moment I started reading and learning more about Big Data, Data pipelines, Data Science, Artificial Intelligence.

Certainly, I’m more convinced now when observing how that we ( humans ) are continuously massive producers of data and I cannot resist digging further about how, when, where we could take advantage of this new oil and get the most out of it.

*Disclaimer*: the purpose of this post is to share the knowledge I’ve been getting meanwhile I study and work with Machine Learning. I’ll try to make references as much as I can to resources so you could take a look too if you wish to expand further.

The following post is composed in 3 sections, what is Machine Learning and how is it changing paradigms from a Software Engineering perspective, to finalize then on why everyone is so interested or not in getting it into their lives.

---

## What is Machine Learning ? 

There are millions of definitions about Machine Learning everywhere, I’ll point out here my favorite ones.

> The analogy with the human brain is used as a guiding principle ... The investigation mainly centres round an analogous teaching process applied to machines. [Turing, A. (1948). Intelling Machinery, 1948.](https://storage.googleapis.com/laurau_blog/intelligent_machinery_turing.pdf)

> A field of study that gives the computer(s) the ability to learn, without being explicitly programmed. 
>
> **Arthur Samuel, 1959**

### How do I define it?

Machine Learning is an area of science that helps you to detect patterns at a faster speed than we humans could do with the help of course of computers. Imagine it like applying all your math knowledge to data and applying techniques to make possible to take those patterns and get answers out of the data you are giving as an Input to algorithms.

## How the programming paradigm is shifting with ML ? 

If you are a Software Engineer, the following explanation might resonate with you.

Imagine you are building a smartwatch application that will help you to detect the type of workout you are doing at the gym, and as of now and for the purpose of this example your application only detects when you are using the treadmill or the stair stepper.

Great! You have built-in some rules in the backend of the application to detect if it’s using any of those machines and calculate the calories, as observed below:

```python
max_speed_on_treadmill = 19 # 19 km/h
max_speed_on_climber = 126 # 126 per minute

if inclination > 2 and speed > 0 and speed < max_speed_on_treadmill: 
    running_on_treadmill = True
    calculate_calories()
elif steps_per_minute < max_speed_on_climber: 
    using_stair_stepper = True
    calculate_calories()
```

But, here is the catch:

- What would happen in 1 year when 5 more machines with different brands come into the market and the owner of the smartwatch start using one of these machines you haven’t added yet?
- What would happen when a Gym owner wants you to add all of his machines to your app, and he owns lots of brands and there are approx +20 machines?
On top of these new restrictions, you also should account for the person who is doing the exercise, weight, heart rate…etc.

Ok, this seems like a lot of rules to code in the application. This paradigm is still used in many applications and still applies to the common use of programming, as explained below:

>### Traditional Programming Paradigm
> #### Rules and Data are the Input and We get Answers as the output :) 
> **Example**: You used today the treadmill for 15 minutes and your calories burned were 200 :)

*What if I told you that you could turn this around and let the computer figure it out the rules.*

>### New Programming Paradigm
> #### Data and Answers are fed into Algorithms and the outcome are the rules
> **Example**: You let the algorithm know what are the features for using: 
> - Machine A ( velocity, speed, inclination) and its values and you said: “This is what a Treadmill use looks like”.
> - Machine B ( steps per minute, velocity, program ) and its values and you said: “This is what a Stair Stepper use looks like”. </br></br>
>And so on and so forth, you feed the algorithm with lots of examples. So next time the smartwatch owner uses a new machine your smartwatch will be able to infer is using a leg press and will calculate everything on its own.

How great is this? To be able to input data and let the computer figure it out the rules for you so you don’t have to spend hours of your time writing infinite rules ;-)

Now, let’s dive into the last section of the post on which I’ll explain how useful this is and why do we want to actually use it.

## Why do we want to use it? 

There is a lot of research being done around Intelligent Machines, and I have no doubts we might live at some point a new era of sharing our daily lives with those machines. We could even say we are doing it now with the presence of Smart Assistance, Recommendation Systems embedded in the software and applications we use daily, such as Netflix, Airbnb, Amazon between others, but still, I would say there are lots to be done and to experiment with.

If you have read about Machine Learning and Artificial Intelligence before, you might know there is a lot of controversy around this topic, specifically related to automation, machines taking our jobs and we all being fired due to the rise of Machine Intelligence. Well, let me tell you this is far from the truth.

From my point of view, Machine Learning / Artificial Intelligence should be seen as a complement to our skills, to our strengths but also to our weakness, ultimately as a tool where we help the computer by building the model and the computer help us to do calculations at the speed we cannot.

The benefits I’ve seen so far in Business, Society, and Life are massive, a couple of examples here:

- The possibility to detect pneumonia from chest X-rays, a Deep Learning algorithm developed by [Standford](https://stanfordmlgroup.github.io/projects/chexnet/).
- [Farmers Companion App](https://blog.google/technology/ai/machine-learning-meets-african-agriculture/), an application that helps to detect and identify when a crop has been infected by a caterpillar and advice you on how to treat it to stop it from happening.

The later is a good example of how we actually can make Machine Learning part of our lives and skills to boost our work and perhaps produce more with less in our business or simply a better world to live in.
If you are more curious about what types of Machine Learning are around and how you could start thinking if it’s worth it or not to apply it to your business use cases, stick around and follow me as I’m planning to write more about it in next posts.
 

