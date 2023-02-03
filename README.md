# Detecting Fake Amazon reviews using NLP


## Why?

Online reviews are becoming increasingly deceptive in today's world where 
e-commerce is the primary form of shopping and the reason for this is the use
of bots. Making an informed decision is no longer a simple task which leaves 
consumers spending less money online overall. This prevents the consumer from
purchasing an item they need and reduces revenue for companies. The solution
is to use AI to detect AI by creating a machine learning model to detect these
fake reviews. The model I have created is able to identify fake reviews with
an 87% overall accuracy.

![online shopping](AI_Academy_Capstone/readme_banner.jpeg)

## Code
The code uses an extensive amount of libraries to enable several different 
cleaning, processing, and modeling capabilities. Such libraries include:
    - pandas
    - nltk
    - sklearn
    - seaborn
    - scipy
    - numpy

Using these libraries, code performs the following data processing steps
    1. Loading the data from a CSV file into a pandas dataframe
    2. exploring the dataset and using seaborn to create intuitive graphs
    3. Cleaning the data through normalizing, tokenizing, lemmatizing, and 
       stop word removal
    4. The sentiment of the reviews is taken using the Sentiment Intesity 
       Analyzer to create a compound sentiment score from -1 to 1
    5. Getting the TF-IDF of the words within the reviews
    6. Using hstack to bring together sentiment and tf-idf to use as one 
       feature for the modeling

Next is creating the base model. This is done using a logistic regression model.
First a train test split is done on the data with 20% being test data and 80%
being training data. This allows the model to test itself on data it has not 
seen before during training. The results of the model are then captured using
the several different statistical metrics. 

The base model performed quite well:
Accuracy 0.8728194977112458
Precision 0.8879689521345407
Recall 0.8522473305189968
F1 Score 0.8697415103902686

After creating the base model as a comparison, a new model will be used with 
more sophisticated methods. The new method is Random Forest model, which is an
ensemble of decision trees averaged out to predict an outcome. For this model,
a pipeline is created and then the hyperparameters are tuned to create a model
that is hopefully better performing. 

When it came to the actual results of the random forest pipeline, the results 
were not as good as the base model:

Accuracy 0.8566126438203637
Precision 0.83646175504458
Recall 0.8852743978147505
F1 Score 0.8601761370491012

When trying to fine tune the parameters, the results yielded even worse 
results. The reason for this could be the random forest method is not the best
option for this specific dataset, or the parameters adjusted simply were not
the best ones. 

## Which Model should be used?

The winning model seems to be the logistic regression model (base model). While
it is disappointing the more sophisticated model performed worse, having a 
model that performs well at all is valuable when trying to create a more 
transparent online space for shopping. This model is not perfect and will flag
legitimate reviews as fake, so in order to avoid silencing the voices of real
reviewers, this model could be used as a suggestion to the user. It could create
a composite rating for a product after ignoring the ones it flags as fake. 
This could create a more accurate rating despite being imperfect. 


## How to navigate repository
The repository has five main files to be viewed:
    - The Readme 
    - the main notebook called 'AI Academy Capstone.ipynb' (link)
    - the amazon review csv file
    - project proposal
    - project presentation (link)
    
## Reproducing results
To reproduce the results of the notebook, simply fork the repo, ensure the
necessary datasets and library are installed, and then run
    

