# ATX House Price

## Overview 
This project aims to provide an estimate for the price of Houses in the Austin area based on numeric features provided by the user. 

The regression was created via data found on Kaggle. Data was subsequently explored in terms of indepedent variables and their relationship to the price of houses. Regression models were compared, tuned, and pickled for deployment via Flask and Docker. Moreover it can be deployed via AWS on Elastic Beanstalk. 

Motivation: As a potential home buyer in the near future, this project has personal motivation to 

## Data Description 
The [Kaggle dataset](https://www.kaggle.com/datasets/ericpierce/austinhousingprices) can be found here. The zip file contains around ~14k listings in the area. 
- Independent Variables (variables that we are going to explore to predict price): lot size, number of beds/baths, proximity to schools etc.
- Dependent Variable (the variable that we are interested in predicting): price of the home

Should be noted that as of 11/8 that there are some variables of interest that could be included in the feature engineering step that can be abstracted from descriptions but for now keeping things rather simple

> `EDA.ipynb` contains instructions on how to programmatically download the dataset via the CLI OR simply click the above link and press download

## Modeling
Modeling was done using a simple Linear Regression model and Random Forest Regressor and comparing two important metrics for regression: R^2 and RSME. Random Forest was chosen as the final model due to having a higher R^2 and lower error. The model was subsequently tuned to increase performance. The final model was pickled for use in deployment.

> `train.py` trains the and pickles the final model

## Local Deployment

`pipenv` and `docker` were used for deployment for local machines

1) Clone the repo to your local computer via `git clone <INSERT LINK HERE>`
2) `cd` into the repo
3) Build the docker image: `docker build -t <name_your_app> .`
4) Next run: `docker run -p 9696:9696 <name_your_app>`
5) A testing script was provided with `Test-Values.ipynb` or use `http://0.0.0.0:9696/predict`
6) Edit the value in the test script and see the prediction!

To elaborate, since we have used `pipenv` to create a virtual env and having `docker` build an image based on those requirements we can containerize our application and ensure that the Flask app is served and the port exposed correctly 

## Cloud Deployment

Using AWS and Docker Hub (this assumes you have an account for both) 

1) Push the image to docker hub (this will save some steps later on in Elastic Beanstalk): `docker push <username>/<app_name>:latest`
2) Go to [AWS Management Console](https://aws.amazon.com/console/) and login with your credentials
3) Go to `Elastic Beanstalk` and hit "Create Application"
4) Next name your application and select `local files` and upload the provided `Dockerrun.aws.json` in the repo
5) Click next and in the `IAM` roles ensure that your roles for "Elasti Beanstalk" AND "EC2" have the proper policies for permissions (you may have to edit the Principal services which can be found in the Trust Relationships tab in IAM roles) NOTE: Without going into details, we want to ensure that the roles provided for the application have the necessary permissions for the services we are building (but only for those services!)
6) Click Submit! My EB domain: Ahpp-env.eba-tymcjbic.us-east-1.elasticbeanstalk.com (NOTE: Due to AWS pricing the link may no longer work at the time of review)


