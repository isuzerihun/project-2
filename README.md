ML (Machine Learning) Assignment 2  

Authors:  

Issac Zerihun (s3944721) 

Samuel Chan (s3941488) 

Project: 2 (Predicting Energy Use) 

_________________________________________________________________________________ 

Intro/Approach (Issac) 

This report aims to investigate the outcomes of supervised machine learning algorithms and how they can be used to predict energy use. The requirements of the task were to build two supervised learning algorithms, one neural and non-neural to predict the energy use of a building based on various sensor readings around the facility. Then, attempts to improve the accuracy of the machine learning models can be implemented via regularisation, feature selection and/or hyperparameter tuning. 

This report will be divided into 4 sections where we will be addressing the preprocessing of data as well as explanatory data analysis, the building of both supervised algorithms as well as the final judgement. For the evaluation of the supervised algorithms, the use of RMSE was utilised. Selecting RMSE was selected as in most studies relating to the prediction of energy use, RMSE was determined as the loss function (Khan. S, 2023). This could be due to RMSE’s ability to penalise large errors without exaggerating results as much as MSE. _________________________________________________________________________________ 

Part 1: EDA (Exploratory Data Analysis) + Data Preprocessing (Issac) 

Data preprocessing 

Removal of outliers 

Feature scaling  

Change the datatype of column  

Data preprocessing as well as EDA is both integral building machine learning models. They enable the data to be properly processed for the machine learning algorithm so that it can predict more accurately. Data preprocessing requires the data to be in a standardised format so that when performing EDA, proper insights can be made so that the most fitting model can be selected. It also should be noted that with preprocessing and EDA, this is an iterative process so the insights that are found in EDA can be used to further preprocess the dataset. 

The first technique that was utilised was the removal of outliers. This was done to remove all the anomalous readings that were in the data. Such readings can negatively impact the performance of the model as the outliers make it more difficult for the model to determine a common relationship between the attributes and the target feature (Sharma. V, 2023). The way that outliers were removed in this situation was by finding the lower and upper bounds via the IQR (Interquartile Range) and removing values that are outside the bounds.  

Another technique employed was the modification of the columns.  

Changing of column type: The column type of the initial date column was initially stored with an object type. For the model to accurately predict energy use, date should be stated as a datetime object.  

Feature scaling: Feature scaling was implemented to better aid the model in making its predictions. Power transforming was the first technique applied to the dataset and was intended to nullify all the skews present within the dataset. The inherent skews that are present within the dataset can affect the model’s ability to precisely determine the true relationships between the associated attributes and the target feature as it does not accurately predict some of the less common instances. This under sampling can be rectified by shifting the distribution of the data towards its mean. Another method implemented was standard scaling, which converts all the variables into their z-scores, a score which measures how much a value deviates from its mean. This was done to address the issue of scaling. Since the measurements of the different independent variables are vast and utilise different metrics, when the data is standardised, it quickens the training of the model which makes it more efficient.  

Data explanatory analysis. 

Scatterplots 

 

The scatter plots display how the variables are plotted against the date, with a clear separation between the training and testing datasets. With most of the attributes, both a seasonality and trend can be seen in their respective graphs. Some graphs are different, however in that they display a different trend where the output fluctuates in a quadratic function, as can be seen in the RH graphs (RH_3 and RH_7). This could be due to external factors not recorded in the dataset such as less people in the building during specific times. The trends present graphs like T2, T4 AND T9 in the can be explained with the increase in temperature as the months change.  

 

Boxplots 

 

Here we see the spread as well as the distribution of all the instances both the training and testing dataset as well as the target variable. In most of the boxplots, we see that both training and testing datasets for each column are similar to each other. There are some differences between the training and testing sets for most of the temperature columns (T1, T2, T3 etc). This can be explained due to how the data was split, with the later dates being in the testing set while the earlier dates being in the training set. This will play a key role in deciding which type of model will be selected.  

Heatmap – correlation table 

When generating the heatmap the lack of direct correlation with the target feature was surprising. The variable with the strongest correlation with energy output was the RH_out attribute with a weak score of –0.14. A lot of strong corelations exist with the other variables, one notable was the RH_6. This had a strong correlation with multiple temperature sensors in the building. This could be due to the way the building is potentially structured or where the sensors are placed. These findings will also play a key role in deciding which type of model will be selected. 

Histograms 

 

Upon creation of the histograms, several things were apparent in this instance. For some of the attributes, the range of the testing set was vastly different form the training set. This could be due to the underlying changing nature of the time periods for the training and testing sets. The spread and frequency may have varied but it for the majoity of the attributes, the skew was similar for both training and testing sets (exception of RH_6 and a few others). 

Statistics of target feature:  

The target feature itself is consistent in both sets. Both have similar standards of deviation, as well as distribution, quartile ranges and IQRs (Interquartile Range). The mean and median values for both are also similar for both sets. 

  

 

Part 1b: Data Preprocessing Refinements (Samuel) 

Data preprocessing 

Feature Engineering: 

Cleaning row with NaN 

Feature Selection 

Feature scaling 

 

I have done these data preprocessing because they are missing in part1a, so I have added it to here(part1b). 

For Feature Engineering, I have crested many new features including time-based, lag, rolling mean, interaction, then I have removed row with NaN that created by Feature Engineering. 

For Feature Selection, I have use Recursive Feature Elimination, and choose 10 feature to stay. 

For Feature scaling, I have write a loop to loop through the 10 features that have been selected and use MinMaxScaler for RH,  StandardScaler for temperature and the rest. 

Below are the result from the data preprocessing form part1b 

 

Part 2a: Supervised algorithm 1 – base model (Issac) 

Due to the EDA, the main options for building the model were regression-based algorithms. However, the common linear and polynomial regression algorithms would not have been suitable since there were no strong correlations between any of the attributes and the target feature. When considering the data drift that was noticed in the histogram section, as well as the scatterplot we can deduce that time plays a significant role in the output of the energy. As a result, the decision to incorporate time series forecasting as a predictive model was made, which addresses the issue of weak attributes by analysing how the target feature behaviour through time and basing its prediction based off its movement. 

There were multiple time series algorithms that could have been implemented. The first choice was to use Facebook’s Prophet algorithm but due to downloading and importing issues, the second option selected was to use the XGBoost algorithm. This algorithm was preferred because it was commonly used to perform multivariate forecasting, as opposed to other models like ARIMA. Its multiple hyperparameters also allow for detailed fine-tuning so that the algorithm can work as efficiently as possible.  

Before running the model, the training data was split further into validation and training sets. This was done to follow good ML training practices. For the XGBoost, its regressor model was selected and built. After implementing the base model with no tweaks to its hyper parameters the results are shown. 

RMSE for training: (8.572147465301786), MSE for training: (73.48171216687983) 

RMSE for testing: (30.26204195087139), MSE for testing: (915.7911830362999) 

The considerable difference between the MSE and RMSE for training and testing sets suggest that there is overfitting present. Having a RMSE of (30.26204195087139) Watts/hour (WH) is still a decent score however and indicates that if the model’s hyperparameters are refined, then the model will perform well on new independent data.  

 

Part 2b: Supervised algorithm 1(Refinement) (Samuel) 

First, I have used the refined data to build the model using the same method as part 2a, by doing that we can  see that the result have big improvements, that mean what I have done in part 1b is in the right direction. The result are below: 

RMSE for testing (non - neural): 0.7863883143950806 

MSE for testing (non - neural): 0.6184065810171361 

After that, I have done RandomizedSearchCV hyperparameter tunning, 

Which give me a better result: 

RMSE for testing (non - neural): 0.6968827292491029 

MSE for testing (non - neural): 0.48564553832567836 

 

 

Part 3A: Supervised algorithm 2 (neural) – base model (Issac) 

The second algorithm selected was a neural network. The type of neural network built was called a recurrent neural network (RNN). Compared to other models, RNN and LSTM networks perform better at predicting stock prices because they can recognise temporal relationships with target features in data can be used in multiple areas such as stock price predictions, weather predictions and more. The specific RNN utilised which was LSTM (Long-Short-Term-Memory) aims to store memory from past inputs via a cell state and then tries to make its predictions based on the new inputs, with the cell state being continually being updated when added information comes in.  

The base model that was built was a one with 2 layers that are layered sequentially. The first layer was an LSTM layer with 64 neurons. The second layer was a Dense Layer with one neuron and its activation function set to ‘linear’. This Second layer was put there so that the neuron can combine the output from all neurons and produce a single value which would be its prediction. The model has in total 65 neurons and 16,961 trainable parameters with no activation functions set.  

The model was built, fitted onto the data and here are the results.  

RMSE for training (neural): (31.59309307924621) 
MSE for training (neural): (998.1235303139148)  
RMSE for testing (neural): (26.099640198031796) 
MSE for testing (neural): (681.1912184667171)  

This was a reasonable performance as both the MSE and RMSE of the training and testing sets are within a small range of reach other. This means that the was no overfitting present and that the model generalised well to the data. Having the RMSE of (26.09964019803179) for the testing is a respectable score and it performed much better than the base model for the non-neural algorithm. However well this data performed with its RMSE, when graphing its predictions against the validation, it displayed a linear graph, which does not display the energy’s fluctuating output over time which does not reflect reality. Overall, this demonstrates the neural network’s capability to make better predictions based on the detailed hyperparameter tuning and regularisation.  

Part 3b: Suggested improvements (Issac) 

Selecting The Best Activation Function. 

Selecting the best activation function is paramount to the performance of the neural network. They are responsible for how the model trains on its data and to select a suitable function would be paramount to improving efficiency.  

There are many activation functions available to use. Due to previous research conducted, the RELU activation function was commonly used to aid the help of time series forecasting when building neural networks (Sharma. S, 2023). Some of the most popular functions relating to RELU like leaky RELU, ELU, SELU were tried and tested, and the results were.  

 

 

While they all performed well, the best performing activation function was when the activation function was set to (FUNCT 1) and (FUNCT 2) for our two layers, which had a RMSE of (22.611593) for its testing dataset. 

 

Selecting The Best Number of Features. 

Selecting the right number of features is important to improving our model. The dataset currently has 26 attributes which is quite expansive. Using all these attributes to conclude our model would unnecessarily increase the complexity of the network, but having too few can hinder the model’s performance. It is critical to obtain the right number of features. 

The way this was achieved was by importing the SelectKBest and fclassif functions from the feature selection library. These were the tools that aided us with selecting the optimal features for our dataset. The only problem was finding out how many features were optimal. After iterating through all columns, here were the results:  

 

 

Applying Regularisation. 

Applying regularisation to a model can be a great refinement tool that can aid the processing of the model. Regularisation is a type of feature selection that minimises the significance of attributes that have little to no effect on the results of the model. There are many types of regularisations like Lasso and Ridge regularisation, but these types all produce the same effect. L1 (lasso regression) regularisation was the technique selected, and there was one parameter to tune which was the lambda function. The lambda function is the function which is responsible for how extreme the effects of regularisation are on the model built. If the lambda parameter is too high, then bias also increases with a decrease in variance (What is lasso regression?, 2024).  

 

The model was built with L1 regularisation applied, the lambda function set 0.1 and the optimum parameters from previously and here are the results.  

RMSE for training (neural): (23.916455971536315) 
MSE for training (neural): (571.9968662384351)  
RMSE for testing (neural): (31.254655704589194) 
MSE for testing (neural): (976.8535032124098) 

 

Here the results performed worse than previous iterations. This suggests that regularisation isn’t necessary. 

Part 4: Final judgement (Samuel) 

Based on the results, I have decided the non – neural(refined) model is the best among those we have done, because the RMSE and MSE is the lowest. 

So, I have tested the model on the preserved data that we save from the start, and the result are below, 

RMSE : 0.8421429385245848 

MSE : 0.9176834631421582 

 

 
As you can see the trend is on the right, but certainly there can still be improvement on it 

 

Part 5: comparing with other paper(Samuel) 

Data-Driven Modeling of Appliance Energy Usage by Cameron Francis Assadian, 2023 
https://doi.org/10.3390/en16227536 

In this paper, they have mention that they have done multiple linear regression, support vector regression, random forest, gradient boosting, xgboost, and the extra trees regressor. And both feature engineering and hyperparameter tuning methodologie were use on these models. 

The RMSE that they get is 59.61 and ours is 0.84, which mean our model is very good or is just because their way of preprocessing the data is different to us, which can make the value in their data is bigger than us, so it can be normal for their RMSE to be bigger. 

 

 

 

References 

Sharma, V. (2023, April 20). Outliers in machine learning: Understanding their impact on data analysis. Medium. https://levelup.gitconnected.com/outliers-in-machine-learning-understanding-their-impact-on-data-analysis-7ee025fb0022 

Sharma, S., & Sharma, S. (2020). ACTIVATION FUNCTIONS IN NEURAL NETWORKS. Towards Data Sci, 4(12), 310–316. https://ijeast.com/papers/310-316,Tesma412,IJEAST.pdf 

 

Khan, N., Khan, S. U., & Baik, S. W. (2023). Deep dive into hybrid networks: A comparative study and novel architecture for efficient power prediction. Renewable and Sustainable Energy Reviews, 182, 113364. https://doi.org/10.1016/j.rser.2023.113364 

 

What is lasso regression? (2024, January 16). What Is Lasso Regression? | IBM; IBM. https://www.ibm.com/topics/lasso-regression 
