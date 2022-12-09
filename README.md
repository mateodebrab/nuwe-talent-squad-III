## background
#### Influencias en el rendimiento académico
Se ha realizado un estudio para ver si el rendimiento académico de los hijos es influenciado por el nivel académico de sus padres. Por ello, se evaluarán los resultados academicos de los estudiantes en función de varias variables.
***
## problem
El objetivo es predecir si los resultados académicos del alumno están influenciados por el nivel educativo de los padres.  
Estos tienen un número que representa el siguiente nivel educativo:  
high school: 0,  
some high school: 1,  
some college: 2,  
associate's degree: 3,  
bachelor's degree: 4,  
master's degree: 5  
Es un problema de clasificación multiclase. Ordenando las etiquetas del nivel educativo de menos a mayor (some high school: 0, high school:1, el resto igual), merece la pena probar con una regresión.  
El notebook de la solución propuesta esta disponible en el link: [app.py](/app.py) 
***
## analysis
We have 8 features, all numerical, corresponding to the parameters measured by the different sensors. The training dataset has 2100 records and the test dataset 900. The distribution between the 3 labels is balanced: 33% of the records for each label. The features are standardized. We normalize them too to put them all on the same scale from -1 to 1. 3 features have almost no correlation with the target: feature7, feature8, feature4. The feature importance graph shows us the same thing. The decision is made not to remove them because 2 percentage points of f1_score are lost by removing them. 
***
## solution
I have trained a ramdom forest model. To do so, I have reserved 30% of the train dataset to measure the model performance.  
I have compiled a gridsearch to optimize the model with the best parameters, with a gain of 0.3% for the f1_score.
***
## results
The result for the final f1_score for the ramdom forest model is 91.54%.  
The model is used to predict the label for the test dataset. The result is in the csv-file [predictions.csv](/predictions.csv) and the json-file [predictions.json](/predictions.json)
***
## license
To resolve the problem, i used the technical package:
> - pycharm
> - python
> - pandas
> - scikit-learn
> - matplotlib
> - seaborn  

I hope you enjoy it as much as I enjoyed solving it.  
Thank you  
Mathieu Debrabander  
mathieudebrabander@hotmail.com  
https://www.linkedin.com/in/mathieu-debrabander-9b780528/
***
