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
Si el modelo consigue preveer el nivel de educación parental en función de las notas del niño podremos decir que hay una influencia.  
El notebook de la solución propuesta esta disponible en el link: [app.py](/app.py) 
***
## analysis
Tenemos 6 variables para intentar predecir el nivel de educación parental: las notas de matemáticas, lectura y escritura, el genero del niño, si tiene becas y si asiste a la academia. El analisis de las correlaciones indica que solo las notas tienen relación con el nivel educacional parental, una relación positiva (más nivel, mejor nota), pero muy pequeña.  
![Alt text](/correlaciones.png)  
Si miramos el gráfico de la distribución del nivel parental según la nota de escritura, dispersando un poco lo elementos alrededor del nivel según una distribución normal para verlo mejor, no se puede apreciar un patrón claro. Lo mismo pasa con las otras variables.  
![Alt text](/relación nivel de educación parental VS nota de escritura del niño.png)  
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
