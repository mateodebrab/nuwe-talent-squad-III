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
![Alt text](/relación_nivel_de_educacion_parental_VS_nota_de_escritura_del_niño.png)  
![Imagen de biblioteca de pocimas de druida](https://image.freepik.com/foto-gratis/libreria-antigua-magia-hechiceros-libros-misteriosos-encantamientos_154868-242.jpg)
***
## solution
Entrené varios modelos de clasificación multiclase además de un modelo de redes neuronales. El mejor modelo, por poco delante de las redes neuronales, es el GradientBoostingClassifier con un F1_score de 0.2460 una vez hiperparametrizado.  
Entrené también unos modelos de regresión, después de ordenar las etiquetas de nivel de educación parental. Después de redondear las predicciones, el mejor modelo de regresión es la regresión linear, muy poco por encima de la linea de una predicción aleatoria, con un F1_score de 0.1770.  
***
## results
El resultado es que el mejor modelo GradientBoostingClassifier con un F1_score de 0.2460 no es un buen modelo. Un predicción hecha al hazar del nivel de educación parental daría un F1_score de 0.1666. Es decir que al hazar se acertaría el nivel de educación parental una vez sobre 6, al tener 6 etiquetas.  
El mejor modelo acierta una vez sobre 4. Es mejor que el hazar, pero no llegamos ni a acercarnos a acertar la mitad de las veces.  
Mi conclusión es que no podemos decir que el nivel de educación parental influye de forma significativa sobre las notas del niño.  
El modelo mejora un poco reduciendo el número de etiquetas a 3 (sin estudios, high school, universitario), pero tampoco mucho.  
En el archivo csv [predictions.csv](/predictions.csv) encontrarás las predicciones en csv y en el archivo [predictions.json](/predictions.json) las predicciones en json.  
***
## license
Para resolver el problema utilice el siguiente paquete:
> - python~=3.10
> - numpy~=1.23.5
> - tensorflow~=2.11.0
> - pandas~=1.5.2
> - scikit-learn~=1.2.0
> - matplotlib~=3.6.2
> - seaborn~=0.12.1  

Espero disfrutes de mi codigo tanto como yo resolviendo este ejercicio.    
Muchas gracias   
Mathieu Debrabander  
https://www.linkedin.com/in/mathieu-debrabander-9b780528/
***
