import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''
FONTOS: Az első feladatáltal visszaadott DataFrame-et kell használni a további feladatokhoz. 
A függvényeken belül mindig készíts egy másolatot a bemenő df-ről, (new_df = df.copy() és ezzel dolgozz tovább.)
'''

'''
Készíts egy függvényt, ami egy string útvonalat vár paraméterként, és egy DataFrame ad visszatérési értékként.

Egy példa a bemenetre: 'test_data.csv'
Egy példa a kimenetre: df_data
return type: pandas.core.frame.DataFrame
függvény neve: csv_to_df
'''
def csv_to_df(path: str) -> pd.core.frame.DataFrame:
    retDf = pd.read_csv(path)
    return retDf

df = csv_to_df("StudentsPerformance.csv")


'''
Készíts egy függvényt, ami egy DataFrame-et vár paraméterként, 
és átalakítja azoknak az oszlopoknak a nevét nagybetűsre amelyiknek neve nem tartalmaz 'e' betüt.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_capitalized
return type: pandas.core.frame.DataFrame
függvény neve: capitalize_columns
'''
def capitalize_columns(input: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    newDf = input.copy()
    idxs = np.array(newDf.keys())

    for x in range(len(idxs)):
        if 'e' not in idxs[x]:
            idxs[x] = idxs[x].upper()

    newDf.columns = idxs
    return newDf
'''
Készíts egy függvényt, ahol egy szám formájában vissza adjuk, hogy hány darab diáknak sikerült teljesíteni a matek vizsgát.
(legyen az átmenő ponthatár 50).

Egy példa a bemenetre: df_data
Egy példa a kimenetre: 5
return type: int
függvény neve: math_passed_count
'''
def math_passed_count(input: pd.core.frame.DataFrame) -> int:
    sum = len(input[input["math score"] >= 50])
    return sum


'''
Készíts egy függvényt, ahol Dataframe ként vissza adjuk azoknak a diákoknak az adatait (sorokat), akik végeztek előzetes gyakorló kurzust.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_did_pre_course
return type: pandas.core.frame.DataFrame
függvény neve: did_pre_course
'''
def did_pre_course(input: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    done = input[input["test preparation course"] == "completed"]
    return done


'''
Készíts egy függvényt, ahol a bemeneti Dataframet a diákok szülei végzettségi szintjei alapján csoportosításra kerül,
majd aggregációként vegyük, hogy átlagosan milyen pontszámot értek el a diákok a vizsgákon.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_average_scores
return type: pandas.core.frame.DataFrame
függvény neve: average_scores
'''
def average_scores(input: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    scores = ["math score","reading score","writing score"]
    averages = input.groupby('parental level of education')[scores].mean()
    return averages


'''
Készíts egy függvényt, ami a bementeti Dataframet kiegészíti egy 'age' oszloppal, töltsük fel random 18-66 év közötti értékekkel.
A random.randint() függvényt használd, a random sorsolás legyen seedleve, ennek értéke legyen 42.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_with_age
return type: pandas.core.frame.DataFrame
függvény neve: add_age
'''
def add_age(input: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    newDf = input.copy()
    np.random.seed(42)
    newDf["age"] = np.random.randint(18, 66)
    return newDf



'''
Készíts egy függvényt, ami vissza adja a legjobb teljesítményt elérő női diák pontszámait.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: (99,99,99) #math score, reading score, writing score
return type: tuple
függvény neve: female_top_score
'''
def female_top_score(input: pd.core.frame.DataFrame) -> tuple:
    scores = ["math score", "reading score", "writing score"]

    bestFemaleStudent = input[input['gender'] == 'female']
    bestFemaleStudentAvarage = bestFemaleStudent[scores].mean(axis=1)

    bests = bestFemaleStudent[bestFemaleStudent[scores].mean(axis=1) == bestFemaleStudentAvarage][scores]
    best = bests.iloc[0]
    tup = best.tolist()

    return tuple(tup)


'''
Készíts egy függvényt, ami a bementeti Dataframet kiegészíti egy 'grade' oszloppal. 
Számoljuk ki hogy a diákok hány százalékot ((math+reading+writing)/300) értek el a vizsgán, és osztályozzuk őket az alábbi szempontok szerint:

90-100%: A
80-90%: B
70-80%: C
60-70%: D
<60%: F

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_with_grade
return type: pandas.core.frame.DataFrame
függvény neve: add_grade
'''
def add_grade(input: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    newDf = input.copy()
    scores = ["math score", "reading score", "writing score"]

    newDf['grade'] = ""

    for i in range(len(newDf)):
        percent = (newDf.iloc[i][scores].sum() / 300) * 100
        if 90 <= percent <= 100:
            newDf['grade'][i] = 'A'
        elif 80 <= percent < 90:
            newDf['grade'][i] = 'B'
        elif 70 <= percent < 80:
            newDf['grade'][i] = 'C'
        elif 60 <= percent < 70:
            newDf['grade'][i] = 'D'
        else:
            newDf['grade'][i] = 'F'


    return newDf


'''
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan oszlop diagrammot,
ami vizualizálja a nemek által elért átlagos matek pontszámot.

Oszlopdiagram címe legyen: 'Average Math Score by Gender'
Az x tengely címe legyen: 'Gender'
Az y tengely címe legyen: 'Math Score'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: math_bar_plot
'''
def math_bar_plot(input: pd.core.frame.DataFrame):
    newData = input.groupby('gender')['math score'].mean()

    fig, ax = plt.subplots()
    x = newData.index
    height = newData.values

    ax.bar(x, height)
    ax.set_title('Average Math Score by Gender')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Math Score')


    return fig



''' 
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan histogramot,
ami vizualizálja az elért írásbeli pontszámokat.

A histogram címe legyen: 'Distribution of Writing Scores'
Az x tengely címe legyen: 'Writing Score'
Az y tengely címe legyen: 'Number of Students'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: writing_hist
'''
def writing_hist(input: pd.core.frame.DataFrame):
    newData = input['writing score']

    fig, ax = plt.subplots()

    ax.hist(newData)
    ax.set_title('Distribution of Writing Scores')
    ax.set_xlabel('Writing Score')
    ax.set_ylabel('Number of Students')


    return fig

''' 
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan kördiagramot,
ami vizualizálja a diákok etnikum csoportok szerinti eloszlását százalékosan.

Érdemes megszámolni a diákok számát, etnikum csoportonként,majd a százalékos kirajzolást az autopct='%1.1f%%' paraméterrel megadható.
Mindegyik kör szelethez tartozzon egy címke, ami a csoport nevét tartalmazza.
A diagram címe legyen: 'Proportion of Students by Race/Ethnicity'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: ethnicity_pie_chart
'''
def ethnicity_pie_chart(input: pd.core.frame.DataFrame):
    fig, ax = plt.subplots()
    labels = input['race/ethnicity'].unique()
    ethnicity_count = input.groupby('race/ethnicity')['race/ethnicity'].count()

    ax.pie(ethnicity_count.values, labels=labels, autopct='%1.1f%%')
    ax.set_title('Proportion of Students by Race/Ethnicity')

    return fig
