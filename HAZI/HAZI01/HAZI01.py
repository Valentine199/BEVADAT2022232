# Create a function that returns with a subsest of a list.
# The subset's starting and ending indexes should be set as input parameters (the list aswell).
# return type: list
# function name must be: subset
# input parameters: input_list,start_index,end_index
def subset(input_list: list, start_index: int, end_index: int) -> list:
    return input_list[start_index: end_index]


#Készíts egy függvényt ami egy listát vár paraméterként és ennek a listának minden n-edik elemét adja vissza.
#Paraméterként lehessen állítani azt hogy hanyadik elemeket szeretnénk viszakapni.
#NOTE: a 0. elem is legyen benne
#Egy példa a bemenetre: input_list=[1,2,3,4,5,6,7,8,9], n=3
#Egy példa a kimenetre: [1,4,7]
#return type: list
#függvény neve legyen: every_nth
def every_nth(input_list: list, step_size: int) -> list:
    filtered = list()
    for i in range(0, len(input_list), step_size):
        filtered.append(input_list[i])
    return filtered


# Create a function that can decide whether a list contains unique values or not
# return type: bool
# function name must be: unique
# input parameters: input_list
def unique(input_list: list) -> bool:
    uniques = list()
    for x in input_list:
        if x not in uniques:
            uniques.append(x)

    return len(uniques) == len(input_list)


# Create a function that can flatten a nested list ([[..],[..],..])
# return type: list
# fucntion name must be: flatten
# input parameters: input_list
def flatten(input_list: list) -> list:
    flat_list = list()
    for nest in input_list:
        for x in nest:
            flat_list.append(x)

    return flat_list

# Create a function that concatenates n lists
# return type: list
# function name must be: merge_lists
# input parameters: *args
def merge_lists(*args) -> list:
    con_list = list()
    for x in args:
        con_list += x
    return con_list


#Készíts egy függvényt ami paraméterként egy listát vár amiben 2 elemet tartalmazó tuple-ök vannak,
#és visszatér ezeknek a tuple-nek a fordítottjával.
#Egy példa a bemenetre: [(1,2),(3,4)]
#Egy példa a kimenetre: [(2,1),(4,3)]
#return type: list
#függvény neve legyen: reverse_tuples
def reverse_tuples(input_list: list) -> list:
    reversed_list= list()
    for x in input_list:
        result = reversed(x)
        result = tuple(result)
        reversed_list.append(result)
    return reversed_list


# Create a function that removes duplicates from a list
# return type: list
# fucntion name must be: remove_tuplicates
# input parameters: input_list
def remove_duplicates(input_list: list) -> list:
    uniques = list()
    for i in range(len(input_list)):
        if input_list[i] not in uniques:
            uniques.append(input_list[i])

    return uniques



    return len(uniques) == len(input_list)

# Create a function that transposes a nested list (matrix)
# return type: list
# function name must be: transpose
# input parameters: input_list


def transpose(input_list: list) -> list:
    transposed = list()

    for i in range(len(input_list[0])):
        row = list()
        for inner_list in input_list:
            row.append(inner_list[i])
        transposed.append(row)

    return transposed


#Készíts egy függvényt ami paraméterként egy listát vár és visszatér a lista csoportosított változatával.
#Egy olyan listával térjen vissza amiben a paraméterként átadott chunk_size méretű listák vannak.
#Egy példa a bemenetre: [1,2,3,4,5,6,7,8]
#Egy példa a kimenetre: [[1,2,3],[4,5,6],[7,8]]
#NOTE: ha nem mindegyik lista elemet lehet chunk_size méretű listába tenni akkor a maradékot a példában látott módon kezeljétek
#return type: list
#függvény neve legyen: split_into_chunks
def split_into_chunks(input_list: list, chunk_size: int) -> list:
    resized_list = list()
    size_now = 0
    list_idx = 0
    resized_list.append(list())
    for x in input_list:
        if size_now == chunk_size:
            list_idx += 1
            resized_list.append(list())
            resized_list[list_idx].append(x)
            size_now = 1
        else:
            resized_list[list_idx].append(x)
            size_now += 1

    return resized_list





# Create a function that can merge n dictionaries
# return type: dictionary
# function name must be: merge_dicts
# input parameters: *dict
def merge_dicts(*dict) -> dict:
    main_dict = {}

    for i in range(len(dict)):
        main_dict.update(dict[i])

    return main_dict


# Create a function that receives a list of integers and sort them by parity
# and returns with a dictionary like this: {"even":[...],"odd":[...]}
# return type: dict
# function name must be: by_parity
# input parameters: input_list
def by_parity(input_list: list) -> dict:
    sorted_dict = {"even": [], "odd": []}
    for x in input_list:
        if x % 2 == 0:
            sorted_dict["even"].append(x)
        else:
            sorted_dict["odd"].append(x)

    return sorted_dict

# Create a function that receives a dictionary like this: {"some_key":[1,2,3,4],"another_key":[1,2,3,4],....}
# and return a dictionary like this : {"some_key":mean_of_values,"another_key":mean_of_values,....}
# in short calculates the mean of the values key wise
# return type: dict
# function name must be: mean_key_value
# input parameters: input_dict
def mean_key_value(input_dict: dict):
    result_dict = dict()

    for key in input_dict.keys():
        list_at_idx = input_dict[key]
        avarage = sum(list_at_idx) / len(list_at_idx)
        result_dict[key] = avarage

    return result_dict
