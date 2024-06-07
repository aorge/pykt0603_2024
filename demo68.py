import random


def calculateBmi(height, weight):
    bmi = weight / ((height / 100) ** 2)
    if bmi < 18.5:
        return 'thin'
    elif bmi < 25:
        return 'normal'
    return 'fat'


with open('data/bmi.csv', 'w', encoding='UTF-8') as file1:
    file1.write('height,weight,label\n')
    category = {'thin': 0, 'normal': 0, 'fat': 0}
    for i in range(30000):
        currentHeight = random.randint(110, 200)
        currentWeight = random.randint(40, 60)
        label = calculateBmi(currentHeight, currentWeight)
        category[label] += 1
        file1.write('%d,%d,%s\n' % (currentHeight, currentWeight, label))
print(category)
print("OK, check data dir")