from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from nltk.stem.snowball import RussianStemmer

def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(ord(el)-1000)
        print(result)
        return result

def fullfil(x):
    result = list(x);
    if len(result) < 15:
        for i in range(15-len(result)):
            result.append("Ϩ")
            return result

if __name__ == "__main__":
print(chr(1000))
cities = {1 : "Москва", 2: "Санкт-Петербург" , 3: "Пермь", 4 : "Омск", 5 : "Екатеринбург", 6 : "Тюмень", 7 : "Ижевск"}
stemmer = RussianStemmer()
s1 = list(stemmer.stem("Москва"))
net = buildNetwork(15, 30, 1)
dataset = SupervisedDataSet(15, 1)

dataset.addSample(flatten(fullfil(stemmer.stem("Москва"))), 1)
dataset.addSample(flatten(fullfil(stemmer.stem("Моск"))), 1)
dataset.addSample(flatten(fullfil(stemmer.stem("Мск"))), 1)
dataset.addSample(flatten(fullfil(stemmer.stem("Нерезин"))), 2)
dataset.addSample(flatten(fullfil(stemmer.stem("Нерезиновск"))), 1)
dataset.addSample(flatten(fullfil(stemmer.stem("Резин"))), 1)
dataset.addSample(flatten(fullfil(stemmer.stem("Дефолт"))), 1)
dataset.addSample(flatten(fullfil(stemmer.stem("Дефолтсити"))), 1)

dataset.addSample(flatten(fullfil(stemmer.stem("Питер"))), 2)
dataset.addSample(flatten(fullfil(stemmer.stem("Петр"))), 2)
dataset.addSample(flatten(fullfil(stemmer.stem("Петер"))), 2)
dataset.addSample(flatten(fullfil(stemmer.stem("Санкт-Петербург"))), 2)
dataset.addSample(flatten(fullfil(stemmer.stem("Спб"))), 2)
dataset.addSample(flatten(fullfil(stemmer.stem("Ленинград"))), 2)
dataset.addSample(flatten(fullfil(stemmer.stem("Ленинск"))), 2)
dataset.addSample(flatten(fullfil(stemmer.stem("Ленин"))), 2)

dataset.addSample(flatten(fullfil(stemmer.stem("Пермь"))), 3)
dataset.addSample(flatten(fullfil(stemmer.stem("Перм"))), 3)
dataset.addSample(flatten(fullfil(stemmer.stem("Молотов"))), 2)
dataset.addSample(flatten(fullfil(stemmer.stem("Молот"))), 2)

dataset.addSample(flatten(fullfil(stemmer.stem("Омск"))), 4)
dataset.addSample(flatten(fullfil(stemmer.stem("Омс"))), 4)
dataset.addSample(flatten(fullfil(stemmer.stem("Омич"))), 4)

dataset.addSample(flatten(fullfil(stemmer.stem("Екатеринбург"))), 5)
dataset.addSample(flatten(fullfil(stemmer.stem("Екат"))), 5)
dataset.addSample(flatten(fullfil(stemmer.stem("Ебург"))), 5)
dataset.addSample(flatten(fullfil(stemmer.stem("Свердловск"))), 5)
dataset.addSample(flatten(fullfil(stemmer.stem("Свердл"))), 5)

dataset.addSample(flatten(fullfil(stemmer.stem("Тюмень"))), 6)
dataset.addSample(flatten(fullfil(stemmer.stem("Тюм"))), 6)
dataset.addSample(flatten(fullfil(stemmer.stem("Тюма"))), 6)

dataset.addSample(flatten(fullfil(stemmer.stem("Ижевск"))), 7)
dataset.addSample(flatten(fullfil(stemmer.stem("Иж"))), 7)
dataset.addSample(flatten(fullfil(stemmer.stem("Ижев"))), 7)

trainer = BackpropTrainer(net, dataset)
error = 10
iteration = 0
while error > 1:
    error = trainer.train()
    iteration += 1
print("Error on iteration {0} is {1}".format(iteration, error))

a = float(net.activate(flatten(fullfil((stemmer.stem("Мол"))))))
print(a)
print("\n", cities.get(round(a)))
a = float(net.activate(flatten(fullfil(stemmer.stem(("Питр"))))))
print(a)
print("\n", cities.get(round(a)))
a = float(net.activate(flatten(fullfil(stemmer.stem(("Резина"))))))
print(a)
print("\n", cities.get(round(a)))
a = float(net.activate(flatten(fullfil((stemmer.stem("Тюмен"))))))
print(a)
print("\n", cities.get(round(a)))
a = float(net.activate(flatten(fullfil((stemmer.stem("Ижа"))))))
print(a)
print("\n", cities.get(round(a)))