import random

lists = []
for i in range(100):
    should_get_same_class = random.randint(0, 1)
    if should_get_same_class:
        lists.append(should_get_same_class)

if __name__ == '__main__':
    print(len(lists))
