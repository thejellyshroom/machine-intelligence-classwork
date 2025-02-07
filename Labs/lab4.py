'''
Jessica Sheng
ACAD 222, Spring 2025
jlsheng@usc.edu
Lab 4
'''
import random

n = int(input("How many random numbers would you like? "))
random_numbers = [random.randrange(40000) for _ in range(n)]

# dictionary to store the frequency of factors
factors_count = {}
for i in range(2, 56):
    factors_count[i] = 0

# determine even division of each
for number in random_numbers:
    for factor in range(2, 56):
        if number % factor == 0:
            factors_count[factor] += 1

# display chart.
print("\nFrequency Chart:")
for factor, count in factors_count.items():
    if count > 0:
        print(f"{factor}: {'*' * count}")

# display numbers
print("\nRandom Numbers:")
print(random_numbers)
