'''
Jessica Sheng
ACAD 222, Spring 2025
jlsheng@usc.edu
Homework 1
'''
# input prompts
name = input("Hello, input your name to calculate BMI: ")
feet = int(input("Input your height in feet and inches. \nFeet: "))
inches = int(input("Inches: "))
weight = float(input("Input your weight in pounds: "))

# calculate height
total_inches = (feet * 12) + inches
# print(f"Total inches: {total_inches}")
total_meters = total_inches / 39.37
# print(f"Total meters: {total_meters}")

# calculate weight
total_kilograms = weight / 2.2
# print(f"Total kilograms: {total_kilograms}")

# calculate BMI
bmi = total_kilograms / (total_meters ** 2)
print(f"{name}'s BMI: {bmi:.1f}")