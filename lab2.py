'''
Jessica Sheng
ACAD 222, Spring 2025
jlsheng@usc.edu
Lab 2
'''

name = input("Enter your name: ")
if len(name) == 0:
    print("You entered nothing. Exiting program.")
    exit()
else:
    # counter
    vowel_count = 0
    for letter in name:
        if letter.lower() in ['a', 'e', 'i', 'o', 'u', 'y']:
            vowel_count += 1

    print(f"Number of vowels: {vowel_count}")