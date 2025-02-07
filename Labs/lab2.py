'''
Jessica Sheng
ACAD 222, Spring 2025
jlsheng@usc.edu
Lab 2
'''

while True:
    name = input("Enter your name (enter nothing to exit): ")
    if len(name) == 0:
        print("You entered nothing. Exiting program.")
        break
    else:
        # counter
        vowel_count = 0
        for letter in name:
            if letter.lower() in ['a', 'e', 'i', 'o', 'u', 'y']:
                vowel_count += 1

        print(f"Number of vowels: {vowel_count}")