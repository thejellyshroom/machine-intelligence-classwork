'''
Jessica Sheng
ACAD 222, Spring 2025
jlsheng@usc.edu
Assignment 2
'''
import random

#part 1
wordList = ["apple", "plantain", "cherry", "date", "elderberry", "mangosteen", "grape", "honeydew"]

print("Welcome to the Word Jumble Game!")
print("Choosing word...")
word = random.choice(wordList)
print("Scrambling word...")

#scramble
#split the word into a list of characters
jumbledWord = list(word)
# process is repeated 10 times to randomize.
# for each iteration, two letters are swapped.
# nested for each loop within a for loop

for i in range(10):
    for j in range(len(jumbledWord)):
        index = random.randint(0, len(jumbledWord) - 1)
        jumbledWord[j], jumbledWord[index] = jumbledWord[index], jumbledWord[j]

# join the list back into a string
jumbledWord = " ".join(jumbledWord)
print(f"Scrambled word: {jumbledWord}")

incorrect = True
guesses = 0
while incorrect:
    guess = input("Guess the word: ").lower()
    if guess == word:
        print("Correct! It took you", guesses, "guesses.")
        incorrect = False
    else:
        print("Incorrect. Try again.")
        guesses += 1

#part 2
print("Great! Now you have acces to the Caesar Cipher.")
alphabetList = list("abcdefghijklmnopqrstuvwxyz")

shiftIndex = int(input("Enter the shift value: "))
message = input("Enter the message to encode: ").lower()

# create cipher
# start with index to move by and make that beginning of list
# joins the first half of alphabet to the second half
cipheredAlphabet = alphabetList[shiftIndex:] + alphabetList[:shiftIndex]
cipheredAlphabet = "".join(cipheredAlphabet)

# encrypt
encryptedMessage = ""
for letter in message:
    #if letter is an alphabet, find its index in unedited alphabet list and use that index to find the ciphered letter
    if letter in alphabetList:
        index = alphabetList.index(letter)
        #append cipher letter to encrypted string
        encryptedMessage += cipheredAlphabet[index]
    else:
        encryptedMessage += letter

print(f"Encrypted message: {encryptedMessage}")
decryptBool = input("Decrypt message? (y/n): ").lower()

# decrypt
if decryptBool == "y":
    decryptedMessage = ""
    for letter in encryptedMessage:
        if letter in alphabetList:
            index = cipheredAlphabet.index(letter)
            decryptedMessage += alphabetList[index]
        else:
            decryptedMessage += letter
    print(f"Decrypted message: {decryptedMessage}")
    print("Original message: ", message)
