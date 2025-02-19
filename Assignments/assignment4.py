'''
Jessica Sheng
ACAD 222, Spring 2025
jlsheng@usc.edu
Homework 4
'''
import random

def roll():
    num_1 = random.randrange(1, 9)
    num_2 = random.randrange(1, 9)
    num_3 = random.randrange(1, 9)
    rolls_list = [num_1, num_2, num_3]
    return rolls_list

def computeBetResult(rolls, bet, guess):
    count = 0
    earnings = 0
    for roll in rolls:
        if roll == guess:
            count += 1
    if count == 0:
        earnings = -bet
    elif count == 1:
        earnings = bet
    elif count == 2:
        earnings = 3 * bet
    elif count == 3:
        earnings = 10 * bet

    return earnings

def main():
    #initialize money
    money = 200

    #user inputs to play game
    print(f"Welcome to the dice game! You have ${money}.")

    playing = True
    while playing:
        bet = int(input("How much would you like to bet? "))
        while bet <= 0 or bet > money:
            print(f"Please enter a bet between 1 and {money}")
            bet = int(input("How much would you like to bet? "))
        print(f"You have bet ${bet}.")
        guess = int(input("What number would you like to bet on? "))
        while guess < 1 or guess > 8:
            print("Please enter a number between 1 and 8.")
            guess = int(input("What number would you like to bet on? "))
        print(f"You have bet on {guess}.")
        print("You have rolled the dice.")

        #store rolls in a list and compute
        rolls = roll()
        print(f"The dice rolls are {rolls[0]}, {rolls[1]}, {rolls[2]}.")

        result = computeBetResult(rolls, bet, guess)
        if result > 0:
            print(f"Congratulations! You have won ${result}.")
        else:
            print("You have lost the bet. Better luck next time.")
        money += result
        print(f"You now have ${money}.")
        if money <= 0:
            print("You have run out of money. Game over.")
            playing = False
            break
        play = input("Would you like to play again? (Y/N) ").upper()
        if play == "N":
            playing = False
            break
    print(f"You have ended the game with ${money}.")
    print("Goodbye!")

main()