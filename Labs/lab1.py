'''
Jessica Sheng
ACAD 222, Spring 2025
jlsheng@usc.edu
Lab 1
'''

print("MENU: Breakfast special (Main: toast)" + "\n" +
      "      Lunch special (Main: spaghetti)" + "\n" +
      "      Dinner special (Main: steak)")

invalid = True
while invalid:
    meal = input("\nPlease enter the meal you would like to order: ")
    if meal == "Breakfast":
        print("Breakfast special: $5.00" + "\n\n" +
              "Combo options: " + "\n" +
              "1. Toast with large coffee" + "\n" +
              "2. Toast with orange juice" + "\n" +
              "3. Toast with milk" + "\n" +
              "4. Toast with sliced fruit")
        invalid = False
        invalidCombo = True
        while invalidCombo:
            combo = input("Which combo would you like to order? (1, 2, 3, 4) ")
            if combo == "1":
                print("Toast with large coffee coming up!")
                invalidCombo = False
            elif combo == "2":
                print("Toast with orange juice coming up!")
                invalidCombo = False
            elif combo == "3":
                print("Toast with milk coming up!")
                invalidCombo = False
            elif combo == "4":
                print("Toast with sliced fruit coming up!")
                invalidCombo = False
            else:
                print("Invalid input. Please enter '1', '2', '3', or '4'.")
    elif meal == "Lunch":
        print("Lunch special: $10.00" + "\n\n" +
                "Combo options: " + "\n" +
                "1. Spaghetti with garlic bread" + "\n" +
                "2. Spaghetti with salad" + "\n" +
                "3. Spaghetti with soda" + "\n" +
                "4. Spaghetti with garlic bread and salad" + "\n")
        invalid = False
        invalidCombo = True
        while invalidCombo:
            combo = input("Which combo would you like to order? (1, 2, 3, 4) ")
            if combo == "1":
                print("Spaghetti with garlic bread coming up!")
                invalidCombo = False
            elif combo == "2":
                print("Spaghetti with salad coming up!")
                invalidCombo = False
            elif combo == "3":
                print("Spaghetti with soda coming up!")
                invalidCombo = False
            elif combo == "4":
                print("Spaghetti with garlic bread and salad coming up!")
                invalidCombo = False
            else:
                print("Invalid input. Please enter '1', '2', '3', or '4'.")
    elif meal == "Dinner":
        print("Dinner special: $15.00" + "\n\n" +
                "Combo options: " + "\n" +
                "1. Steak with mashed potatoes" + "\n" +
                "2. Steak with corn" + "\n" +
                "3. Steak with soda" + "\n" +
                "4. Steak with red wine")
        invalid = False
        invalidCombo = True
        while invalidCombo:
            combo = input("Which combo would you like to order? (1, 2, 3, 4) ")
            if combo == "1":
                print("Steak with mashed potatoes coming up!")
                invalidCombo = False
            elif combo == "2":
                print("Steak with corn coming up!")
                invalidCombo = False
            elif combo == "3":
                print("Steak with soda coming up!")
                invalidCombo = False
            elif combo == "4":
                print("Steak with red wine coming up!")
                invalidCombo = False
            else:
                print("Invalid input. Please enter '1', '2', '3', or '4'.")
    else:
        print("Invalid input. Please enter 'Breakfast', 'Lunch', or 'Dinner'.")