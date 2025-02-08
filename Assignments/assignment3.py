'''
Jessica Sheng
ACAD 222, Spring 2025
jlsheng@usc.edu
Homework 3
'''

months = { 
    "january": 31, "february": 28, "march": 31, "april": 30, 
    "may": 31, "june": 30, "july": 31, "august": 31, 
    "september": 30, "october": 31, "november": 30, "december": 31 }
calendar = {}

# goes through each key pair in months
for month, days in months.items():
    # create list of empty strings based on days in month for each month
    calendar[month] = [""] * days
    
# print(calendar)

while True:
    date = input("Enter a date for a holiday (i.e. 'July 1' or press enter to view results): ").lower()

    #view results
    if date == "":
        print('Goodbye!')
        break

    date = date.split()
    # bad input
    if len(date) != 2:
        print("I don't see good input in there!")
        continue
    
    month = date[0] #month is first element
    day = date[1] #day is second element

    #bad month
    if month not in calendar:
        print(f"I don't know about the month '{month}'")
    #bad day
    elif int(day) > len(calendar[month]) or int(day) < 1:
        print(f"That month only has {len(calendar[month])} days!")
    #good input
    else:
        event = input(f"What happens on {month}, {day}? ")
        calendar[month][int(day) - 1] = event

print("\nResults:")
for month, events in calendar.items():
    day = 0
    for event in events:
        if event != "":
            print(f"{month.capitalize()} {day + 1} : {event}")
        day += 1  