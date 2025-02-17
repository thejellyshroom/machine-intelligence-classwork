'''
Jessica Sheng
ACAD 222, Spring 2025
jlsheng@usc.edu
Lsb 5
'''

import os
def loadFile(filename, data):
    try:
        '''#issues with trying to open file
        full_path = os.path.abspath(filename)
        print(f"Trying to open: {full_path}")

        # Check if file exists
        if not os.path.exists(full_path):
            print(f"File does not exist: {full_path}")
            return False'''

        with open(filename, 'r') as file:
            for line in file:
                line = line.strip().split(',')

                #skip bad data
                if len(line) < 2:
                    print("Skipping invalid line")
                    continue

                #turn all capital to lower
                line[0] = line[0].lower()

                #add to dictionary
                data[line[0]] = line[1]
        return True
    except Exception as e:
        #debugging
        print(f"Could not open the file \"{filename}\"")
        print(f"Error: {e}")
        return False
    
def main():
    data = {}
    exit_program = False
    while exit_program == False:
        filename = input("Choose file to parse: ")
        if filename == "":
            print("Goodbye!")
            exit_program = True
        if loadFile(filename, data):
            print("Loading data...")
            parsing_subreddit = True
            while parsing_subreddit:
                subreddit = input("Choose a Subreddit: ").lower()
                if subreddit == "":
                    print("Goodbye!")
                    parsing_subreddit = False
                    exit_program = True
                if subreddit in data:
                    print(f"{subreddit.capitalize()} --> {data[subreddit]}")
                else:
                    print(f"\"{subreddit}\" is an unknown subreddit")

main()