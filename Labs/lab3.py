'''
Jessica Sheng
ACAD 222, Spring 2025
jlsheng@usc.edu
Lab 3
'''

#dialog input
censoredWords = ["fren", "fral", "drel", "gron", "glud", "zarp", "nark"]

while True:
    dialog = input("Enter a line of text to censor (leave blank to exit): ")
    if dialog == "":
        break
    else:
        #splits by word
        dialogList = dialog.split()
        for i in range(len(dialogList)):
            #case agnostic
            if dialogList[i].lower() in censoredWords:
                dialogList[i] = "BEEP"
        censored_dialog = " ".join(dialogList)
        print(censored_dialog)