#Build a sentance one word at a time.

sentance   = ''
done   = False

while not done:
    word = raw_input("Enter word: ")
    if (word is '!') or (word is '.') or (word is '?'):
        done = True
        sentance = sentance + ' '+word
        print sentance
    else:
        sentance = sentance + ' '+word
        print "So far: "+sentance


