print "Hello World"
print "The quick brown fox jumped over the lazy dog"

faren = -1000
max_attempts = 6
attempts = 0

while faren < 100 and (attempts < max_attempts):
    newfaren = float(raw_input("Enter Temp"))

    if newfaren > faren:
        print "Hotter"
    elif newfaren < faren:
        print "Colder"
    else:
        continue

    faren = newfaren
    attempts += 1


print "too many attempts"
