def detectCapitalUse(word):
    if word.isupper():
        return True
    elif word.islower():
        return True
    elif word.istitle():
        return True
    else:
        return False
print(detectCapitalUse("USA"))