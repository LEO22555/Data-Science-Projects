def findRelativeRanks(scores):
    sorted_nums = sorted(scores, reverse = True)
    ranks = {}
    for i, j in enumerate(sorted_nums):
        if i == 0:
            ranks[j] = "Gold Medal"
        elif i == 1:
            ranks[j] = "Silver Medal"
        elif i == 2:
            ranks[j] = "Bronze Medal"
        else:
            ranks[j] = str(i + 1)
    return [ranks[j] for j in scores]

scores = [5,4,3,2,1]
print(findRelativeRanks(scores))