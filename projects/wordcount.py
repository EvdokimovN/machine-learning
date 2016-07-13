def count_words(string, n):
    string_list = string.split(' ')
    word_occurenccies = {}
    for word in string_list:
        if word in word_occurenccies:
            word_occurenccies[word] += 1
        else: word_occurenccies[word] = 1
    word_occurenccies = sorted(word_occurenccies.items(), key= lambda tup: tup[1], reverse=True)
    return word_occurenccies[:n]
print count_words("betty bought a bit of butter but the butter was bitter", 3)

d = {"fw":1}
print "fw" in d

import math
print math.floor(27.0/2)