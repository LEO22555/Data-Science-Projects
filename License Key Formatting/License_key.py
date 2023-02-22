def licenseKeyFormatting(s, k):
    s = s.upper().replace('-', '')
    size = len(s)
    s = s[::-1]
    res = []
    for i in range(0, size, k):
        res.append(s[i:i+k])
    return '-'.join(res)[::-1]

print(licenseKeyFormatting("2-4A0r7-4k", 3))