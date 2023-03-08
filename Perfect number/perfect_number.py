def checkPerfectNumber(num):
    if num <= 1:
        return False
    div_sum = 1
    i = 2
        
    # Check divisors up to square root of num
    while i*i <= num:
        if num % i == 0:
            div_sum += i
            if i != num//i:
                div_sum += num//i
        i += 1
    # Check if num is a perfect number
    if div_sum == num:
        return True
    else:
        return False
print(checkPerfectNumber(28))