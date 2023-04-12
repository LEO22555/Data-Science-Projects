def findErrorNums(nums):
    n = len(nums)
    
    duplicate = -1
    for num in nums:
        if nums[abs(num) - 1] < 0:
            duplicate = abs(num)
        else:
            nums[abs(num) - 1] *= -1
    
    missing = -1
    for i in range(n):
        if nums[i] > 0:
            missing = i + 1
    return [duplicate, missing]

nums = [1,2,2,4]
print(findErrorNums(nums))