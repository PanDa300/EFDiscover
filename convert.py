symbol = "sub(sub(mul(add(add(add(sub(X12, X20), X6), sub(X21, X22)), sub(X21, X22)), add(add(X17, X25), sub(X12, X20))), div(sub(X17, X6), div(mul(add(X29, X35), sub(X22, X3)), mul(X6, X11)))), div(div(add(add(X0, X6), sub(X21, X22)), sub(X17, X6)), div(mul(sub(X24, sub(div(mul(mul(add(div(X35, X11), X6), add(sub(X12, X20), X6)), sub(X22, X3)), mul(X6, X11)), X19)), X10), mul(add(X33, X18), add(sub(X12, X20), X6)))))"


# add(sub(X12, X20), X6), sub(X21, X22))
def add(str):
    l = 1
    r = 1
    cnt = -1
    mid = []
    while r < len(str) and cnt < 0:
        if str[r] == '(':
            cnt -= 1
        elif str[r] == ')':
            cnt += 1
        if cnt == -1 and str[r] == ',':
            mid.append(r)
        r += 1
    res = []
    last = l
    for i in mid:
        res = res + find_symbol(str[last:i]) + ['+']
        last = i + 1
    res = res + find_symbol(str[last:r])
    res = ['('] + res + [')']
    return res


def sub(str):
    l = 1
    r = 1
    cnt = -1
    mid = []
    while r < len(str) and cnt < 0:
        if str[r] == '(':
            cnt -= 1
        elif str[r] == ')':
            cnt += 1
        if cnt == -1 and str[r] == ',':
            mid.append(r)
        r += 1
    res = []
    last = l
    for i in mid:
        res = res + find_symbol(str[last:i]) + ['-']
        last = i + 1
    res = res + find_symbol(str[last:r])
    res = ['('] + res + [')']
    return res


def mul(str):
    l = 1
    r = 1
    cnt = -1
    mid = []
    while r < len(str) and cnt < 0:
        if str[r] == '(':
            cnt -= 1
        elif str[r] == ')':
            cnt += 1
        if cnt == -1 and str[r] == ',':
            mid.append(r)
        r += 1
    res = []
    last = l
    for i in mid:
        res = res + find_symbol(str[last:i]) + ['*']
        last = i + 1
    res = res + find_symbol(str[last:r])
    res = ['('] + res + [')']
    return res


def div(str):
    l = 1
    r = 1
    cnt = -1
    mid = []
    while r < len(str) and cnt < 0:
        if str[r] == '(':
            cnt -= 1
        elif str[r] == ')':
            cnt += 1
        if cnt == -1 and str[r] == ',':
            mid.append(r)
        r += 1
    res = []
    last = l
    for i in mid:
        res = res + find_symbol(str[last:i]) + ['/']
        last = i + 1
    res = res + find_symbol(str[last:r])
    res = ['('] + res + [')']
    return res


def find_symbol(str):
    xx = "".join(str)
    print(xx)
    l = 0
    r = len(str)
    while l < r and (str[l] == ' '):
        l += 1
    while l < r and (str[r-1] == ' '):
        r -= 1
    if str[l] == '(':
        l += 1
        r -= 1
    while l < r and (str[l] == ' '):
        l += 1
    while l < r and (str[r-1] == ' '):
        r -= 1
    res = []
    if str[l] == 's':
        res = sub(str[l + 3:r])
    elif str[l] == 'a':
        res = add(str[l + 3:r])
    elif str[l] == 'm':
        res = mul(str[l + 3:r])
    elif str[l] == 'd':
        res = div(str[l + 3:r])
    elif str[l] == 'X':
        r = l + 1
        print(str[l:r])
        while r<len(str) and str[r].isdigit():
            r += 1
        res = str[l:r]

    xx = "".join(res)
    print(str[l], "OUT", xx)
    return res

def simply(symbol):
    

def symbol_covert(symbol):
    res = find_symbol(list(symbol))
    res=simply(res)
    return res


print(symbol_covert(symbol))
