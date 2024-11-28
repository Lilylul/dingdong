# num_list=[1, 2, 3, 4, 5, 6, 7, 8]
# n=2
# # x=len(num_list)/n
# # print(x)


# y =[]
# lnl=len(num_list)//n
# for i in range(0,lnl):
#     y.append(num_list[(i*n):(i+1)*n])

# # print(y)
# i=1
# j =13
# k=1
# cnt=0

# for n in range(i,j+1):
#     for num in list(str(n)):
#         if num in str(k):
#             cnt+=1
# print(cnt)

# array=[7, 77, 17]
# total=0
# for i in array:
#     total +=str(i).count('7')
# print(total)
# lst=[]
# array = [10, 15, 50, 40, 20]
# array.sort()
# num = 0
# n = 30
# for i in array:
#     if i>n:
#         num = i-n
#         lst.append(num)
#     else:
#         num = n-i
#         lst.append(num)
# answer = array[lst.index(min(lst))]
# print(answer)
# emergency=[30, 10, 23, 6, 100]
# emergency_sort = sorted(emergency,reverse=True)
# emergency_dict = {}
# answer = []
# print(emergency_sort)
# for i in emergency:
#     answer.append(emergency_sort.index(i)+1)
# print(answer)

# 8. 2진수를 5진수와 10진수로 바꾸는 함수를 작성하시오.
# 10진수가 주어지면 2진수와 5진수로 바꿀수 있는 한꺼번에 작동할 수 있는 함수를 작성하시오. 
# 2진수를 10진수와 5진수로 변환하는 함수

def solution():
    n = 0 
    de_number = 0
    x =int(input('2진수를 입력해주세요:'))
    x_lst = list(str(x))

    for i in range(len(x_lst)-1,-1,-1):
        de_number +=int(x_lst[n])*(2**i)
        n +=1
    
    def dec_to_base5(number):
        if number == 0:
            return '0'
        base5 = ''
        while number > 0:
            base5 = str(number % 5) + base5
            number //= 5
        return base5

    base5_number = dec_to_base5(de_number)
    print(f'{x}의 10진수는 {de_number}이고, 5진수는 {base5_number}입니다.')

solution()


# solution()
# my_string = "aAb1B2cC34oOp"
# my_num = 0
# new_my_string = ''

# for t in my_string:
#     try:
#         int(t)
#         new_my_string +=t
#     except ValueError:
#         new_my_string +=(','+t+',')
# new = new_my_string.split(',')

# for num in new:
#     try:
#         int(num)
#         my_num += int(num)
#     except ValueError:
#         continue
# print(my_num)

def text_cnt(): #문자열 단어수와 각 단어의 문자수를 계산하는 함수 
    my_text =input("문자열을 입력하세요: ").strip() #사용자 입력을 받아 양쪽 공백을 제거
    cnt1 =0 #단어 개수를 저장할 변수 초기화
    cnt2 = [] #각 단어의 문자를 저장할 리스트 초기화

    if not my_text: #my_text에 입력된 값이 없으면,
        print("입력된 값이 없습니다.") #입력된 값이 없다를 출력
        return
    else: #빈 문자열이 아닌경우
        text =my_text.split() #공백을 기준으로 단어를 분리하여 리스트에 저장
        cnt1 =len(text) #text 리스트에 담긴 단어의 개수를 센다.
        cnt2 = [len(t) for t in text] #text에 담긴 글자를 1개씩 꺼내서 센다.
    print(f"단어는 총 {cnt1}개 이며, 각 단어의 문자 개수는 {cnt2}입니다.")

text_cnt() #함수 실행