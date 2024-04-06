import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_random_exponential,
)  # for exponential backoff


# @retry(wait=wait_random_exponential(min=1, max=60), retry=retry_if_exception_type((openai.error.RateLimitError, openai.error.APIError)))
def collect_one(prompt, api_key, sample_num=1):
    openai.api_key = api_key
    ret = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        max_tokens=500, # 最大prompt长度
        temperature=0.8, # 温度越高产生结果越多样
        n=sample_num, # 输出结果数量
        top_p=0.95 # 对前5%的结果采样
    )

    samples = ret['choices']
    candidates = []
    for id, i in enumerate(samples):
        candi_lst = i['message']['content'].strip().split("\n")
        candi = ""
        for snippet in candi_lst:
            if snippet.endswith("END_OF_CASE"):
                break
            else:
                candi += snippet + "\n"
        candi = candi.strip()
        if candi != "":
            candidates.append(candi)
    return candidates


if __name__ == "__main__":
    requirement = "write a binary search function in Python."
    API_KEY = ""
    messages = [{"role": "system", "content": "You are a programmer proficient in python"}]
    messages += [{"role": "user", "content": "write a bubble sort function in Python."}]
    messages += {"role": "assistant", "content": """
def bubbleSort(arr):
    n = len(arr)
    for i in range(n-1):
        swapped = False
        for j in range(0, n-i-1):
            if arr[j] > arr[j + 1]:
                swapped = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
        if not swapped:
            return
    END_OF_CASE"""},
    messages += [{"role": "user", "content": requirement}]
    print(collect_one(messages, API_KEY)[0])
