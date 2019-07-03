import requests
import json

# url = "http://clp.free.idcfengye.com/jacky"
# method = "/test"
#
# results = requests.get(url+method)
# json_results = json.loads(results.text)
#
# # print(results.text)
# # print(json_results)
#
# status = json_results['status']
# message = json_results['message']
# success = json_results['success']
# print(' status: ',status)
# print('message: ',message)
# print('success: ',success)



#login
url = "http://clp.free.idcfengye.com/jacky/login"
login_data = {"username":"jacky","password":"fuckyouironman"}
json_login = json.dumps(login_data)

# print(json_login,type(json_login))

res = requests.post(url,json_login)
json_res = json.loads(res.text)
# json_res = res.text

# print(res)
print(json_res)

