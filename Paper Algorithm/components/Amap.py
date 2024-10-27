import requests
import json

class Amap:
    def __init__(self, key):
        self.key = key

    def get_location(self, address):
        url = 'https://restapi.amap.com/v3/geocode/geo?address={}&output=JSON&key={}'.format(address, self.key)
        response = requests.get(url)
        answer = json.loads(response.text)
        if answer['status'] == '1':
            return answer
        else:
            return None
    
    def get_direction(self, origin, destination, way):
        # way: walking, driving, bicycling, transit
        # origin, destination: lon,lat
        url = 'https://restapi.amap.com/v3/direction/{}?origin={}&destination={}&output=JSON&key={}'.format(way, origin, destination, self.key)
        response = requests.get(url)
        if response.status_code == 200:
            answer = json.loads(response.text)
            if answer['status'] == '1':
                return answer
            else:
                return None

    # 定义一个函数，接受路径规划的类型（步行、公交、驾车、骑行）、起点、终点、输出格式和Key作为参数
    def path_planning(self, type, origin, destination, output, city):
        
        # 构造请求参数
        params = {
            "origin": origin, # 起点坐标，经度和纬度用","分隔
            "destination": destination, # 终点坐标，经度和纬度用","分隔
            "output": output, # 输出格式，可选JSON或XML
            "key": self.key # 用户申请的Key
        }

        # 根据类型选择不同的API地址
        if type == "步行":
            url = "https://restapi.amap.com/v3/direction/walking"
        elif type == "公交":
            url = "https://restapi.amap.com/v3/direction/transit/integrated"
            #添加城市参数
            params["city"] = city
        elif type == "驾车":
            url = "https://restapi.amap.com/v3/direction/driving"
            params["extensions"] = "base"
        elif type == "骑行":
            url = "https://restapi.amap.com/v4/direction/bicycling"
        else:
            return "无效的类型，请输入步行、公交、驾车或骑行"

        # 发送HTTP请求，获取响应数据
        response = requests.get(url, params=params)

        # 根据输出格式，解析数据
        if output == "JSON":
            data = response.json() # 将响应数据转换为JSON对象
            if type == "公交":
                # add data['route']['paths'][0]['duration'] = data['route']['transits'][0]['duration']
                #add paths key
                try:
                    data['route']['paths'] = data['route']['transits']
                except:
                    print(params)
                # data['route']['paths'][0]['duration'] = data['route']['transits'][0]['duration']
            elif type == "骑行":
                # add data['route'] = data['data']
                data['route'] = data['data']
                
        elif output == "XML":
            data = response.text # 将响应数据转换为XML字符串
        else:
            return "无效的输出格式，请输入JSON或XML"

        # 返回数据
        return data