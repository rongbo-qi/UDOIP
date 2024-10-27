import math
import numpy as np
import json
from datetime import datetime, timedelta

class JsonData:
    def __init__(self, dataset_file, poi_trans_centers_file, center_positions_file):
        self.dataset_file = dataset_file
        self.dataset = None

        '''
            2408新增
        '''

        self.poi_trans_centers_file = poi_trans_centers_file
        self.poi_trans_centers = None

        self.center_positions_file = center_positions_file
        self.center_positions = None

    def connect(self):
        try:
            with open(self.dataset_file, 'r', encoding='utf-8') as file:
                self.dataset = json.load(file)
                #遍历进行处理
                for poi in self.dataset:
                    poi['poi_open_time'] = self.time_list_to_seconds(poi['poi_open_time_list'])
                    poi['poi_end_time'] = self.time_list_to_seconds(poi['poi_end_time_list'])
                    poi['poi_best_open_time_list'] = self.time_list_to_seconds(poi['poi_best_open_time_list'])
                    poi['poi_best_end_time_list'] = self.time_list_to_seconds(poi['poi_best_end_time_list'])
                    
            '''
                2408新增
            '''
            with open(self.poi_trans_centers_file, 'r', encoding='utf-8') as file:
                self.poi_trans_centers = json.load(file)
            with open(self.center_positions_file, 'r', encoding='utf-8') as file:
                self.center_positions = json.load(file)
        except Exception as e:
            print(e)
    
    def close(self):
        self.dataset = None

    # 定义转换时间为秒的函数
    def time_list_to_seconds(self, time_list):
        seconds_list = []
        for time_str in time_list:
            if time_str is None:
                seconds_list.append(None)
            else:
                try:
                    # 解析时间
                    time_obj = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
                    # 转换为秒
                    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
                    seconds_list.append(total_seconds)
                except ValueError:
                    # 处理可能的格式错误
                    seconds_list.append(None)
        return seconds_list

    def time_to_seconds(self, time_str):
        """将时间字符串转换为总秒数。
        
        参数:
            time_str (str): 时间字符串，格式应为 'HH:MM:SS'.
        
        返回:
            int: 时间字符串对应的总秒数.
        """
        if time_str is None:
            return None
        # 将时间字符串转换为 timedelta 对象
        time_delta = datetime.strptime(time_str, "%H:%M:%S") - datetime.strptime("00:00:00", "%H:%M:%S")
        
        # 将 timedelta 对象转换为总秒数
        return int(time_delta.total_seconds())
    
    def time_to_seconds_list(self, time_str_list):
        """将时间字符串转换为总秒数。
        
        参数:
            time_str_list (list): 时间字符串列表，格式应为 'HH:MM:SS'.
        
        返回:
            int: 时间字符串对应的总秒数.
        """
        return [self.time_to_seconds(time_str) for time_str in time_str_list]

    def getNearestPOIbyPosition(self, position, k, poi_cate):
        pois = self.dataset
        pois.sort(key=lambda poi: (poi['poi_lat'] - position[1])**2 + (poi['poi_long'] - position[0])**2)
        filtered_pois = [
                        (poi['poi_id'], 
                          poi['poi_cate'], 
                          poi['poi_name'], 
                          poi['poi_lat'], 
                          poi['poi_long'],
                          poi['poi_rec_time'],
                          poi['poi_open_time'],
                          poi['poi_end_time'],
                          poi['poi_score'],
                          poi['poi_comment_num'],
                          poi['poi_best_open_time_list'],
                          poi['poi_best_end_time_list']
                          )
                          for poi in pois if poi['poi_cate'] == poi_cate][:k]
        return np.asarray(filtered_pois, dtype=object)

    def getNearestPOIs(self, poi_id, k, poi_cate):
        pois = self.dataset
        position = self.getOnePOI(poi_id)[3:5]
        pois.sort(key=lambda poi: (poi['poi_lat'] - position[0])**2 + (poi['poi_long'] - position[1])**2)
        filtered_pois = [
                        (poi['poi_id'], 
                          poi['poi_cate'], 
                          poi['poi_name'], 
                          poi['poi_lat'], 
                          poi['poi_long'],
                          poi['poi_rec_time'],
                          poi['poi_open_time'],
                          poi['poi_end_time'],
                          poi['poi_score'],
                          poi['poi_comment_num'],
                          poi['poi_best_open_time_list'],
                          poi['poi_best_end_time_list']
                          )
                          for poi in pois if poi['poi_cate'] == poi_cate][:k]
        return np.asarray(filtered_pois, dtype=object)
    
    def getManyPOI(self, poi_ids):
        if not poi_ids:
            return np.asarray([]), None
        pois = self.dataset
        return np.asarray([
                        (poi['poi_id'], 
                          poi['poi_cate'], 
                          poi['poi_name'], 
                          poi['poi_lat'], 
                          poi['poi_long'],
                          poi['poi_rec_time'],
                          poi['poi_open_time'],
                          poi['poi_end_time'],
                          poi['poi_score'],
                          poi['poi_comment_num'],
                          poi['poi_best_open_time_list'],
                          poi['poi_best_end_time_list']
                          )
                          for poi in pois if poi['poi_id'] in poi_ids], dtype=object), None
    
    def getOnePOI(self, poi_id):
        # Iterate over the POIs to find the one with the matching poi_id
        for poi in self.dataset:
            if poi['poi_id'] == poi_id:
                # Return all values of the matched POI as a tuple
                return  (poi['poi_id'], 
                          poi['poi_cate'], 
                          poi['poi_name'], 
                          poi['poi_lat'], 
                          poi['poi_long'],
                          poi['poi_rec_time'],
                          poi['poi_open_time'],
                          poi['poi_end_time'],
                          poi['poi_score'],
                          poi['poi_comment_num'],
                          poi['poi_best_open_time_list'],
                          poi['poi_best_end_time_list']
                          )
        return None  # Return None if no match is found
    
    def getIdbyName(self, name_list, poi_cate):
        pois = self.dataset
        matched_pois = []
        for poi in pois:
            for name in name_list:
                if (name in poi['poi_name'] or poi['poi_name'] in name) and poi_cate == poi['poi_cate']:
                # if name == poi['poi_name'] and poi_cate == poi['poi_cate']:
                    matched_pois.append((
                        poi['poi_id'],
                        poi.get('poi_rec_time', None)  # Assuming poi_rec_time exists in the data
                    ))
                    break  # Assuming you want just one match per name, remove this if multiple matches are needed
        return matched_pois
    
    def getPOIIdbyCate(self, poi_cate):
        pois = self.dataset
        return np.asarray([
                        poi['poi_id']
                          for poi in pois if poi['poi_cate'] == poi_cate])
    
    '''
        2408新增
    '''

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        # Earth radius in kilometers (can be changed to miles by using 3956)
        R = 6371
        return R * c

    # Function to find the nearest cluster for given POI coordinates
    def find_clusters(self, poi_coordinates):
        cluster_labels = []
        for (lat, lon) in poi_coordinates:
            min_distance = float('inf')
            nearest_cluster = None
            for row in self.center_positions:
                distance = self.haversine_distance(lat, lon, row['poi_lat'], row['poi_long'])
                if distance < min_distance:
                    min_distance = distance
                    nearest_cluster = row['poi_cluster']
            # if lat == 21.049757 and lon == 109.14357:
            #     print(nearest_cluster)
            cluster_labels.append(nearest_cluster)
        return cluster_labels
    
    def getPOICluster(self, poi_ids):
        # poi_coordinates = [(poi['poi_lat'], poi['poi_long']) for poi in self.dataset if poi['poi_id'] in poi_ids]
        poi_dict = {poi['poi_id']: (poi['poi_lat'], poi['poi_long']) for poi in self.dataset}
        poi_coordinates = [poi_dict[poi_id] for poi_id in poi_ids]
        cluster_labels = self.find_clusters(poi_coordinates)
        return_res = []
        for poi, cluster in zip(poi_ids, cluster_labels):
            return_res.append((poi, cluster))
        return np.asarray(return_res, dtype='object')
    
    def getPOITransCenter(self, poi_centers):
        trans_centers = self.poi_trans_centers
        return np.asarray([
            (center['poi_from_center'], 
             center['poi_to_center'], 
             center['poi_center_walking_time'], 
             center['poi_center_bicycling_time'],
             center['poi_center_driving_time'],
             center['poi_center_transit_time'])
            for center in trans_centers if center['poi_from_center'] in poi_centers or center['poi_to_center'] in poi_centers
        ])
    
    def getPOIClusterCenterPositions(self, poi_centers):
        centers = self.center_positions
        return np.asarray([
            (center['poi_cluster'], center['poi_lat'], center['poi_long'])
            for center in centers if center['poi_cluster'] in poi_centers
        ])

    def getHotPOIs(self, k, poi_cate):
        pois = self.dataset
        pois.sort(key=lambda poi: poi['poi_comment_num'], reverse=True)
        filtered_pois = [
                        (poi['poi_id'], 
                          poi['poi_cate'], 
                          poi['poi_name'], 
                          poi['poi_lat'], 
                          poi['poi_long'],
                          poi['poi_rec_time'],
                          poi['poi_open_time'],
                          poi['poi_end_time'],
                          poi['poi_score'],
                          poi['poi_comment_num'],
                          poi['poi_best_open_time_list'],
                          poi['poi_best_end_time_list']
                          )
                          for poi in pois if poi['poi_cate'] == poi_cate][:k]
        return np.asarray(filtered_pois, dtype=object)