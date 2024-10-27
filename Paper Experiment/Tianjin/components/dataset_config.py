# import Pgsql as pg


# class DatasetConfig:
#     def __init__(self):
#         # Pgsql
#         self.poi_dict = {
#             'poi_id': 0,  # POI编号 *
#             'poi_cate': 1,  # POI类型 *
#             'poi_name': 2,  # POI名称
#             'poi_city': 3,  # POI所在城市
#             'poi_score': 4,  # POI评分 *
#             'poi_desc': 5,  # POI描述
#             'poi_lat': 6,  # POI纬度 *
#             'poi_long': 7,  # POI经度 *
#             'poi_rec_time': 8,  # POI建议游玩时长 *
#             'poi_comment_num': 9,  # POI评论数 *
#             'poi_hot_comment': 10,  # POI热门评论
#             'poi_total_data': 11,  # POI总数据
#             'poi_geo': 12,  # POI地理位置
#             'poi_address': 13,  # POI地址
#             'poi_open_time_list': 14,  # POI开放时间list *
#             'poi_end_time_list': 15,  # POI结束时间list *
#             'poi_cost': 16,  # POI花费
#             'poi_best_open_time_list': 17,
#             'poi_best_end_time_list': 18,
#             'poi_cluster': {
#                 'poi_id': 0,
#                 'poi_cluster': 1
#             },
#             'poi_cluster_position': {
#                 'poi_cluster': 0,
#                 'poi_center_lat': 1,
#                 'poi_center_long': 2
#             },
#             'poi_trans_center': {
#                 'poi_from_center': 0,
#                 'poi_to_center': 1,
#                 'poi_center_walking_time': 2,  # 步行时间 *
#                 'poi_center_bicycling_time': 3,  # 骑行时间 *
#                 'poi_center_driving_time': 4,  # 驾车时间 *
#                 'poi_center_transit_time': 5,  # 公交时间 *
#             }
#         }

#         self.poi_trans_dict = {
#             'poi_trans_from_id': 0,  # 起点编号
#             'poi_trans_to_id': 1,  # 终点编号
#             'poi_trans_walking_time': 2,  # 步行时间 *
#             'poi_trans_bicycling_time': 3,  # 骑行时间 *
#             'poi_trans_driving_time': 4,  # 驾车时间 *
#             'poi_trans_transit_time': 5,  # 公交时间 *
#             'poi_trans_amap_data': 6  # 高德地图数据
#         }

#         self.poi_cate_dict = {
#             'attraction': 0,  # 景点
#             'restaurant': 1,  # 餐厅
#             'hotel': 2  # 酒店
#         }

#         self.poi_cate_dict_name = self.poi_cate_dict

#         # ip port user password database amap_key
#         self.database = pg.Pgsql('localhost', '5432', 'postgres', '123456', 'tianjin_routes',
#                                  '7f0f1aa885678f613708b9b1724f3334')

#         self.database_func = {
#             'getNearestPOIs': self.database.getNearestPOI,
#             'getNearestPOIsbyPosition': self.database.getNearestPOIbyPosition,
#             'getOnePOI': self.database.getOnePOI,
#             'getPOIsTransTimeAll': self.database.getPOIsTransTimeAll,
#             'getHotPOIs': self.database.getHotPOIs,
#             'getPOICluster': self.database.getPOICluster,
#             'getPOITransCenter': self.database.getPOITransCenter,
#             'getPOIClusterCenterPositions': self.database.getPOIClusterCenterPositions,
#             'getManyPOI': self.database.getManyPOI,
#             'getIdbyName': self.database.getIdbyName
#         }

#         self.pre_set_dict = {
#             0: {
#                 'description': '热门方案',
#                 'N_c_min': [0, 2, 2],
#                 'N_c_max': [10, 3, 4],
#                 'alpha': 0.5,
#                 'maxIterations': 3,
#                 'RCLsize': 5,
#                 'poi_id_list': [8, 14, 6, 10],
#                 'm': 3,
#                 'start_end_poi': 4860,
#                 'start_day_time': '09:00:00',
#                 'plan_max_time': 12,
#                 'pre_set_poi': [8, 14, 6, 10]
#             },
#             1: {
#                 'description': '亲子方案',
#                 'N_c_min': [0, 2, 2],
#                 'N_c_max': [10, 3, 4],
#                 'alpha': 0.5,
#                 'maxIterations': 3,
#                 'RCLsize': 5,
#                 'poi_id_list': [1, 3, 6],
#                 'm': 3,
#                 'start_end_poi': 4860,
#                 'start_day_time': '10:00:00',
#                 'plan_max_time': 10,
#                 'pre_set_poi': [10, 3, 4]
#             },
#             8776: {
#                 'description': '其他方案',
#                 'N_c_min': [0, 2, 2],
#                 'N_c_max': [10, 3, 4],
#                 'alpha': 0.5,
#                 'maxIterations': 3,
#                 'RCLsize': 5,
#                 'poi_id_list': [14],
#                 'm': 1,
#                 'start_end_poi': 4860,
#                 'start_day_time': '08:00:00',
#                 'plan_max_time': 14,
#                 'pre_set_poi': [12, 43, 14]
#             }
#         }


import Json as js

class DatasetConfig:
    def __init__(self):
        self.poi_dict = {
            'poi_id': 0,  # POI编号 *
            'poi_cate': 1,  # POI类型 *
            'poi_name': 2,  # POI名称 *
            'poi_lat': 3,  # POI纬度 *
            'poi_long': 4,  # POI经度 *
            'poi_rec_time': 5,  # POI建议游玩时长 *
            'poi_open_time_list': 6,  # POI开放时间list *
            'poi_end_time_list': 7,  # POI结束时间list *
            'poi_score': 8,  # POI评分 *
            'poi_comment_num': 9,  # POI评论数 *
            'poi_best_open_time_list': 10,
            'poi_best_end_time_list': 11,

            'poi_cluster': {
                'poi_id': 0,
                'poi_cluster': 1
            },
            'poi_cluster_position': {
                'poi_cluster': 0,
                'poi_center_lat': 1,
                'poi_center_long': 2
            },
            'poi_trans_center': {
                'poi_from_center': 0,
                'poi_to_center': 1,
                'poi_center_walking_time': 2,  # 步行时间 *
                'poi_center_bicycling_time': 3,  # 骑行时间 *
                'poi_center_driving_time': 4,  # 驾车时间 *
                'poi_center_transit_time': 5,  # 公交时间 *
            }
        }

        self.poi_trans_dict = {

        }

        self.poi_cate_dict = {
            'attraction': '0',  # 景点
            'restaurant': '1',  # 餐厅
            'hotel': '2'  # 酒店
        }

        self.poi_cate_dict_name = self.poi_cate_dict

        
        self.database = js.JsonData('../data/tianjin_attractions_data.json','../data/tianjin_transit_data(gpt4).json','../data/tianjin_cluster_center_data.json')

        self.database_func = {
            'getNearestPOIsbyPosition': self.database.getNearestPOIbyPosition,
            'getNearestPOIs': self.database.getNearestPOIs,
            'getIdbyName': self.database.getIdbyName,
            'getManyPOI': self.database.getManyPOI,
            'getOnePOI': self.database.getOnePOI,
            'getPOIIdbyCate': self.database.getPOIIdbyCate,
            'getHotPOIs': self.database.getHotPOIs,
            'getPOICluster': self.database.getPOICluster,
            'getPOITransCenter': self.database.getPOITransCenter,
            'getPOIClusterCenterPositions': self.database.getPOIClusterCenterPositions
        }

        self.pre_set_dict = {

        }

