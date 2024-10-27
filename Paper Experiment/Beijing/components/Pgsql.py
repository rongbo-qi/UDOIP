import threading
import time
import numpy as np
import psycopg2
from Amap import Amap
import json
import traceback
from psycopg2 import pool

class Pgsql:
    def __init__(self, host, port, user, password, dbname, amapkey):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.dbname = dbname
        self.amap = Amap(amapkey) #'7f0f1aa885678f613708b9b1724f3334'
        self.conn = psycopg2.connect(host=self.host, port=self.port, user=self.user, password=self.password, dbname=self.dbname)
        # self.lock = threading.Lock()
        # self.pool = pool.SimpleConnectionPool(1, 20, host=self.host, port=self.port, user=self.user, password=self.password, dbname=self.dbname)

    def connect(self):
        try:
            self.conn = psycopg2.connect(host=self.host, port=self.port, user=self.user, password=self.password, dbname=self.dbname)
            self.cur = self.conn.cursor()
            # self.conn = self.pool.getconn()
            # self.cur = self.conn.cursor()
        except Exception as e:
            print(e)

    def cursor(self):
        return self.cur

    def ping(self, reconnect=True):
        try:
            self.cur.execute('SELECT 1')
            if reconnect:
                self.connect()
        except Exception as e:
            if reconnect:
                self.connect()
            else:
                print(e)

    def close(self):
        try:
            self.cur.close()
            self.conn.close()
        except Exception as e:
            print(e)

    def execute(self, sql):
        try:
            self.cur.execute(sql)
            self.conn.commit()
        except Exception as e:
            print(e)

    def fetchall(self, sql):
        try:
            self.cur.execute(sql)
            return self.cur.fetchall()
        except Exception as e:
            print(e)

    def fetchone(self, sql):
        try:
            self.cur.execute(sql)
            return self.cur.fetchone()
        except Exception as e:
            print(e)

    def fetchmany(self, sql, size):
        try:
            self.cur.execute(sql)
            return self.cur.fetchmany(size)
        except Exception as e:
            print(e)

    def getNearestPOI(self, poi_id, k, poi_cate):
        # sql = "SELECT *, ST_Distance(poi_geo, (SELECT poi_geo FROM poi_info WHERE poi_id = {})) AS distance FROM poi_info WHERE poi_id != {} AND poi_cate = {} ORDER BY distance LIMIT {}".format(poi_id, poi_id, poi_cate ,k)
        sql = """
            SELECT 
                poi_id,
                poi_cate,
                poi_name,
                poi_city,
                poi_score,
                poi_disc,
                poi_lat,
                poi_long,
                poi_rec_time,
                poi_comment_num,
                poi_hot_comment,
                poi_total_data,
                poi_geo,
                poi_address,
                ARRAY(SELECT EXTRACT(EPOCH FROM time_val)::integer FROM UNNEST(poi_open_time_list) AS time_val) AS poi_open_time_list_seconds,
                ARRAY(SELECT EXTRACT(EPOCH FROM time_val)::integer FROM UNNEST(poi_end_time_list) AS time_val) AS poi_end_time_list_seconds,
                poi_cost,
                ARRAY(SELECT EXTRACT(EPOCH FROM time_val)::integer FROM UNNEST(poi_best_open_time_list) AS time_val) AS poi_open_time_list_seconds,
                ARRAY(SELECT EXTRACT(EPOCH FROM time_val)::integer FROM UNNEST(poi_best_end_time_list) AS time_val) AS poi_end_time_list_seconds,
                ST_Distance(poi_geo, (SELECT poi_geo FROM poi_info WHERE poi_id = {})) AS distance
            FROM poi_info 
            WHERE poi_id != {} AND poi_cate = {}
            ORDER BY distance 
            LIMIT {}
        """.format(poi_id, poi_id, poi_cate, k)
        try:
            self.cur.execute(sql)
            return np.delete(np.asarray(self.cur.fetchall(), dtype=object), -1, axis=1)
        except Exception as e:
            print(e)

    def getNearestPOIbyPosition(self, position, k, poi_cate):
        # sql = "SELECT *, ST_Distance(poi_geo, (SELECT poi_geo FROM poi_info WHERE poi_id = {})) AS distance FROM poi_info WHERE poi_id != {} AND poi_cate = {} ORDER BY distance LIMIT {}".format(poi_id, poi_id, poi_cate ,k)
        sql = """
            SELECT 
                poi_id,
                poi_cate,
                poi_name,
                poi_city,
                poi_score,
                poi_disc,
                poi_lat,
                poi_long,
                poi_rec_time,
                poi_comment_num,
                poi_hot_comment,
                poi_total_data,
                poi_geo,
                poi_address,
                ARRAY(SELECT EXTRACT(EPOCH FROM time_val)::integer FROM UNNEST(poi_open_time_list) AS time_val) AS poi_open_time_list_seconds,
                ARRAY(SELECT EXTRACT(EPOCH FROM time_val)::integer FROM UNNEST(poi_end_time_list) AS time_val) AS poi_end_time_list_seconds,
                poi_cost,
                ARRAY(SELECT EXTRACT(EPOCH FROM time_val)::integer FROM UNNEST(poi_best_open_time_list) AS time_val) AS poi_open_time_list_seconds,
                ARRAY(SELECT EXTRACT(EPOCH FROM time_val)::integer FROM UNNEST(poi_best_end_time_list) AS time_val) AS poi_end_time_list_seconds,
                ST_Distance(poi_geo, ST_GeomFromText('POINT({} {})', 4326)) AS distance
            FROM poi_info 
            WHERE poi_cate = {}
            ORDER BY distance 
            LIMIT {}
        """.format(position[0], position[1], poi_cate, k)
        try:
            self.cur.execute(sql)
            return np.delete(np.asarray(self.cur.fetchall(), dtype=object), -1, axis=1)
        except Exception as e:
            print(e)

    def getOnePOI(self, poi_id):
        '''
        Name,Data type
        poi_id,integer  0
        poi_cate,integer    1
        poi_name,character varying  2
        poi_city,character varying  3
        poi_score,real  4
        poi_disc,text   5
        poi_lat,real    6
        poi_long,real   7
        poi_rec_time,bigint 8
        poi_comment_num,integer 9
        poi_hot_comment,text    10
        poi_total_data,jsonb    11
        poi_geo,geography   12
        poi_address,text    13
        poi_open_time_list,time without time zone[] 14
        poi_end_time_list,time without time zone[]  15
        '''

        sql = """
            SELECT 
                poi_id,
                poi_cate,
                poi_name,
                poi_city,
                poi_score,
                poi_disc,
                poi_lat,
                poi_long,
                poi_rec_time,
                poi_comment_num,
                poi_hot_comment,
                poi_total_data,
                poi_geo,
                poi_address,
                ARRAY(SELECT EXTRACT(EPOCH FROM time_val)::integer FROM UNNEST(poi_open_time_list) AS time_val) AS poi_open_time_list_seconds,
                ARRAY(SELECT EXTRACT(EPOCH FROM time_val)::integer FROM UNNEST(poi_end_time_list) AS time_val) AS poi_end_time_list_seconds,
                poi_cost,
                ARRAY(SELECT EXTRACT(EPOCH FROM time_val)::integer FROM UNNEST(poi_best_open_time_list) AS time_val) AS poi_open_time_list_seconds,
                ARRAY(SELECT EXTRACT(EPOCH FROM time_val)::integer FROM UNNEST(poi_best_end_time_list) AS time_val) AS poi_end_time_list_seconds
            FROM poi_info 
            WHERE poi_id = {}
        """.format(poi_id)
        try:
            self.cur.execute(sql)
            return np.asarray(self.cur.fetchone(), dtype=object)
        except Exception as e:
            print(e)

    def getPOIsTransTime(self, poi_from, poi_to):
        '''
        poi_trans_from_id   0
        poi_trans_to_id 1
        poi_trans_min_time  2
        poi_trans_max_time  3
        poi_trans_amap_data 4
        '''

        #poi_from and poi_to are the POI objects returned by getOnePOI
        #The database poi trans table is first queried for the existence of the entry
        sql = "SELECT * FROM poi_trans WHERE poi_trans_from_id = {} and poi_trans_to_id = {}".format(poi_from[0], poi_to[0])
        try:
            self.cur.execute(sql)
            # print('rowcount: ')
            # print(self.cur.rowcount)
            if self.cur.rowcount == 0:
                #Amap API is used to query the distance between the two POIs
                # distance = self.getDistance(poi_from[6], poi_from[7], poi_to[6], poi_to[7])
                way = ['步行', '驾车', '骑行', '公交']
                res = {}
                maxTime = 0
                minTime = 1e9
                output = 'JSON'
                for w in way:
                    # answer = Amap.get_direction(str(poi_from[6])+','+str(poi_from[7]), str(poi_to[6])+','+str(poi_to[7]), w)
                    origin = str(poi_from[7])+','+str(poi_from[6])
                    destination = str(poi_to[7])+','+str(poi_to[6])
                    answer = self.amap.path_planning(w, origin, destination, output, city='北海')
                    # print(answer)
                    res[w] = answer
                    try:
                        way_time = answer['route']['paths'][0]['duration']
                    except:
                        continue
                    way_time = int(way_time)
                    if way_time > maxTime:
                        maxTime = way_time
                    if way_time < minTime:
                        minTime = way_time
                #save to the database
                sql_save = 'INSERT INTO poi_trans(poi_trans_from_id, poi_trans_to_id, poi_trans_min_time, poi_trans_max_time, poi_trans_amap_data) VALUES ({}, {}, {}, {}, \'{}\')'.format(poi_from[0], poi_to[0], minTime, maxTime, json.dumps(res, ensure_ascii=False))
                self.execute(sql_save)
                self.conn.commit()
                self.cur.execute(sql)
                return self.cur.fetchone()
            else:
                return self.cur.fetchone()
        except Exception as e:
            traceback.print_exc()
            print(e)
                
    def getPOIsTransTimeAll(self, poi_from, poi_to):
        '''
        poi_trans_from_id   0
        poi_trans_to_id 1
        poi_trans_walking_time  2
        poi_trans_bicycling_time  3
        poi_trans_driving_time 4
        poi_trans_transit_time 5
        # poi_trans_amap_data 6
        '''

        #poi_from and poi_to are the POI objects returned by getOnePOI
        #The database poi trans table is first queried for the existence of the entry
        
        # start_time = time.time() # 计时开始
        # with self.lock:
        # self.ping(reconnect=True)
        sql = "SELECT * FROM poi_trans_all WHERE poi_trans_from_id = {} and poi_trans_to_id = {}".format(poi_from[0], poi_to[0])
        
        self.cur.execute(sql)
        # print('rowcount: ')
        # print(self.cur.rowcount)
        if self.cur.rowcount == 0:
            #Amap API is used to query the distance between the two POIs
            # distance = self.getDistance(poi_from[6], poi_from[7], poi_to[6], poi_to[7])
            way = ['步行', '驾车', '骑行', '公交']
            res = {}
            walking_time = 86401
            bicycling_time = 86401
            driving_time = 86401
            transit_time = 86401
            output = 'JSON'
            for w in way:
                # answer = Amap.get_direction(str(poi_from[6])+','+str(poi_from[7]), str(poi_to[6])+','+str(poi_to[7]), w)
                origin = str(poi_from[7])+','+str(poi_from[6])
                destination = str(poi_to[7])+','+str(poi_to[6])
                answer = self.amap.path_planning(w, origin, destination, output, city='北海')
                # print(answer)
                res[w] = answer
                try:
                    way_time = answer['route']['paths'][0]['duration']
                except:
                    continue
                way_time = int(way_time)
                if w == '步行':
                    walking_time = way_time
                elif w == '驾车':
                    driving_time = way_time
                elif w == '骑行':
                    bicycling_time = way_time
                elif w == '公交':
                    transit_time = way_time
            #save to the database
            sql_save = 'INSERT INTO poi_trans_all(poi_trans_from_id, poi_trans_to_id, poi_trans_walking_time, poi_trans_bicycling_time, poi_trans_driving_time, poi_trans_transit_time) VALUES ({}, {}, {}, {}, {}, {})'.format(poi_from[0], poi_to[0], walking_time, bicycling_time, driving_time, transit_time)
            self.execute(sql_save)
            sql_save1 = 'INSERT INTO poi_trans_amap_data(poi_trans_from_id, poi_trans_to_id, poi_trans_amap_data) VALUES ({}, {}, \'{}\')'.format(poi_from[0], poi_to[0], json.dumps(res, ensure_ascii=False))
            self.execute(sql_save1)
            self.conn.commit()
            self.cur.execute(sql)       
            # end_time = time.time() # 计时结束
            # print('getPOIsTransTimeAll time:', end_time-start_time)
            return self.cur.fetchone()
        else:
            # end_time = time.time() # 计时结束
            # print('getPOIsTransTimeAll time:', end_time-start_time)
            return np.asarray(self.cur.fetchone(), dtype=object)
            
    def getPOIsTransAll(self, poi_from, poi_to):
        '''
        poi_trans_from_id   0
        poi_trans_to_id 1
        poi_trans_walking_time  2
        poi_trans_bicycling_time  3
        poi_trans_driving_time 4
        poi_trans_transit_time 5
        poi_trans_amap_data 6
        '''

        #poi_from and poi_to are the POI objects returned by getOnePOI
        #The database poi trans table is first queried for the existence of the entry
        
        # start_time = time.time() # 计时开始
        # with self.lock:
        sql = "SELECT * FROM poi_trans_all WHERE poi_trans_from_id = {} and poi_trans_to_id = {}".format(poi_from[0], poi_to[0])
        self.cur.execute(sql)
        return self.cur.fetchone()

    def getHotPOIs(self, k, poi_cate):
        sql = '''
        SELECT 
                poi_id,
                poi_cate,
                poi_name,
                poi_city,
                poi_score,
                poi_disc,
                poi_lat,
                poi_long,
                poi_rec_time,
                poi_comment_num,
                poi_hot_comment,
                poi_total_data,
                poi_geo,
                poi_address,
                ARRAY(SELECT EXTRACT(EPOCH FROM time_val)::integer FROM UNNEST(poi_open_time_list) AS time_val) AS poi_open_time_list_seconds,
                ARRAY(SELECT EXTRACT(EPOCH FROM time_val)::integer FROM UNNEST(poi_end_time_list) AS time_val) AS poi_end_time_list_seconds,
                poi_cost,
                ARRAY(SELECT EXTRACT(EPOCH FROM time_val)::integer FROM UNNEST(poi_best_open_time_list) AS time_val) AS poi_open_time_list_seconds,
                ARRAY(SELECT EXTRACT(EPOCH FROM time_val)::integer FROM UNNEST(poi_best_end_time_list) AS time_val) AS poi_end_time_list_seconds
         FROM poi_info WHERE poi_cate = {} ORDER BY poi_comment_num DESC LIMIT {}
        '''.format(poi_cate, k)
        try:
            self.cur.execute(sql)
            return np.asarray(self.cur.fetchall(), dtype=object)
        except Exception as e:
            print(e)

    #分页查询POI
    def getPOIsbyPage(self, page, size, poi_cate):
        # self.ping(reconnect=True)
        cur = self.conn.cursor()
        sql = "SELECT * FROM poi_info WHERE poi_cate = {} ORDER BY poi_id LIMIT {} OFFSET {}".format(poi_cate, size, (page-1)*size)
        try:
            cur.execute(sql)
            results = cur.fetchall()
            # if results is None:
            #     print('No data')
            # if cur.rowcount == 0:
            #     print('No data')
            cur.close()
            return results, cur
        except Exception as e:
            print(e)

    def userRegister(self, username, password):
        #query count of the user_id
        sql = "SELECT COUNT(*) FROM user_info"
        self.cur.execute(sql)
        count = self.cur.fetchone()[0]
        user_id = count + 1
        #query if the username exists
        sql = "SELECT COUNT(*) FROM user_info WHERE user_name = '{}'".format(username)
        self.cur.execute(sql)
        count = self.cur.fetchone()[0]
        if count > 0:
            return False
        sql = "INSERT INTO user_info(user_id, user_name, user_password) VALUES ({}, '{}', '{}')".format(user_id, username, password)
        try:
            self.cur.execute(sql)
            self.conn.commit()
            return True
        except Exception as e:
            print(e)

    def userLogin(self, username, password):
        sql = "SELECT * FROM user_info WHERE user_name = '{}' AND user_password = '{}'".format(username, password)
        try:
            self.cur.execute(sql)
            return self.cur.fetchone(), self.cur
        except Exception as e:
            print(e)
    
    def savePlan(self, user_id, plan_result, plan_demand, route_start_poi_time, route_wait_poi_time, route_trans_poi_time, route_total_score, route_demand_satisfied):
        cur = self.conn.cursor()
        sql_insert_route_info = "INSERT INTO route_info(route_id, route_demand, route_create_time, route_demand_satisfied) VALUES ({}, \'{}\', {}, \'{}\') RETURNING route_id"
        route_info_count = "SELECT COUNT(*) FROM route_info"
        cur.execute(route_info_count)
        count = cur.fetchone()[0]
        route_id = count + 1
        current_timestamp = int(time.time())
        sql_insert_route_info = sql_insert_route_info.format(route_id, json.dumps(plan_demand, ensure_ascii=False), current_timestamp, json.dumps(route_demand_satisfied, ensure_ascii=False))
        cur.execute(sql_insert_route_info)
        route_id = cur.fetchone()[0]
        sql_insert_route_path = "INSERT INTO route_path(route_id, route_sub_day, route_sub_path, route_sub_start_time, route_sub_wait_time, route_sub_trans_time, route_sub_total_score) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        for i in range(len(plan_result)):
            # sql_insert_route_path = sql_insert_route_path.format(route_id, i+1, plan_result[i], route_start_poi_time[i])
            # try:
            cur.execute(sql_insert_route_path, (route_id, i+1, plan_result[i], route_start_poi_time[i], route_wait_poi_time[i], route_trans_poi_time[i], route_total_score[i]))
            # except Exception as e:
            #     print(e)
            #     print((route_id, i+1, plan_result[i], route_start_poi_time[i], route_wait_poi_time[i], route_trans_poi_time[i], route_total_score[i]))
        # self.conn.commit()
        sql_insert_user_route = "INSERT INTO user_route(user_id, route_id) VALUES ({}, {})".format(user_id, route_id)
        cur.execute(sql_insert_user_route)
        self.conn.commit()
        return route_id
    
    def getPlan(self, user_id):
        cur = self.conn.cursor()
        sql = '''
            SELECT user_id, route_path.*, route_info.route_demand, route_info.route_create_time, route_info.route_demand_satisfied FROM user_route right join route_path on user_route.route_id = route_path.route_id left join route_info on route_path.route_id = route_info.route_id WHERE user_id = {}
        '''.format(user_id)
        cur.execute(sql)
        result = cur.fetchall()
        cur.close()
        return result, cur
    
    def getPlanPOIs(self, user_id):
        cur = self.conn.cursor()
        sql = '''
            SELECT distinct poi_info.* 
            FROM (SELECT user_id, route_path.*, route_info.route_demand, unnest(route_path.route_sub_path) as path 
            FROM user_route right join route_path on user_route.route_id = route_path.route_id left join route_info on route_path.route_id = route_info.route_id 
            WHERE user_id = {}) join poi_info on poi_info.poi_id = path
        '''.format(user_id)
        cur.execute(sql)
        result = cur.fetchall()
        cur.close()
        return result, cur
    
    def getPOI(self, poi_id):
        cur = self.conn.cursor()
        sql = "SELECT * FROM poi_info WHERE poi_id = {}".format(poi_id)
        cur.execute(sql)
        return cur.fetchone(), cur
        
    #批量查询POI_cluster
    def getPOICluster(self, poi_ids):
        sql = "SELECT * FROM poi_cluster WHERE poi_id = ANY(%s)"
        self.cur.execute(sql, (poi_ids,))
        return np.asarray(self.cur.fetchall(), dtype=object)
    
    #批量查询poi_trans_centers
    def getPOITransCenter(self, poi_centers):
        sql = "SELECT * FROM poi_trans_centers WHERE poi_from_center = ANY(%s) or poi_to_center = ANY(%s)"
        self.cur.execute(sql, (poi_centers, poi_centers))
        return np.asarray(self.cur.fetchall(), dtype=object)
    
    #批量查询聚簇的聚类中心的POI_id的经纬度
    def getPOIClusterCenterPositions(self, poi_centers):
        sql = '''
        SELECT pcs.poi_cluster, poi_info.poi_lat, poi_info.poi_long
        FROM (SELECT * FROM poi_cluster WHERE cluster_center = TRUE) AS pcs LEFT JOIN poi_info ON pcs.poi_id = poi_info.poi_id
        WHERE pcs.poi_cluster = ANY(%s)
        '''
        self.cur.execute(sql, (poi_centers,))
        return np.asarray(self.cur.fetchall(), dtype=object)
    
    def getManyPOI(self, poi_ids):
        sql = '''
        SELECT 
            poi_id,
            poi_cate,
            poi_name,
            poi_city,
            poi_score,
            poi_disc,
            poi_lat,
            poi_long,
            poi_rec_time,
            poi_comment_num,
            poi_hot_comment,
            poi_total_data,
            poi_geo,
            poi_address,
            ARRAY(SELECT EXTRACT(EPOCH FROM time_val)::integer FROM UNNEST(poi_open_time_list) AS time_val) AS poi_open_time_list_seconds,
            ARRAY(SELECT EXTRACT(EPOCH FROM time_val)::integer FROM UNNEST(poi_end_time_list) AS time_val) AS poi_end_time_list_seconds,
            poi_cost,
            ARRAY(SELECT EXTRACT(EPOCH FROM time_val)::integer FROM UNNEST(poi_best_open_time_list) AS time_val) AS poi_open_time_list_seconds,
            ARRAY(SELECT EXTRACT(EPOCH FROM time_val)::integer FROM UNNEST(poi_best_end_time_list) AS time_val) AS poi_end_time_list_seconds
        FROM poi_info WHERE poi_id = ANY(%s)
        '''
        self.cur.execute(sql, (poi_ids,))
        return np.asarray(self.cur.fetchall(), dtype=object), self.cur
    
    def getIdbyName(self, name_list_str, poi_cate):
        sql = '''
            WITH poi_names AS (
                SELECT unnest(string_to_array('{}', ',')) AS name
            )
            SELECT poi_id, poi_rec_time
            FROM poi_info, poi_names
            WHERE poi_name ILIKE '%' || name || '%' and poi_cate = {};
        '''.format(name_list_str, poi_cate)
        self.cur.execute(sql)
        return self.cur.fetchall()

